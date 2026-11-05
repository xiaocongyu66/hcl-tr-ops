"""TileLang / FlashQLA 在 T4(SM75) 上承接 GDN-2 chunkwise 的可行性探测。

设计约束：
  * 不静默回退。依赖缺失、编译失败、数值不符都写入 blockers 并如实返回。
  * 参考实现独立于 gdn2_ref：逐 token fp64 递推，便于发现移植中的公式错误。
  * 本模块的 CPU 部分可被单元测试覆盖；GPU 部分只能由真实 GPU 运行证明。
"""

from __future__ import annotations

import importlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

DEFAULT_TOLERANCE = dict(rtol=2e-3, atol=2e-4)


def _version(module_name: str) -> str | None:
    try:
        return getattr(importlib.import_module(module_name), "__version__", "unknown")
    except Exception:
        return None


def dist_info(dist_name: str) -> dict:
    """按发行版名查询版本与顶层模块；导入名与包名经常不同，不能靠猜。"""
    try:
        from importlib.metadata import distribution
    except Exception:  # noqa: BLE001
        return dict(installed=False, version=None, top_level=[])
    try:
        dist = distribution(dist_name)
    except Exception:  # noqa: BLE001
        return dict(installed=False, version=None, top_level=[])
    top_level = []
    try:
        text = dist.read_text("top_level.txt") or ""
        top_level = [line.strip() for line in text.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        pass
    if not top_level:
        names = set()
        for f in dist.files or []:
            parts = str(f).split("/")
            if parts and parts[0].endswith(".py"):
                names.add(parts[0][:-3])
            elif len(parts) > 1 and not parts[0].endswith((".dist-info", ".data")):
                names.add(parts[0])
        top_level = sorted(names)
    return dict(installed=True, version=dist.version, top_level=top_level)


def environment() -> dict:
    """记录影响结论的软硬件事实；GPU 信息缺失时明确标记。"""
    info = dict(
        torch=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=torch.version.cuda,
        triton=_version("triton"),
        tilelang_dist=dist_info("tilelang"),
        flash_qla_dist=dist_info("flash-qla"),
    )
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info.update(device=props.name, capability=list(torch.cuda.get_device_capability(0)),
                    total_memory_bytes=props.total_memory)
    else:
        info.update(device=None, capability=None, total_memory_bytes=None)
    return info


def environment_blockers(info: dict, *, require_capability=None) -> list[str]:
    blockers = []
    if not info["cuda_available"]:
        blockers.append("无 CUDA 设备：无法验证 TileLang 编译与 GPU 计时")
    elif require_capability is not None and tuple(info["capability"]) != tuple(require_capability):
        blockers.append(
            f"设备 capability={info['capability']}，与目标 {list(require_capability)} 不符")
    return blockers


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def reference_recurrence(q, k, v, g, b, w, *, divide_by_sqrt_t=True, l2norm_qk=True):
    """独立的逐 token fp64 递推，作为设备无关的真值。

    与 gdn2_ref 的批量实现刻意不复用代码，避免同一处公式错误在两边同时出现。
    """
    q, k, v, g, b, w = (t.double() for t in (q, k, v, g, b, w))
    if l2norm_qk:
        q, k = _l2norm(q), _l2norm(k)
    B, T, H, dk = q.shape
    # 设备必须跟随输入；否则 GPU 张量与 CPU 参考会在 einsum 处直接报错。
    S = torch.zeros(B, H, dk, v.shape[-1], dtype=torch.float64, device=q.device)
    outs = []
    for t in range(T):
        S_bar = g[:, t].exp().unsqueeze(-1) * S
        r = torch.einsum("bhk,bhkv->bhv", b[:, t] * k[:, t], S_bar)
        S = S_bar + k[:, t].unsqueeze(-1) * (w[:, t] * v[:, t] - r).unsqueeze(-2)
        o = torch.einsum("bhk,bhkv->bhv", q[:, t], S)
        if divide_by_sqrt_t:
            o = o / math.sqrt(t + 1)
        outs.append(o)
    return torch.stack(outs, dim=1)


def compare_backend(backend, tensors: dict, *, chunk: int, tolerance=None) -> dict:
    """对同一输入比较 backend 与 fp64 真值；容差失败不抛异常，交由调用方判定。"""
    tolerance = dict(DEFAULT_TOLERANCE if tolerance is None else tolerance)
    expected = reference_recurrence(
        tensors["q"], tensors["k"], tensors["v"], tensors["g"], tensors["b"], tensors["w"])
    actual = backend(**tensors, chunk=chunk)
    if actual.shape != expected.shape:
        return dict(ok=False, reason=f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    diff = (actual.double() - expected).abs()
    finite = bool(torch.isfinite(actual).all())
    scale = expected.abs().clamp_min(1e-12)
    ok = finite and bool(torch.all(diff <= tolerance["atol"] + tolerance["rtol"] * scale))
    return dict(ok=ok, finite=finite, max_abs=float(diff.max()) if diff.numel() else 0.0,
                mean_abs=float(diff.mean()) if diff.numel() else 0.0,
                rtol=tolerance["rtol"], atol=tolerance["atol"],
                reason="" if ok else "超出数值容差或出现非有限值")


def make_tensors(*, device="cpu", batch=2, tokens=16, heads=3, dk=8, dv=8, seed=0,
                 requires_grad=False, realistic_decay=False):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (batch, tokens, heads, dk)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    v = torch.randn((batch, tokens, heads, dv), generator=gen)
    if realistic_decay:
        a_scale = torch.empty((heads,), dtype=torch.float32).uniform_(1.0, 16.0, generator=gen)
        dt = torch.empty((heads, dk), dtype=torch.float32).uniform_(1e-3, 1e-1, generator=gen)
        dt_bias = dt + torch.log(-torch.expm1(-dt))
        pre = torch.randn(shape, generator=gen) * 0.04
        g = -a_scale.view(1, 1, heads, 1) * torch.nn.functional.softplus(
            pre + dt_bias.view(1, 1, heads, dk))
    else:
        g = -torch.rand(shape, generator=gen)
    b = torch.rand(shape, generator=gen)
    w = torch.rand((batch, tokens, heads, dv), generator=gen)
    out = dict(q=q, k=k, v=v, g=g, b=b, w=w)
    return {name: t.to(device).requires_grad_(requires_grad) for name, t in out.items()}


def time_callable(fn, *, warmup=3, reps=10) -> dict:
    """CUDA 事件计时；无 CUDA 时退化为墙钟并标注来源。"""
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        if use_cuda:
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
    return dict(source="cuda_event_ms" if use_cuda else "wall_ms", samples=samples,
                median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), reps=reps)


def _tilelang_smoke_compile() -> dict:
    """在真实设备上编译并运行一个最小 kernel，验证 SM75 目标可用。

    TileLang 各版本 API 有差异，失败时保留完整异常原文，供下一轮修正；
    这里绝不把"能导入"当作"能编译"。
    """
    try:
        import tilelang
        import tilelang.language as T
    except Exception as exc:  # noqa: BLE001 - 需要把完整原因写入报告
        return dict(ok=False, stage="import", error=f"{type(exc).__name__}: {exc}")
    version = getattr(tilelang, "__version__", "unknown")
    if not torch.cuda.is_available():
        return dict(ok=False, stage="device", version=version, error="需要 CUDA 才能编译")

    n, block = 1024, 128

    try:
        @tilelang.jit
        def add_kernel():
            @T.prim_func
            def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
                     C: T.Tensor((n,), "float32")):
                with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
                    for i in T.Parallel(block):
                        idx = bx * block + i
                        C[idx] = A[idx] + B[idx]

            return main

        kernel = add_kernel()
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        out = torch.empty(n, device="cuda")
        kernel(a, b, out)
        torch.cuda.synchronize()
        exact = bool(torch.equal(out, a + b))
        return dict(ok=exact, stage="run", version=version,
                    error="" if exact else "kernel 输出与 torch 加法不一致")
    except Exception as exc:  # noqa: BLE001
        return dict(ok=False, stage="compile", version=version,
                    error=f"{type(exc).__name__}: {exc}")


def discover_flash_qla(keywords=("gdn", "chunk", "delta", "fused")) -> dict:
    """发现 FlashQLA 的导入入口与疑似 GDN 算子名。

    只做列举，不猜测签名去调用；真正接线要等看到实际 API 后再做。
    """
    info = dist_info("flash-qla")
    report = dict(dist=info, modules={}, candidates=[])
    for module_name in info["top_level"]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            report["modules"][module_name] = f"{type(exc).__name__}: {exc}"
            continue
        names = [n for n in dir(module)
                 if any(k in n.lower() for k in keywords)
                 and callable(getattr(module, n, None))]
        report["modules"][module_name] = dict(ok=True, candidates=names[:20])
        report["candidates"].extend(f"{module_name}.{n}" for n in names[:20])
    return report


def profile_pytorch_breakdown(*, tokens=8192, chunk=512, heads=10, dk=64, dv=64,
                              reps=3, device="cuda") -> dict:
    """把 chunkwise 路径拆成阶段计时，定位该先优化哪一段。

    与 gdn2_ref.gdn2_chunkwise_batched_forward 的公式一致，但只用于计时，
    不参与正确性判定；任何数值问题都由 compare_backend 单独检查。
    """
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv, seed=1)
    q, k, v, g, b, w = (x.float() for x in (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"]))
    l2 = _l2norm
    n = (tokens + chunk - 1) // chunk
    pad = n * chunk - tokens
    tril_lt = torch.ones(chunk, chunk, device=device, dtype=torch.bool).tril(-1)
    tril_le = torch.ones(chunk, chunk, device=device, dtype=torch.bool).tril()

    def prep():
        K = l2(k)
        Q = l2(q)
        def window(x):
            part = x if not pad else torch.cat(
                (x, x.new_zeros(1, pad, heads, x.shape[-1])), dim=1)
            return part.reshape(1, n, chunk, heads, x.shape[-1]).permute(0, 1, 3, 2, 4)
        Qw, Kw, Vw, Gw, Bw, Ww = (window(x) for x in (Q, K, v, g, b, w))
        G = Gw.cumsum(dim=-2)
        expG = G.exp()
        Bk_e = (Bw * Kw) * expG
        Q_e = Qw * expG
        Ke_i = Kw * torch.exp(-G.clamp(min=-40.0))
        return Q_e, Ke_i, Bk_e, Vw, Ww, expG

    def build_masks(Q_e, Ke_i, Bk_e, Vw, Ww):
        A = (Bk_e @ Ke_i.transpose(-1, -2)).masked_fill(~tril_lt, 0.0).float()
        P = (Q_e @ Ke_i.transpose(-1, -2)).masked_fill(~tril_le, 0.0)
        rhs = torch.cat((Ww * Vw, Bk_e), dim=-1).float()
        return A, P, rhs

    def solve(A, rhs):
        L = torch.eye(chunk, device=device, dtype=torch.float32) + A
        return torch.linalg.solve_triangular(L, rhs, upper=False)

    def recurrence(P, Q_e, Ke_i, expG, X):
        X0, X1 = X[..., :dv], X[..., dv:]
        S = torch.zeros(1, heads, dk, dv, device=device, dtype=torch.float32)
        outs = []
        for idx in range(n):
            x = X0[:, idx] - X1[:, idx] @ S
            out = Q_e[:, idx] @ S + P[:, idx] @ x
            outs.append(out)
            decay = expG[:, idx, :, -1]
            S = decay.unsqueeze(-1) * S + ((decay.unsqueeze(-2) * Ke_i[:, idx]).transpose(-1, -2) @ x)
        return torch.cat(outs, dim=1)

    def full():
        Q_e, Ke_i, Bk_e, Vw, Ww, expG = prep()
        A, P, rhs = build_masks(Q_e, Ke_i, Bk_e, Vw, Ww)
        X = solve(A, rhs)
        return recurrence(P, Q_e, Ke_i, expG, X)

    stages = {}
    Q_e, Ke_i, Bk_e, Vw, Ww, expG = prep()
    A, P, rhs = build_masks(Q_e, Ke_i, Bk_e, Vw, Ww)
    X = solve(A, rhs)
    stages["prep"] = time_callable(prep, warmup=1, reps=reps)
    stages["masks"] = time_callable(lambda: build_masks(Q_e, Ke_i, Bk_e, Vw, Ww),
                                    warmup=1, reps=reps)
    stages["trisolve"] = time_callable(lambda: solve(A, rhs), warmup=1, reps=reps)
    stages["recurrence"] = time_callable(lambda: recurrence(P, Q_e, Ke_i, expG, X),
                                         warmup=1, reps=reps)
    stages["total"] = time_callable(full, warmup=1, reps=reps)
    return dict(ok=True, tokens=tokens, chunk=chunk, heads=heads, dk=dk, dv=dv,
                chunks=n, stages={k: dict(median_ms=v["median_ms"], source=v["source"])
                                  for k, v in stages.items()})


def compare_fused_gpu(*, tokens_list=(64, 2048), chunk=512, heads=10, dk=64, dv=64,
                      device="cuda", check_gradients=True) -> dict:
    """直接对照 fused 输出与现有 batched 输出，防止性能快来自漏算。"""
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward
    from hcl_tr.models.gdn2_chunk_opt import chunkwise_fused
    rows = {}
    for tokens in tokens_list:
        t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv,
                         seed=7, realistic_decay=True)
        args = (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"])
        dmax = float((-t["g"].detach()).amax())
        chunk_eff = max(1, min(chunk, int(40.0 / dmax))) if dmax > 0 else chunk
        ref_base = [x.detach() for x in args]
        ref_numeric = gdn2_chunkwise_batched_forward(*ref_base, chunk=chunk,
                                                     divide_by_sqrt_t=True,
                                                     l2norm_qk=True, chunk_eff_in=chunk_eff)
        row = {}
        for name, kwargs in (("fused_solve", dict(solve="reference")),
                             ("fused_blocked64", dict(solve="blocked", block=64))):
            try:
                fused_inputs = [x.detach().requires_grad_(check_gradients and tokens <= 64)
                                for x in args]
                ref_inputs = [x.detach().requires_grad_(check_gradients and tokens <= 64)
                              for x in args]
                ref = gdn2_chunkwise_batched_forward(*ref_inputs, chunk=chunk,
                                                     divide_by_sqrt_t=True,
                                                     l2norm_qk=True, chunk_eff_in=chunk_eff)
                got = chunkwise_fused(*fused_inputs, chunk=chunk, chunk_eff=chunk_eff, **kwargs)
            except ValueError as exc:
                row[name] = dict(skipped=str(exc))
                continue
            diff = (got.float() - ref.float()).abs()
            scale = ref.float().abs().clamp_min(1e-12)
            row[name] = dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                             max_rel=float((diff / scale).max()),
                             finite=bool(torch.isfinite(got).all()))
            if check_gradients and tokens <= 64:
                weight = torch.randn_like(ref)
                (ref * weight).sum().backward()
                (got * weight).sum().backward()
                grad_errors = [float((a.grad - b.grad).abs().max())
                               for a, b in zip(fused_inputs, ref_inputs)]
                row[name]["grad_max_abs"] = max(grad_errors)
                row[name]["grad_each_max_abs"] = grad_errors
                row[name]["grad_finite"] = all(bool(torch.isfinite(a.grad).all())
                                                for a in fused_inputs)
        rows[f"t{tokens}"] = row
    ok = all(v["finite"] and v["max_abs"] <= 5e-4
             and v.get("grad_finite", True) and v.get("grad_max_abs", 0.0) <= 5e-4
             for r in rows.values() for v in r.values())
    return dict(ok=ok, rows=rows, requested_chunk=chunk,
                effective_chunk=chunk_eff, dmax=dmax, realistic_decay=True,
                tolerance_abs=5e-4)


def compare_widestep_truth(*, tokens=256, heads=10, dk=64, dv=64, device="cuda") -> dict:
    """放大 chunk + 放宽 clamp 后，直接对 fp64 逐 token 真值校验。"""
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    from hcl_tr.models.gdn2_chunk_opt import chunkwise_fused
    t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv,
                     seed=11, realistic_decay=True)
    args = (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"])
    truth = reference_recurrence(*args)
    dmax = float((-t["g"].detach()).amax())
    rows = {}
    for chunk in (21, 43, 86):
        out = chunkwise_fused(*args, chunk=chunk, chunk_eff=chunk, solve="reference",
                              neg_clamp=80.0)
        diff = (out.double() - truth).abs()
        rows[f"chunk{chunk}"] = dict(max_abs=float(diff.max()),
                                     mean_abs=float(diff.mean()),
                                     finite=bool(torch.isfinite(out).all()))
    # 最大 chunk 的梯度与既有批量实现对照（前向已对 fp64 真值校验）。
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward
    ref_inputs = [x.detach().requires_grad_(True) for x in args]
    fused_inputs = [x.detach().requires_grad_(True) for x in args]
    # 对照组必须用它自己准确的 chunk：reference 的 clamp 固定 -40，
    # 在 chunk=86 时其前向本身就不可信（实测误差 ~1e-1）。
    ref_chunk = max(1, min(chunk, int(40.0 / dmax))) if dmax > 0 else chunk
    ref = gdn2_chunkwise_batched_forward(*ref_inputs, chunk=ref_chunk,
                                          chunk_eff_in=ref_chunk,
                                          divide_by_sqrt_t=True, l2norm_qk=True)
    fused = chunkwise_fused(*fused_inputs, chunk=chunk, chunk_eff=chunk,
                            solve="reference", neg_clamp=80.0)
    weight = torch.randn_like(ref)
    (ref * weight).sum().backward()
    (fused * weight).sum().backward()
    grad_err = [float((a.grad - b.grad).abs().max()) for a, b in zip(fused_inputs, ref_inputs)]
    grad_ok = all(bool(torch.isfinite(a.grad).all()) and e <= 1e-4
                  for a, e in zip(fused_inputs, grad_err))
    ok = (all(v["finite"] and v["max_abs"] <= 1e-5 for v in rows.values()) and grad_ok)
    return dict(ok=ok, tokens=tokens, dmax=dmax, clamp=-80.0, rows=rows,
                grad_chunk=chunk, grad_max_abs=max(grad_err),
                grad_each_max_abs=grad_err, grad_ok=grad_ok, tolerance_abs=1e-5)


def compare_compiled_gradients(*, tokens=256, chunk=86, heads=10, dk=64, dv=64,
                               device="cuda") -> dict:
    """torch.compile 版本的前向与反向都必须和 eager 一致（训练可用性门槛）。"""
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    from hcl_tr.models.gdn2_chunk_opt import chunkwise_fused, chunkwise_fused_compiled
    t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv,
                     seed=13, realistic_decay=True)
    base = [t["q"], t["k"], t["v"], t["g"], t["b"], t["w"]]
    eager_in = [x.detach().requires_grad_(True) for x in base]
    comp_in = [x.detach().requires_grad_(True) for x in base]
    eager = chunkwise_fused(*eager_in, chunk=chunk, chunk_eff=chunk,
                            solve="reference", neg_clamp=80.0)
    compiled = chunkwise_fused_compiled(*comp_in, chunk=chunk, chunk_eff=chunk,
                                        solve="reference", neg_clamp=80.0)
    fwd = float((compiled.float() - eager.float()).abs().max())
    weight = torch.randn_like(eager)
    (eager * weight).sum().backward()
    (compiled * weight).sum().backward()
    grads = [float((a.grad - b.grad).abs().max()) for a, b in zip(comp_in, eager_in)]
    ok = fwd <= 1e-5 and all(g <= 1e-4 and bool(torch.isfinite(a.grad).all())
                             for g, a in zip(grads, comp_in))
    return dict(ok=ok, tokens=tokens, chunk=chunk, forward_max_abs=fwd,
                grad_each_max_abs=grads, grad_max_abs=max(grads))


def benchmark_torch_features(*, tokens_list=(2048, 8192), chunk=86, heads=10, dk=64, dv=64,
                             reps=5, device="cuda") -> dict:
    """测试可开关的运行时特性：torch.compile 三种模式与精度设置。

    每个模式独立 try/except：某一个不可用不应导致整份报告失败。
    """
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    from hcl_tr.models.gdn2_chunk_opt import chunkwise_fused

    caps = dict(
        torch=torch.__version__,
        compile_available=hasattr(torch, "compile"),
        tf32_allow=bool(torch.backends.cuda.matmul.allow_tf32),
        fp32_matmul_precision=torch.get_float32_matmul_precision(),
        bf16_supported=bool(torch.cuda.is_bf16_supported()) if device == "cuda" else False,
        capability=list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
    )

    if device == "cuda":
        _warm_clocks()

    rows = {}
    # 基准：eager
    eager_times = {}
    for tokens in tokens_list:
        t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv,
                         seed=1, realistic_decay=True)
        args = (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"])
        dmax = float((-t["g"].detach()).amax())
        ce = max(1, min(chunk, int(160.0 / dmax))) if dmax > 0 else chunk
        timing = time_callable(lambda: chunkwise_fused(*args, chunk=ce, chunk_eff=ce,
                                                       solve="reference", neg_clamp=80.0),
                               warmup=2, reps=reps)
        eager_times[f"t{tokens}"] = timing["median_ms"]

    modes = ("default", "reduce-overhead", "max-autotune")
    for mode in modes:
        if not hasattr(torch, "compile"):
            rows[mode] = dict(error="torch.compile 不可用")
            continue
        try:
            compiled = torch.compile(chunkwise_fused, mode=mode, dynamic=False)
            row = {}
            for tokens in tokens_list:
                t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads,
                                 dk=dk, dv=dv, seed=1, realistic_decay=True)
                args = (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"])
                dmax = float((-t["g"].detach()).amax())
                ce = max(1, min(chunk, int(160.0 / dmax))) if dmax > 0 else chunk
                start = time.perf_counter()
                out = compiled(*args, chunk=ce, chunk_eff=ce, solve="reference", neg_clamp=80.0)
                if device == "cuda":
                    torch.cuda.synchronize()
                compile_s = time.perf_counter() - start
                ref_out = chunkwise_fused(*args, chunk=ce, chunk_eff=ce,
                                          solve="reference", neg_clamp=80.0)
                err = float((out.float() - ref_out.float()).abs().max())
                timing = time_callable(
                    lambda: compiled(*args, chunk=ce, chunk_eff=ce,
                                     solve="reference", neg_clamp=80.0),
                    warmup=2, reps=reps)
                row[f"t{tokens}"] = dict(
                    median_ms=timing["median_ms"], eager_ms=eager_times[f"t{tokens}"],
                    speedup=eager_times[f"t{tokens}"] / timing["median_ms"],
                    compile_s=compile_s, max_abs_vs_eager=err)
            rows[mode] = row
        except Exception as exc:  # noqa: BLE001
            rows[mode] = dict(error=f"{type(exc).__name__}: {exc}")
    return dict(ok=True, capabilities=caps, results=rows)


def benchmark_stage_compare(*, tokens=8192, chunk=512, heads=10, dk=64, dv=64,
                            reps=5, device="cuda") -> dict:
    """把"参考实现 vs 融合实现"的差异定位到具体算子。

    共享同一份预处理结果，只比较：掩码构造（两次 GEMM vs 一次合并 GEMM）
    与三角求解（整块 vs 分块）。
    """
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    if device == "cuda":
        _warm_clocks()
    t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv, seed=1)
    q, k, v, g, b, w = (x.float() for x in (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"]))
    n = (tokens + chunk - 1) // chunk
    pad = n * chunk - tokens
    mask_lt = torch.ones(chunk, chunk, device=device, dtype=torch.bool).tril(-1)
    mask_le = torch.ones(chunk, chunk, device=device, dtype=torch.bool).tril()

    def window(x):
        part = x if not pad else torch.cat((x, x.new_zeros(1, pad, heads, x.shape[-1])), dim=1)
        return part.reshape(1, n, chunk, heads, x.shape[-1]).permute(0, 1, 3, 2, 4)

    Q, K, V, Gm, Bm, W = (window(x) for x in (_l2norm(q), _l2norm(k), v, g, b, w))
    cum = Gm.cumsum(dim=-2)
    exp_cum = cum.exp()
    bk_e = (Bm * K) * exp_cum
    q_e = Q * exp_cum
    k_ei = K * torch.exp(-cum.clamp(min=-40.0))
    rhs = torch.cat((W * V, bk_e), dim=-1).float()

    def masks_reference():
        a = (bk_e @ k_ei.transpose(-1, -2)).masked_fill(~mask_lt, 0.0).float()
        p = (q_e @ k_ei.transpose(-1, -2)).masked_fill(~mask_le, 0.0)
        return a, p

    def masks_fused():
        both = torch.cat((bk_e, q_e), dim=-2) @ k_ei.transpose(-1, -2)
        a = both[..., :chunk, :].masked_fill(~mask_lt, 0.0).float()
        p = both[..., chunk:, :].masked_fill(~mask_le, 0.0)
        return a, p

    a_ref, _ = masks_reference()
    lin = torch.eye(chunk, device=device, dtype=torch.float32) + a_ref

    from hcl_tr.models.gdn2_chunk_opt import _blocked_solve

    stages = {
        "masks_two_gemm": masks_reference,
        "masks_one_gemm": masks_fused,
        "solve_whole": lambda: torch.linalg.solve_triangular(lin, rhs, upper=False),
        "solve_blocked64": lambda: _blocked_solve(a_ref, rhs, 64),
    }
    out = {}
    for name, fn in stages.items():
        timing = time_callable(fn, warmup=2, reps=reps)
        out[name] = dict(median_ms=timing["median_ms"], min_ms=timing["min_ms"])
    return dict(ok=True, tokens=tokens, chunk=chunk, reps=reps, results=out)


def benchmark_chunk_variants(*, tokens_list=(2048, 8192), chunk=512, heads=10, dk=64,
                             dv=64, reps=5, device="cuda") -> dict:
    """同场对比：现有批量实现 vs 融合 A/P vs 融合 + 分块前代。

    所有变体用同一批输入、同一轮内计时，避免时钟爬升造成的跨轮偏差。
    """
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward
    from hcl_tr.models.gdn2_chunk_opt import chunkwise_fused

    if device == "cuda":
        _warm_clocks()

    results = {}
    for tokens in tokens_list:
        t = make_tensors(device=device, batch=1, tokens=tokens, heads=heads, dk=dk, dv=dv,
                         seed=1, realistic_decay=True)
        args = (t["q"], t["k"], t["v"], t["g"], t["b"], t["w"])
        dmax = float((-t["g"].detach()).amax())
        chunk_eff = max(1, min(chunk, int(40.0 / dmax))) if dmax > 0 else chunk
        # 中点平移后单因子幅度上界是 e^{chunk×dmax/2}，故上限为 160/dmax。
        wide_eff = max(1, min(86, int(160.0 / dmax))) if dmax > 0 else 86
        variants = {
            "reference": lambda a, c, ce: gdn2_chunkwise_batched_forward(
                *a, chunk=c, chunk_eff_in=ce, divide_by_sqrt_t=True, l2norm_qk=True),
            "fused_solve": lambda a, c, ce: chunkwise_fused(
                *a, chunk=c, chunk_eff=ce, solve="reference"),
            "fused_blocked64": lambda a, c, ce: chunkwise_fused(
                *a, chunk=c, chunk_eff=ce, solve="blocked", block=64),
            "fused_blocked128": lambda a, c, ce: chunkwise_fused(
                *a, chunk=c, chunk_eff=ce, solve="blocked", block=128),
            "fused_widestep": lambda a, c, ce: chunkwise_fused(
                *a, chunk=wide_eff, chunk_eff=wide_eff, solve="reference",
                neg_clamp=80.0),
        }
        row = {}
        # 正反两轮：若某变体只在"第一个跑"时慢，说明是时钟爬升而不是实现差异。
        for pass_name, order in (("forward", list(variants)),
                                 ("reverse", list(variants)[::-1])):
            for name in order:
                fn = variants[name]
                try:
                    timing = time_callable(lambda fn=fn: fn(args, chunk, chunk_eff), warmup=2, reps=reps)
                except ValueError as exc:
                    row.setdefault(name, {}).setdefault("skipped", str(exc))
                    continue
                entry = row.setdefault(name, {})
                entry.setdefault("passes_ms", {})[pass_name] = timing["median_ms"]
                entry["median_ms"] = min(entry["passes_ms"].values())
                entry["max_ms"] = max(entry.get("max_ms", 0.0), timing["max_ms"])
        baseline = row.get("reference", {}).get("median_ms")
        for name, value in row.items():
            if baseline and "median_ms" in value:
                value["speedup_vs_reference"] = baseline / value["median_ms"]
                passes = value["passes_ms"]
                if len(passes) == 2:
                    value["order_effect"] = max(passes.values()) / min(passes.values())
        results[f"t{tokens}_c{chunk}"] = dict(
            requested_chunk=chunk, effective_chunk=chunk_eff, wide_chunk=wide_eff,
            dmax=dmax, variants=row)
    return dict(ok=True, requested_chunk=chunk, heads=heads, dk=dk, dv=dv,
                realistic_decay=True, reps=reps,
                order_control="forward+reverse, median_ms=两轮较小值", results=results)


def _warm_clocks(seconds: float = 1.5) -> None:
    """先把 GPU 时钟拉满，避免第一个测量项被低频拖慢。"""
    a = torch.randn(2048, 2048, device="cuda")
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        a = a @ a
    torch.cuda.synchronize()


def probe_matmul_baseline(*, device="cuda", shape=(2048, 64, 64), reps=10) -> dict:
    """给出 torch 在代表性 chunk 规模上的基线，作为 TileLang 后续对比的分母。

    (2048, 64, 64) 对应 8K 序列、chunk=64 时约 B*H*N 个 64x64 块乘。
    """
    if device == "cuda" and not torch.cuda.is_available():
        return dict(ok=False, error="需要 CUDA")
    a = torch.randn(shape, device=device)
    b = torch.randn(shape, device=device)
    timing = time_callable(lambda: torch.matmul(a, b), reps=reps)
    return dict(ok=True, shape=list(shape), timing=timing)


def run_probe(*, output: Path | str | None = None, tolerance=None,
              require_capability=(7, 5)) -> dict:
    """执行可行性探测。返回 report；任何前置条件失败都写入 blockers。"""
    report = dict(environment=environment(), blockers=[], steps={}, feasible=False)
    report["blockers"].extend(
        environment_blockers(report["environment"], require_capability=require_capability))
    try:
        report["steps"]["flash_qla"] = discover_flash_qla()
    except Exception as exc:  # noqa: BLE001
        report["steps"]["flash_qla"] = dict(error=f"{type(exc).__name__}: {exc}")
    report["steps"]["tilelang"] = _tilelang_smoke_compile()
    if not report["steps"]["tilelang"].get("ok"):
        report["blockers"].append("TileLang 不可用或 API 不符: "
                                 + report["steps"]["tilelang"].get("error", "unknown"))
    if report["environment"]["cuda_available"]:
        report["steps"]["torch_matmul_baseline"] = probe_matmul_baseline()
        try:
            tensors = make_tensors(device="cuda", batch=2, tokens=64, heads=4, dk=16, dv=16)
            from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward

            def backend(**kw):
                chunk = kw.pop("chunk")
                return gdn2_chunkwise_batched_forward(
                    kw["q"], kw["k"], kw["v"], kw["g"], kw["b"], kw["w"],
                    chunk=chunk, divide_by_sqrt_t=True, l2norm_qk=True)

            report["steps"]["pytorch_batched_numeric"] = compare_backend(
                backend, tensors, chunk=16, tolerance=tolerance)
            # 生产配置基线：B=1,H=10,dk=dv=64,chunk=512（见 scripts/ab_probe.py）。
            timings = {}
            for tokens in (2048, 8192):
                prod = make_tensors(device="cuda", batch=1, tokens=tokens, heads=10,
                                    dk=64, dv=64, seed=1)
                timings[f"t{tokens}_c512"] = time_callable(
                    lambda prod=prod: backend(**prod, chunk=512), warmup=1, reps=3)
            report["steps"]["pytorch_batched_timing"] = timings
            report["steps"]["breakdown_8k_c512"] = profile_pytorch_breakdown(
                tokens=8192, chunk=512, reps=3)
            report["steps"]["variant_benchmark"] = benchmark_chunk_variants(
                tokens_list=(2048, 8192), chunk=512, reps=5)
            report["steps"]["stage_compare_8k"] = benchmark_stage_compare(
                tokens=8192, chunk=512, reps=5)
            report["steps"]["fused_correctness"] = compare_fused_gpu(
                tokens_list=(64, 2048), chunk=512)
            report["steps"]["widestep_correctness"] = compare_widestep_truth(
                tokens=256, heads=10, dk=64, dv=64)
            report["steps"]["torch_features"] = benchmark_torch_features(
                tokens_list=(2048, 8192), chunk=86, reps=5)
            report["steps"]["compiled_gradients"] = compare_compiled_gradients(
                tokens=256, chunk=86, heads=10, dk=64, dv=64)
        except Exception as exc:  # noqa: BLE001
            report["blockers"].append(f"PyTorch batched 对照失败: {type(exc).__name__}: {exc}")
    report["feasible"] = (not report["blockers"]
                          and report["steps"].get("pytorch_batched_numeric", {}).get("ok", False))
    report["note"] = ("本轮只验证工具链与对照基线；完整 GDN-2 TileLang kernel 尚未移植，"
                      "因此 feasible=True 仅表示可以进入下一步，不代表已有加速。")
    if output is not None:
        Path(output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/feasibility.json"))
    args = parser.parse_args()
    report = run_probe(output=args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
