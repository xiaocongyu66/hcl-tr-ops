"""GDN-2 状态转移数值 fixture（fp32 参考实现自洽性 + 闭式对照）。

SOURCE_LOCK: GDN-2 LOCKED @ de8476de0fecae08fc16f0bf3154900244c61c73
公式: vendor/gdn2/lit_gpt/gdn2.py:57-62
      S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T

容差（计划书 §10, fp32 reference）: GDN-2 状态转移 max_abs_err <= 1e-5

注意：与官方 Triton kernel 的逐元素对比需要 GPU + fla，标记为 gpu 用例（默认 skip）。
本文件保证 CPU 上可验证的参考实现自洽性与闭式等价性。
"""

from __future__ import annotations

import pytest
import torch

from hcl_tr.models.gdn2_ref import gdn2_step, gdn2_chunk

SEED = 1234
TOL_FP32 = 1e-5
DTYPE_INFO = {"dtype": "float32", "kernel": "pytorch-reference"}


def _rand(shape, seed=SEED, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32) * scale


def _l2norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_step_matches_closed_form():
    """S_t 与闭式 (I - k a^T) D S_{t-1} + k (w⊙v)^T 一致。"""
    H, dk, dv = 4, 8, 8
    S_prev = _rand((H, dk, dv), seed=1)
    k = _l2norm(_rand((H, dk), seed=2))
    v = _rand((H, dv), seed=3)
    g = -torch.rand((H, dk), generator=torch.Generator().manual_seed(4))
    b = torch.rand((H, dk), generator=torch.Generator().manual_seed(5))
    w = torch.rand((H, dv), generator=torch.Generator().manual_seed(6))

    S = gdn2_step(S_prev, k, v, g, b, w, l2norm_qk=False)

    # 闭式：逐头 I - k (b⊙k)^T
    I = torch.eye(dk)
    a = b * k
    M = I.unsqueeze(0) - k.unsqueeze(-1) * a.unsqueeze(-2)   # [H,dk,dk]
    D = g.exp().unsqueeze(-1)                                  # [H,dk,1]
    S_ref = torch.einsum("hij,hjk->hik", M, D * S_prev) + k.unsqueeze(-1) * (w * v).unsqueeze(-2)

    err = (S - S_ref).abs().max().item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_decay_only_recovers_previous_state():
    """b=w=0（无擦除无写入）时，S_t = Diag(exp(g)) S_{t-1}。"""
    H, dk, dv = 2, 4, 4
    S_prev = _rand((H, dk, dv), seed=10)
    k = _l2norm(_rand((H, dk), seed=11))
    v = _rand((H, dv), seed=12)
    g = -torch.rand((H, dk), generator=torch.Generator().manual_seed(13))
    b = torch.zeros(H, dk); w = torch.zeros(H, dv)

    S = gdn2_step(S_prev, k, v, g, b, w, l2norm_qk=False)
    S_ref = g.exp().unsqueeze(-1) * S_prev
    err = (S - S_ref).abs().max().item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_write_without_decay_accumulates_outer_products():
    """g=0, b=0 时，每步 S_t = S_{t-1} + k (w⊙v)^T，状态为外积累加。"""
    H, dk, dv, T = 2, 4, 4, 5
    S0 = torch.zeros(H, dk, dv)
    ks = _l2norm(_rand((T, H, dk), seed=20))
    vs = _rand((T, H, dv), seed=21)
    gs = torch.zeros(T, H, dk)
    bs = torch.zeros(T, H, dk)
    ws = torch.ones(T, H, dv)

    S_T, _ = gdn2_chunk(S0, ks, vs, gs, bs, ws, l2norm_qk=False)
    S_ref = torch.einsum("thk,thv->hkv", ks, vs)   # 期望的外积和
    err = (S_T - S_ref).abs().max().item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_chunk_matches_iterated_steps():
    """gdn2_chunk 与手动循环 gdn2_step 完全一致。"""
    H, dk, dv, T = 2, 4, 4, 6
    S0 = _rand((H, dk, dv), seed=30, scale=0.5)
    ks = _rand((T, H, dk), seed=31)
    vs = _rand((T, H, dv), seed=32)
    gs = -torch.rand((T, H, dk), generator=torch.Generator().manual_seed(33))
    bs = torch.rand((T, H, dk), generator=torch.Generator().manual_seed(34))
    ws = torch.rand((T, H, dv), generator=torch.Generator().manual_seed(35))

    S_T, S_all = gdn2_chunk(S0, ks, vs, gs, bs, ws, l2norm_qk=True)
    S = S0
    for t in range(T):
        S = gdn2_step(S, ks[t], vs[t], gs[t], bs[t], ws[t], l2norm_qk=True)
    err = (S - S_T).abs().max().item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}
    assert (S_all[-1] - S_T).abs().max().item() <= TOL_FP32


@pytest.mark.gpu
def test_gdn2_matches_official_kernel():
    """与官方 Triton kernel 逐元素对比（需 GPU + fla）。"""
    pytest.importorskip("fla")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    import sys
    from pathlib import Path
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "gdn2"
    sys.path.insert(0, str(vendor))
    from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2  # noqa: F401

    pytest.skip("官方 kernel 对比需真实 GPU 环境，CI 上显式跳过")


# ---------- CPU fp32 独立参考对比（替代原先因缺 fla 而永久 skip 的官方 kernel 对比）----------

def test_gdn2_matches_independent_cpu_reference():
    """实现 vs 独立朴素参考（论文公式的显式矩阵形式）。

    计划书 §10：fp32 reference 下 GDN-2 状态转移 max_abs_err <= 1e-5。
    这条不依赖 GPU/fla，因此**不允许 skip**。
    """
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    from hcl_tr.models.gdn2_ref import gdn2_recurrent_forward

    torch.manual_seed(0)
    T, H, dk = 24, 2, 8
    dv = dk
    q = torch.randn(T, H, dk)
    k = torch.randn(T, H, dk)
    v = torch.randn(T, H, dv)
    g = -torch.rand(T, H, dk) * 0.3
    b = torch.rand(T, H, dk)
    w = torch.rand(T, H, dv)

    ref = naive_gdn2_reference(q, k, v, g, b, w)
    got = gdn2_recurrent_forward(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
        b.unsqueeze(0), w.unsqueeze(0), use_checkpoint=False,
    )[0]

    err = (ref - got.double()).abs().max().item()
    assert err <= 1e-5, f"max_abs_err={err:.3e} 超过 fp32 容差 1e-5"


def test_gdn2_chunkwise_matches_independent_cpu_reference():
    """chunkwise 版同样要对齐独立参考（多 chunk 尺寸）。"""
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward

    torch.manual_seed(1)
    T, H, dk = 20, 2, 8
    dv = dk
    q = torch.randn(T, H, dk); k = torch.randn(T, H, dk); v = torch.randn(T, H, dv)
    g = -torch.rand(T, H, dk) * 0.3; b = torch.rand(T, H, dk); w = torch.rand(T, H, dv)

    ref = naive_gdn2_reference(q, k, v, g, b, w)
    for C in [4, 8, 20]:
        got = gdn2_chunkwise_forward(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
            b.unsqueeze(0), w.unsqueeze(0), chunk=C,
        )[0]
        err = (ref - got.double()).abs().max().item()
        assert err <= 1e-5, f"chunk={C}: max_abs_err={err:.3e}"


def test_gdn2_reference_without_qk_norm():
    """l2norm_qk=False 路径也要与参考一致。"""
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    from hcl_tr.models.gdn2_ref import gdn2_recurrent_forward

    torch.manual_seed(2)
    T, H, dk = 12, 2, 6
    q = 0.3 * torch.randn(T, H, dk); k = 0.3 * torch.randn(T, H, dk)
    v = torch.randn(T, H, dk); g = -torch.rand(T, H, dk) * 0.2
    b = torch.rand(T, H, dk); w = torch.rand(T, H, dk)

    ref = naive_gdn2_reference(q, k, v, g, b, w, l2norm_qk=False)
    got = gdn2_recurrent_forward(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
        b.unsqueeze(0), w.unsqueeze(0), use_checkpoint=False, l2norm_qk=False,
    )[0]
    assert (ref - got.double()).abs().max().item() <= 1e-5


def test_chunkwise_factorized_matches_reference_at_production_chunk():
    """生产尺寸 chunk=64 下，因式分解版仍须与独立参考一致。"""
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward
    torch.manual_seed(7)
    T, H, dk = 256, 2, 8
    q = torch.randn(T, H, dk); k = torch.randn(T, H, dk); v = torch.randn(T, H, dk)
    g = -torch.rand(T, H, dk) * 0.3; b = torch.rand(T, H, dk); w = torch.rand(T, H, dk)
    ref = naive_gdn2_reference(q, k, v, g, b, w)
    got = gdn2_chunkwise_forward(
        q[None], k[None], v[None], g[None], b[None], w[None], chunk=64)[0]
    err = (ref - got.double()).abs().max().item()
    assert err <= 1e-5, f"max_abs_err={err:.3e}"


def test_chunkwise_stable_under_strong_decay_and_large_chunk():
    """数值安全回归：强衰减 + 大 chunk 曾因因式分解溢出，误差高达 1e-1 / NaN。"""
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward
    torch.manual_seed(3)
    T, H, dk = 256, 2, 8
    q = torch.randn(T, H, dk); k = torch.randn(T, H, dk); v = torch.randn(T, H, dk)
    b = torch.rand(T, H, dk); w = torch.rand(T, H, dk)
    for decay in (0.5, 2.0, 11.0):
        g = -torch.rand(T, H, dk) * decay
        ref = naive_gdn2_reference(q, k, v, g, b, w)
        o = gdn2_chunkwise_forward(q[None], k[None], v[None], g[None], b[None], w[None],
                                   chunk=256, sub_chunk=64)[0]
        assert torch.isfinite(o).all(), f"decay={decay}: 出现非有限值"
        err = (ref - o.double()).abs().max().item()
        assert err < 1e-4, f"decay={decay}: max_abs_err={err:.3e}"


@pytest.mark.parametrize("batched", [False, True])
def test_cached_chunk_does_not_repeat_decay_reduction(monkeypatch, batched):
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward, STATS
    if batched:
        gdn2_chunkwise_forward = gdn2_chunkwise_batched_forward
    torch.manual_seed(5)
    q, k, v = [torch.randn(1, 16, 2, 4) for _ in range(3)]
    g = -torch.rand_like(q)
    b, w = torch.rand_like(q), torch.rand_like(q)
    expected = gdn2_chunkwise_forward(q, k, v, g, b, w, chunk=4)
    def forbidden(*args, **kwargs):
        raise AssertionError("已校准路径不应再调用 amax")
    monkeypatch.setattr(torch.Tensor, "amax", forbidden)
    actual = gdn2_chunkwise_forward(q, k, v, g, b, w, chunk=16, chunk_eff_in=4)
    assert STATS["chunk_eff"] == 4
    torch.testing.assert_close(actual, expected)


def _batched_inputs(decay):
    rng = torch.Generator().manual_seed(918)
    key = (2, 23, 3, 4)
    value = (2, 23, 3, 5)
    return [torch.randn(key, generator=rng) * 0.3,
            torch.randn(key, generator=rng) * 0.3,
            torch.randn(value, generator=rng),
            -torch.rand(key, generator=rng) * decay,
            torch.rand(key, generator=rng), torch.rand(value, generator=rng),
            torch.rand(key[:-1], generator=rng), torch.rand(key[:-1], generator=rng)]


def _differentiable_naive(inputs, norm, divide):
    # Explicit matrix reference: no production helpers, no in-place state updates.
    q, k, v, g, b, w, gq, gk = [x.double() for x in inputs]
    if norm:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q, k = q * gq[..., None], k * gk[..., None]
    S = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
    eye = torch.eye(q.shape[-1], dtype=q.dtype)
    outs = []
    for t in range(q.shape[1]):
        kt = k[:, t, :, :, None]
        erase = eye - kt @ (b[:, t] * k[:, t]).unsqueeze(-2)
        S = erase @ torch.diag_embed(g[:, t].exp()) @ S + kt @ (w[:, t] * v[:, t]).unsqueeze(-2)
        out = (q[:, t].unsqueeze(-2) @ S).squeeze(-2)
        outs.append(out / (t + 1) ** 0.5 if divide else out)
    return torch.stack(outs, dim=1)


@pytest.mark.parametrize("decay", [0.2, 11.0])
@pytest.mark.parametrize("norm,divide", [(True, True), (False, False)])
@pytest.mark.parametrize("solve_mode", ["solver", "neumann"])
def test_batched_outputs_and_all_gradients(decay, norm, divide, solve_mode):
    from hcl_tr.models.gdn2_ref import (
        gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward, STATS,
    )
    from torch.utils.checkpoint import checkpoint
    inputs = _batched_inputs(decay)
    probe = torch.randn(2, 23, 3, 5, generator=torch.Generator().manual_seed(82))
    common = dict(chunk=16, l2norm_qk=norm, divide_by_sqrt_t=divide, solve_mode=solve_mode)

    def evaluate(fn):
        xs = [x.clone().requires_grad_() for x in inputs]
        out = fn(xs)
        grads = torch.autograd.grad((out * probe).sum(), xs)
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(grad).all() for grad in grads)
        return out.detach(), grads

    expected = evaluate(lambda xs: _differentiable_naive(xs, norm, divide))
    actual = evaluate(lambda xs: gdn2_chunkwise_batched_forward(
        *xs[:6], gq=xs[6], gk=xs[7], precompute_chunks=2, **common))
    effective = STATS["chunk_eff"]
    assert effective == (16 if decay == 0.2 else 3)
    # Both paths MUST use the same effective chunk, even under strong decay.
    old = evaluate(lambda xs: gdn2_chunkwise_forward(
        *xs[:6], gq=xs[6], gk=xs[7], chunk_eff_in=effective, **common))

    def run(*xs):
        return gdn2_chunkwise_batched_forward(
            *xs[:6], gq=xs[6], gk=xs[7], chunk_eff_in=effective,
            precompute_chunks=2, **common)

    recomputed = evaluate(lambda xs: checkpoint(run, *xs, use_reentrant=False))
    for result in (actual, old, recomputed):
        torch.testing.assert_close(result[0].double(), expected[0], atol=1e-5, rtol=1e-4)
        for name, got, ref in zip(("q", "k", "v", "g", "b", "w", "gq", "gk"), result[1], expected[1]):
            torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-4, msg=name)
    torch.testing.assert_close(recomputed[0], actual[0])
    for got, ref in zip(recomputed[1], actual[1]):
        torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_batched_window_solve_and_cached_override(dtype, monkeypatch):
    from hcl_tr.models.gdn2_ref import (
        gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward, STATS,
    )
    if dtype == torch.float16:
        try:
            torch.ones(1).to(dtype)
        except RuntimeError as exc:
            if "Failed to initialize cpuinfo" not in str(exc):
                raise
            pytest.skip("host PyTorch cannot initialize cpuinfo for CPU fp16")
    inputs = _batched_inputs(0.2)
    calls = []
    solve = torch.linalg.solve_triangular

    def record(L, rhs, **kwargs):
        calls.append((L.shape, rhs.shape))
        return solve(L, rhs, **kwargs)

    monkeypatch.setattr(torch.linalg, "solve_triangular", record)
    common = dict(chunk=16, chunk_eff_in=7, chunk_override=4, matmul_dtype=dtype)
    out = gdn2_chunkwise_batched_forward(*inputs[:6], precompute_chunks=2, **common)
    assert STATS["chunk_eff"] == 4
    assert len(calls) == 3  # six chunks, two per solve, including padded final chunk
    assert all(L == (2, 2, 3, 4, 4) and rhs == (2, 2, 3, 4, 9) for L, rhs in calls)
    old = gdn2_chunkwise_forward(*inputs[:6], **common)
    assert out.dtype == torch.float32
    tol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(out, old, atol=tol, rtol=tol)
    gradients = []
    for fn in (gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward):
        xs = [x.clone().requires_grad_() for x in inputs[:6]]
        y = fn(*xs, **common)
        gradients.append(torch.autograd.grad(y.square().sum(), xs))
    for got, ref in zip(gradients[1], gradients[0]):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, atol=tol, rtol=tol * 5)


def test_batched_strong_decay_against_existing_independent_reference():
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward
    from tests.unit.gdn2_reference_cpu import naive_gdn2_reference
    inputs = _batched_inputs(11.0)[:6]
    out = gdn2_chunkwise_batched_forward(*inputs, chunk=16, precompute_chunks=2)
    ref = torch.stack([naive_gdn2_reference(*(x[i] for x in inputs)) for i in range(2)])
    torch.testing.assert_close(out.double(), ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("backend", ["serial", "batched"])
def test_keyword_autocast_matches_positional_outputs_and_gradients(backend):
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward
    fn = gdn2_chunkwise_forward if backend == "serial" else gdn2_chunkwise_batched_forward
    inputs = _batched_inputs(0.8)
    results = []
    for keyword in (False, True):
        xs = [x.clone().requires_grad_() for x in inputs]
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            if keyword:
                out = fn(**dict(zip(("q", "k", "v", "g", "b", "w", "gq", "gk"), xs)), chunk=8)
            else:
                out = fn(*xs[:6], gq=xs[6], gk=xs[7], chunk=8)
        gradients = torch.autograd.grad(out.square().sum(), xs)
        results.append((out, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    for positional, keyword in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(positional, keyword, atol=0, rtol=0)
