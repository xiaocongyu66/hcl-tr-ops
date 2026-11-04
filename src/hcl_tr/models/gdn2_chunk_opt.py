"""GDN-2 chunkwise 的融合实现：A/P 合并 GEMM + 分块前代三角求解。

与 `gdn2_ref.gdn2_chunkwise_batched_forward` 等价，区别只在实现方式：

  * 原有实现做两次 ``[C,dk] @ [dk,C]``：一次给 A（Bk_e 作左操作数），
    一次给 P（Q_e 作左操作数）。这里把两者拼成 ``[2C,dk] @ [dk,C]``，
    单次 GEMM、单个 kernel launch，M 维翻倍更适合 GPU。
  * 原有实现调用整块 ``solve_triangular``。这里提供按 block 分块的前代：
    逐块减去前面块的贡献，再解一个 block×block 的小三角系统。

数学顺序不变，浮点结果可能有末位差异。默认走 ``solve="blocked"``；
``solve="reference"`` 保留整块求解以便逐值对照。
"""

from __future__ import annotations

import torch

__all__ = ["chunkwise_fused"]


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _window(x: torch.Tensor, n: int, chunk: int, heads: int, pad: int) -> torch.Tensor:
    batch = x.shape[0]
    if pad:
        x = torch.cat((x, x.new_zeros(batch, pad, heads, x.shape[-1])), dim=1)
    return x.reshape(batch, n, chunk, heads, x.shape[-1]).permute(0, 1, 3, 2, 4)


def _blocked_solve(a: torch.Tensor, rhs: torch.Tensor, block: int) -> torch.Tensor:
    """解 (I + a) X = rhs，a 严格下三角；按 block 前代。

    a: [..., C, C]，rhs: [..., C, m]。要求 C 能被 block 整除。
    """
    c = a.shape[-1]
    if c % block:
        raise ValueError(f"chunk {c} 不能被 block {block} 整除")
    solved = []
    for i in range(c // block):
        lo, hi = i * block, (i + 1) * block
        current_rhs = rhs[..., lo:hi, :]
        if solved:
            previous = torch.cat(solved, dim=-2)
            current_rhs = current_rhs - a[..., lo:hi, :lo] @ previous
        diag = torch.eye(block, device=a.device, dtype=a.dtype) + a[..., lo:hi, lo:hi]
        solved.append(torch.linalg.solve_triangular(diag, current_rhs, upper=False))
    return torch.cat(solved, dim=-2)


def chunkwise_fused(
    q, k, v, g, b, w, *, chunk: int = 512, divide_by_sqrt_t: bool = True,
    l2norm_qk: bool = True, solve: str = "blocked", block: int = 64,
    matmul_dtype: torch.dtype = torch.float32, chunk_eff: int | None = None,
    chunk_override: int | None = None,
):
    """等价于 gdn2_ref.gdn2_chunkwise_batched_forward 的融合实现。

    ``solve``: "blocked"（默认）或 "reference"（整块 solve_triangular）。
    """
    if solve not in ("blocked", "reference"):
        raise ValueError(f"不支持的 solve 模式: {solve}")
    batch, tokens, heads, dk = q.shape
    dv = v.shape[-1]
    dev = q.device

    qf, kf, vf, gf, bf, wf = (t.to(torch.float32) for t in (q, k, v, g, b, w))
    if l2norm_qk:
        qf, kf = _l2norm(qf), _l2norm(kf)

    if chunk_eff is None:
        dmax = float((-gf.detach()).amax()) if tokens else 0.0
        if not torch.isfinite(torch.as_tensor(dmax)):
            raise ValueError("GDN2 衰减包含 NaN/Inf")
        safe_exp = 10.0 if matmul_dtype == torch.float16 else 40.0
        chunk_eff = chunk if dmax <= 0 else max(1, min(chunk, int(safe_exp / dmax)))
    if chunk_override is not None:
        chunk_eff = max(1, min(chunk, int(chunk_override)))
    chunk = int(chunk_eff)
    n = (tokens + chunk - 1) // chunk
    pad = n * chunk - tokens
    mask_lt = torch.ones(chunk, chunk, device=dev, dtype=torch.bool).tril(-1)
    mask_le = torch.ones(chunk, chunk, device=dev, dtype=torch.bool).tril()

    Q, K, V, Gm, Bm, W = (_window(x, n, chunk, heads, pad) for x in (qf, kf, vf, gf, bf, wf))
    cum = Gm.cumsum(dim=-2)
    exp_cum = cum.exp()
    decay_inv = torch.exp(-cum.clamp(min=-40.0))
    bk_e = (Bm * K) * exp_cum
    q_e = Q * exp_cum
    k_ei = K * decay_inv

    # 一次 GEMM 同时产出 A 与 P：上行 Bk_e 给 A，下行 Q_e 给 P。
    both = torch.cat((bk_e, q_e), dim=-2) @ k_ei.transpose(-1, -2)
    a = both[..., :chunk, :].masked_fill(~mask_lt, 0.0).to(torch.float32)
    p = both[..., chunk:, :].masked_fill(~mask_le, 0.0)

    rhs = torch.cat((W * V, bk_e), dim=-1).to(torch.float32)
    if solve == "blocked":
        solve_block = block if chunk % block == 0 else chunk
        x_all = _blocked_solve(a, rhs, solve_block)
    else:
        lin = torch.eye(chunk, device=dev, dtype=torch.float32) + a
        x_all = torch.linalg.solve_triangular(lin, rhs, upper=False)
    x0, x1 = x_all[..., :dv], x_all[..., dv:]

    md = matmul_dtype
    state = torch.zeros(batch, heads, dk, dv, device=dev, dtype=torch.float32)
    outs = []
    for i in range(n):
        s_m = state.to(md)
        x = (x0[:, i] - x1[:, i] @ s_m.float()).to(md)
        outs.append((q_e[:, i] @ s_m + p[:, i] @ x).permute(0, 2, 1, 3))
        decay = exp_cum[:, i, :, -1]
        k_last = decay.unsqueeze(-2) * k_ei[:, i]
        state = decay.unsqueeze(-1).float() * state + (k_last.transpose(-1, -2) @ x).float()

    o = torch.cat(outs, dim=1)[:, :tokens].float()
    if divide_by_sqrt_t:
        denom = torch.sqrt(torch.arange(1, tokens + 1, device=dev,
                                        dtype=torch.float32)).view(1, tokens, 1, 1)
        o = o / denom
    return o
