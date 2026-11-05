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

_MASK_CACHE: dict = {}
_DENOM_CACHE: dict = {}


def _masks(chunk: int, device):
    key = (chunk, str(device))
    cached = _MASK_CACHE.get(key)
    if cached is None:
        base = torch.ones(chunk, chunk, device=device, dtype=torch.bool)
        cached = (base.tril(-1), base.tril())
        _MASK_CACHE[key] = cached
    return cached


def _inv_sqrt_t(tokens: int, device):
    key = (tokens, str(device))
    cached = _DENOM_CACHE.get(key)
    if cached is None:
        cached = torch.sqrt(torch.arange(1, tokens + 1, device=device,
                                         dtype=torch.float32)).view(1, tokens, 1, 1)
        _DENOM_CACHE[key] = cached
    return cached


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
    chunk_override: int | None = None, neg_clamp: float = 40.0,
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
    mask_lt, mask_le = _masks(chunk, dev)

    Q, K, V, Gm, Bm, W = (_window(x, n, chunk, heads, pad) for x in (qf, kf, vf, gf, bf, wf))
    cum = Gm.cumsum(dim=-2)
    exp_cum = cum.exp()

    # 中点平移：A/P 只依赖 exp(G_t-G_s)，与参考点 c 无关。
    # 取 c = (G_max+G_min)/2 后，两个因子的最大幅度从 e^{R} 降到 e^{R/2}
    # （R = chunk×dmax），同等 fp32 余量下可用 chunk 上限约翻倍。
    mid = 0.5 * (cum.amax(dim=-2, keepdim=True) + cum.amin(dim=-2, keepdim=True))
    shift = (cum - mid).clamp(min=-neg_clamp, max=neg_clamp)
    exp_pos = shift.exp()
    exp_neg = (-shift).exp()

    bk_e = (Bm * K) * exp_cum          # rhs 用真实尺度：exp(G) ≤ 1，不会溢出
    q_e = Q * exp_cum                  # 状态项同样保持真实尺度
    bk_e_s = (Bm * K) * exp_pos
    q_e_s = Q * exp_pos
    k_ei_s = K * exp_neg

    # 一次 GEMM 同时产出 A 与 P：上行 Bk_e 给 A，下行 Q_e 给 P。
    both = torch.cat((bk_e_s, q_e_s), dim=-2) @ k_ei_s.transpose(-1, -2)
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

    # UT 变换：out = Q_e@S + P@(X0 - X1@S) = P@X0 + (Q_e - P@X1)@S。
    # 两项都与状态无关，循环外各用一次批量 GEMM 算完；循环内每 chunk
    # 从 4 个 GEMM 降到 3 个。
    both_out = p @ torch.cat((x0, x1), dim=-1)   # 一次 GEMM 同时算 P@X0 与 P@X1
    px0, px1 = both_out[..., :dv], both_out[..., dv:]
    qeff = q_e - px1

    # 循环内只做 GEMM：把逐 chunk 的指数与衰减提到循环外批量算。
    # K_last = K·exp(G_last - G_s) ≤ K，指数恒非正，不需要巨大中间因子。
    cum_last = cum[..., -1:, :]
    k_last_all = K * torch.exp((cum_last - cum).clamp(min=-neg_clamp))
    decay_all = exp_cum[..., -1, :]

    md = matmul_dtype
    state = torch.zeros(batch, heads, dk, dv, device=dev, dtype=torch.float32)
    outs = []
    for i in range(n):
        s_m = state.to(md)
        x = (x0[:, i] - x1[:, i] @ s_m.float()).to(md)
        outs.append((px0[:, i] + qeff[:, i] @ s_m).permute(0, 2, 1, 3))
        decay = decay_all[:, i]
        state = decay.unsqueeze(-1).float() * state + (
            k_last_all[:, i].transpose(-1, -2) @ x).float()

    o = torch.cat(outs, dim=1)[:, :tokens].float()
    if divide_by_sqrt_t:
        o = o / _inv_sqrt_t(tokens, dev)
    return o
