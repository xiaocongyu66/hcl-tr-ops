"""GDN-2 状态转移的纯 PyTorch 参考实现（支持 batch 与 chunkwise）。

用途：
  1. 数值 fixture 的真值来源（reference），与官方 Triton kernel 逐元素对比。
  2. CPU/GPU 上的可训练实现（当 fla 不可用时）。

公式依据（SOURCE_LOCK: GDN-2 LOCKED, commit de8476de）：
  vendor/gdn2/lit_gpt/gdn2.py:57-62 docstring：
      S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T

与 QED 论文 eq.(8) 等价的展开式（fixture 用此形式便于逐项对齐）：
      S_bar_t = Diag(exp(g_t)) S_{t-1}
      S_t     = S_bar_t + k_t [ w_t ⊙ v_t - S_bar_t^T (b_t ⊙ k_t) ]^T

显存关键：逐步递推时**不保存状态历史** S_all（[B,T,H,dk,dv] 在 8K 序列下会爆），
只保留当前状态并在递推中直接产生读出 o_t。
"""

from __future__ import annotations

import functools
import math

import torch


STATS = {}


def _fp32_region(fn):
    """强制函数体内的矩阵运算走 fp32。

    原因：AMP 的 autocast 会把 einsum/matmul 强制降到 fp16/bf16——即使张量本身
    已被 .float()。GDN-2 的状态递推误差会沿 8K 步累积，必须在 fp32 下算。
    （实测：不包时 einsum 输出为 bf16，与纯 fp32 相差 4e-3。）
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        dev = None
        for a in (*args, *kwargs.values()):
            if isinstance(a, torch.Tensor):
                dev = a.device.type
                break
        if dev in ("cuda", "cpu"):
            with torch.autocast(device_type=dev, enabled=False):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def gdn2_step(
    S_prev: torch.Tensor,   # [B, H, dk, dv] 或 [H, dk, dv]
    k: torch.Tensor,        # [B, H, dk] 或 [H, dk]
    v: torch.Tensor,        # [B, H, dv] 或 [H, dv]
    g: torch.Tensor,        # [B, H, dk] 或 [H, dk]  log-decay（<=0）
    b: torch.Tensor,        # [B, H, dk] 或 [H, dk] 通道级 erase gate
    w: torch.Tensor,        # [B, H, dv] 或 [H, dv] 通道级 write gate
    l2norm_qk: bool = True,
) -> torch.Tensor:
    """单步 GDN-2 状态更新，返回 S_t（形状同 S_prev，fp32）。"""
    S_prev = S_prev.to(torch.float32)
    k, v, g, b, w = (t.to(torch.float32) for t in (k, v, g, b, w))
    if l2norm_qk:
        k = _l2norm(k)

    S_bar = g.exp().unsqueeze(-1) * S_prev                      # 行缩放（key 轴）
    # r = S_bar^T (b ⊙ k)
    r = torch.einsum("...hk,...hkv->...hv", b * k, S_bar)
    delta = w * v - r
    return S_bar + k.unsqueeze(-1) * delta.unsqueeze(-2)


def gdn2_chunk(
    S_init: torch.Tensor,   # [H, dk, dv]
    k: torch.Tensor,        # [T, H, dk]
    v: torch.Tensor,        # [T, H, dv]
    g: torch.Tensor,        # [T, H, dk]
    b: torch.Tensor,        # [T, H, dk]
    w: torch.Tensor,        # [T, H, dv]
    l2norm_qk: bool = True,
    store_states: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐 token 展开 gdn2_step。返回 (S_T, states 或 None)。

    store_states=False 时不累积历史（训练大序列时用，避免 [T,H,dk,dv] 爆显存）。
    """
    S = S_init.to(torch.float32)
    states = [] if store_states else None
    for t in range(k.shape[0]):
        S = gdn2_step(S, k[t], v[t], g[t], b[t], w[t], l2norm_qk=l2norm_qk)
        if store_states:
            states.append(S)
    return S, (torch.stack(states, dim=0) if store_states else None)


@_fp32_region
def gdn2_recurrent_forward(
    q: torch.Tensor,        # [B, T, H, dk]
    k: torch.Tensor,        # [B, T, H, dk]
    v: torch.Tensor,        # [B, T, H, dv]
    g: torch.Tensor,        # [B, T, H, dk]
    b: torch.Tensor,        # [B, T, H, dk]
    w: torch.Tensor,        # [B, T, H, dv]
    divide_by_sqrt_t: bool = True,
    l2norm_qk: bool = True,
    lam_h: torch.Tensor | None = None,   # [H] QED 每头强度；None 表示不加 QED
    ckpt_chunk: int = 256,
    use_checkpoint: bool = True,
    gq: torch.Tensor | None = None,      # [B,T,H] SLA 读门控（在 l2norm 之后乘）
    gk: torch.Tensor | None = None,      # [B,T,H] SLA 写门控
) -> torch.Tensor:
    """批量逐步递推，直接产出读出 o，**不保存状态历史**。

    返回 o: [B, T, H, dv]

    显存：状态本身只有 [B,H,dk,dv]，但**逐 token 的 autograd 图会为每一步保留中间量**，
    8K 序列下约 10+GB（实测 OOM）。因此按 `ckpt_chunk` 分块做梯度检查点：
    反向时重算块内激活，峰值从 O(T) 降到 O(ckpt_chunk)。
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]

    # l2norm 与状态累加用 fp32；大矩阵乘可走 fp16（T4 张量核 fp16 是 fp32 的 8 倍）
    qf, kf, vf, gf, bf, wf = (t.to(torch.float32) for t in (q, k, v, g, b, w))
    if l2norm_qk:
        kf = _l2norm(kf)
        qf = _l2norm(qf)
    # SLA 门控必须在 l2norm 之后：l2norm 是尺度不变的，先乘会被约掉
    if gq is not None:
        qf = qf * gq.to(torch.float32).unsqueeze(-1)
    if gk is not None:
        kf = kf * gk.to(torch.float32).unsqueeze(-1)

    lam = lam_h.to(torch.float32).view(1, H, 1, 1) if lam_h is not None else None

    def scan_chunk(S, q_c, k_c, v_c, g_c, b_c, w_c, t0):
        outs = []
        Tc = q_c.shape[1]
        for i in range(Tc):
            t = t0 + i
            q_t, k_t, v_t = q_c[:, i], k_c[:, i], v_c[:, i]
            g_t, b_t, w_t = g_c[:, i], b_c[:, i], w_c[:, i]

            S_bar = g_t.exp().unsqueeze(-1) * S
            r = torch.einsum("bhk,bhkv->bhv", b_t * k_t, S_bar)
            S = S_bar + k_t.unsqueeze(-1) * (w_t * v_t - r).unsqueeze(-2)

            if lam is not None:
                from .qed import project_out_key
                d = project_out_key(b_t * q_t, k_t)
                s_bar_d = torch.einsum("bhk,bhkv->bhv", d, S_bar)
                S = S - lam * k_t.unsqueeze(-1) * s_bar_d.unsqueeze(-2)

            o_t = torch.einsum("bhk,bhkv->bhv", q_t, S)
            if divide_by_sqrt_t:
                o_t = o_t / (t + 1) ** 0.5
            outs.append(o_t)
        return S, torch.stack(outs, dim=1)

    S = torch.zeros(B, H, dk, dv, device=q.device, dtype=torch.float32)
    pieces = []
    step = max(1, ckpt_chunk)
    for start in range(0, T, step):
        end = min(T, start + step)
        args = (S, qf[:, start:end], kf[:, start:end], vf[:, start:end],
                gf[:, start:end], bf[:, start:end], wf[:, start:end], start)
        if use_checkpoint and torch.is_grad_enabled() and any(
                a.requires_grad for a in args[1:7]):
            from torch.utils.checkpoint import checkpoint
            S, o_c = checkpoint(scan_chunk, *args, use_reentrant=False)
        else:
            S, o_c = scan_chunk(*args)
        pieces.append(o_c)

    return torch.cat(pieces, dim=1)


def gdn2_output(
    S_T: torch.Tensor | None,
    q: torch.Tensor,
    S_all: torch.Tensor,
    divide_by_sqrt_t: bool = True,
    learnable_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """给定逐步状态做读出（fixture 用；训练路径请用 gdn2_recurrent_forward）。

    divide_by_sqrt_t 是计划书 §4.1 超出官方实现的要求：官方只有 SiLU-gated RMSNorm，
    没有序列长度归一化。此处实现为 `o_t / sqrt(t+1)`（t 从 0 起）。
    """
    q = q.to(torch.float32)
    batched = q.dim() == 4
    if batched:
        o = torch.einsum("bthk,bthkv->bthv", q, S_all)
        T = q.shape[1]
        denom_shape = (1, T, 1, 1)
    else:
        o = torch.einsum("thk,thkv->thv", q, S_all)
        T = q.shape[0]
        denom_shape = (T, 1, 1)
    if divide_by_sqrt_t:
        denom = torch.sqrt(torch.arange(1, T + 1, device=q.device, dtype=torch.float32)).view(*denom_shape)
        o = o / denom
    if learnable_scale is not None:
        o = o * learnable_scale
    return o


def _effective_chunk(gf, chunk, matmul_dtype, chunk_eff_in, chunk_override):
    """共享现有自适应约束；缓存值优先，override 最后覆盖。"""
    safe_exp = 10.0 if matmul_dtype == torch.float16 else 40.0
    if chunk_eff_in is not None:
        chunk_eff = max(1, min(chunk, chunk_eff_in))
    else:
        dmax = float((-gf.detach()).amax()) if gf.shape[1] > 0 else 0.0
        if not math.isfinite(dmax):
            raise ValueError("GDN2 衰减包含 NaN/Inf")
        chunk_eff = chunk if dmax <= 0 else max(1, min(chunk, int(safe_exp / dmax)))
        STATS["dmax"] = dmax
    if chunk_override is not None:
        chunk_eff = max(1, min(chunk, chunk_override))
    STATS["chunk_eff"] = chunk_eff
    return chunk_eff


@_fp32_region
def gdn2_chunkwise_forward(
    q: torch.Tensor,        # [B, T, H, dk]
    k: torch.Tensor,        # [B, T, H, dk]
    v: torch.Tensor,        # [B, T, H, dv]
    g: torch.Tensor,        # [B, T, H, dk]
    b: torch.Tensor,        # [B, T, H, dk]
    w: torch.Tensor,        # [B, T, H, dv]
    chunk: int = 64,
    divide_by_sqrt_t: bool = True,
    l2norm_qk: bool = True,
    gq: torch.Tensor | None = None,      # [B,T,H] SLA 读门控
    gk: torch.Tensor | None = None,      # [B,T,H] SLA 写门控
    matmul_dtype: torch.dtype = torch.float32,   # 矩阵乘精度；状态恒 fp32
    sub_chunk: int | None = 64,                  # 保留字段（当前实现未使用）
    chunk_override: int | None = None,           # 强制 chunk 大小（仅用于基准对照）
    chunk_eff_in: int | None = None,             # 调用方缓存的自适应 chunk（避免每前向同步）
    solve_mode: str = "solver",                  # "solver"(cuSOLVER) | "neumann"(纯 GEMM，可 CUDA Graph 捕获)
) -> torch.Tensor:
    """GDN-2 的 chunkwise 并行前向（不逐 token，按 chunk 做矩阵运算）。

    动机：逐 token 的 Python 循环是吞吐瓶颈（实测 ~1.6k tok/s，10B 需上千小时）。
    这里每个 chunk 内把 delta-rule 递推化为**下三角线性方程组**一次求解，
    Python 迭代次数从 T 降到 T/chunk。

    推导（单个 chunk，状态 S0 从上一 chunk 传入）：
      记 G_t = Σ_{i<=t} g_i（chunk 内累积 log 衰减），K=l2norm(k)，Bk=b⊙K，u=w⊙v
      递推:  S_t = diag(e^{G_t}) S_{t-1} + K_t (u_t - r_t)^T,  r_t = (Bk_t)^T diag(e^{g_t}) S_{t-1}
      展开后令 x_t = u_t - r_t，可得严格下三角系统：
        (I + A) x = u - c,   A[t,s] = Σ_k Bk_t,k K_s,k e^{G_t,k - G_s,k} (s<t)
                             c_t   = (Bk_t ⊙ e^{G_t})^T S0
      再解出 S_t 与 o_t（o_t = q_t^T S_t），末态 S_T 传给下一 chunk。
    注意：e^{G_t-G_s} 中 t>=s 时指数 ≤ 0，先 clamp 再 exp，避免溢出。

    只支持非 QED 路径（QED 的额外 erase 项请用 gdn2_recurrent_forward）。
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    dev = q.device

    # l2norm 与状态累加用 fp32；大矩阵乘可走 fp16（T4 张量核 fp16 是 fp32 的 8 倍）
    qf, kf, vf, gf, bf, wf = (t.to(torch.float32) for t in (q, k, v, g, b, w))
    if l2norm_qk:
        kf = _l2norm(kf)
        qf = _l2norm(qf)
    # SLA 门控在 l2norm 之后（否则被归一化抵消）
    if gq is not None:
        qf = qf * gq.to(torch.float32).unsqueeze(-1)
    if gk is not None:
        kf = kf * gk.to(torch.float32).unsqueeze(-1)

    md = matmul_dtype
    qf, kf, vf, bf, wf = (t.to(md) for t in (qf, kf, vf, bf, wf))
    gf_m = gf.to(md)
    S = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    outs = []

    # 与下方 invG 的 clamp(-40) 保持一致；这不是所有输入的误差保证。
    chunk_eff = _effective_chunk(gf, chunk, matmul_dtype, chunk_eff_in, chunk_override)

    for start in range(0, T, chunk_eff):
        end = min(T, start + chunk_eff)
        C = end - start
        K = kf[:, start:end]          # [B,C,H,dk]
        Bk = bf[:, start:end] * K
        Q = qf[:, start:end]
        U = wf[:, start:end] * vf[:, start:end]   # [B,C,H,dv]
        G = torch.cumsum(gf_m[:, start:end], dim=1)  # [B,C,H,dk]

        expG = G.exp()                                  # [B,C,H,dk]
        # 因式分解 exp(G_t-G_s) = exp(G_t)·exp(-G_s)：两次矩阵乘，不物化 [B,C,C,H,dk]。
        # 因 chunk_eff × 衰减 <= safe_exp，两个因子都不会溢出（clamp 仅是兜底）。
        tril_le = torch.ones(C, C, device=dev, dtype=torch.bool).tril()
        tril_lt = torch.ones(C, C, device=dev, dtype=torch.bool).tril(-1)
        invG = torch.exp(-G.clamp(min=-40.0))

        Bk_e = Bk * expG                                # [B,C,H,dk]
        Q_e = Q * expG
        Ke_i = K * invG

        A = torch.einsum("bthk,bshk->bhts", Bk_e, Ke_i)
        A = A.masked_fill(~tril_lt, 0.0)
        c = torch.einsum("bthk,bhkv->bthv", Bk_e, S.to(md))
        rhs = (U - c).permute(0, 2, 1, 3).float()     # [B,H,C,dv]
        if solve_mode == "neumann":
            # (I+A)^-1 = Σ(-A)^k，A 严格下三角 → C 步后精确。
            # 只用 GEMM，不含 cuSOLVER 的内部同步，因此可被 CUDA Graph 捕获。
            Af = A.float()
            x = rhs
            for _ in range(C):
                x = rhs - torch.bmm(Af.reshape(-1, C, C), x.reshape(-1, C, rhs.shape[-1])).reshape_as(rhs)
        else:
            L = torch.eye(C, device=dev, dtype=torch.float32).view(1, 1, C, C) + A.float()
            x = torch.linalg.solve_triangular(L, rhs, upper=False)
        x = x.to(md).permute(0, 2, 1, 3)

        decay_out = torch.einsum("bthk,bhkv->bthv", Q_e, S.to(md))
        P = torch.einsum("bthk,bshk->bhts", Q_e, Ke_i)
        P = P.masked_fill(~tril_le, 0.0)
        outs.append(decay_out + torch.einsum("bhts,bshv->bthv", P, x))

        last = C - 1
        K_last = expG[:, last:last + 1] * Ke_i
        S_new = expG[:, last].unsqueeze(-1).float() * S + torch.einsum(
            "bshk,bshv->bhkv", K_last, x).float()
        S = S_new   # 状态恒 fp32，避免 8K 步累积误差

    o = torch.cat(outs, dim=1).float()
    if divide_by_sqrt_t:
        denom = torch.sqrt(torch.arange(1, T + 1, device=dev,
                                        dtype=torch.float32)).view(1, T, 1, 1)
        o = o / denom
    return o


@_fp32_region
def gdn2_chunkwise_batched_forward(
    q, k, v, g, b, w, chunk=64, divide_by_sqrt_t=True, l2norm_qk=True,
    matmul_dtype=torch.float32, *, gq=None, gk=None, sub_chunk=64,
    chunk_override=None, chunk_eff_in=None, solve_mode="solver",
    precompute_chunks=16,
):
    """跨 chunk 批量预计算 L^-1[U | Bk_e]，状态仍按 chunk 串行传播。

    与旧 forward 一样从零状态开始，只返回 [B,T,H,dv] fp32 输出。
    自适应 chunk、SLA 和精度选项沿用旧路径；浮点运算重排不保证逐位相同。
    每窗口最多 precompute_chunks 块，工作张量为 O(B*H*N*(C*C +
    C*(dk+dv)))，不物化 C*C*dk。训练 autograd 仍可能保留所有窗口的
    激活；需要时由调用方 checkpoint。sub_chunk 与旧接口一样保留但不使用。
    """
    if precompute_chunks < 1:
        raise ValueError("precompute_chunks must be positive")
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    dev = q.device
    md = matmul_dtype
    qf, kf, vf, gf, bf, wf = (t.to(torch.float32) for t in (q, k, v, g, b, w))
    if l2norm_qk:
        kf = _l2norm(kf)
        qf = _l2norm(qf)
    if gq is not None:
        qf = qf * gq.to(torch.float32).unsqueeze(-1)
    if gk is not None:
        kf = kf * gk.to(torch.float32).unsqueeze(-1)
    qf, kf, vf, bf, wf = (t.to(md) for t in (qf, kf, vf, bf, wf))
    gf_m = gf.to(md)
    C = _effective_chunk(gf, chunk, md, chunk_eff_in, chunk_override)
    tril_le = torch.ones(C, C, device=dev, dtype=torch.bool).tril()
    tril_lt = tril_le & ~torch.eye(C, device=dev, dtype=torch.bool)
    S = torch.zeros(B, H, dk, dv, device=dev, dtype=torch.float32)
    outs = []

    for start in range(0, T, precompute_chunks * C):
        end = min(T, start + precompute_chunks * C)
        N = (end - start + C - 1) // C
        pad = N * C - (end - start)

        def window(t):
            part = t[:, start:end]
            if pad:
                part = torch.cat((part, part.new_zeros(B, pad, H, t.shape[-1])), dim=1)
            # [B,N,H,C,d]; padding is an identity state transition (k=g=0).
            return part.reshape(B, N, C, H, t.shape[-1]).permute(0, 1, 3, 2, 4)

        Q, K, V, Gm, Bm, W = (window(t) for t in (qf, kf, vf, gf_m, bf, wf))
        G = Gm.cumsum(dim=-2)
        expG = G.exp()
        Bk_e = (Bm * K) * expG
        Q_e = Q * expG
        Ke_i = K * torch.exp(-G.clamp(min=-40.0))
        A = (Bk_e @ Ke_i.transpose(-1, -2)).masked_fill(~tril_lt, 0.0).float()
        rhs = torch.cat((W * V, Bk_e), dim=-1).float()
        if solve_mode == "neumann":
            # Preserve the existing exact finite triangular iteration, not a truncation.
            X = rhs
            for _ in range(C):
                X = rhs - A @ X
        else:
            L = torch.eye(C, device=dev, dtype=torch.float32) + A
            X = torch.linalg.solve_triangular(L, rhs, upper=False)
        X0, X1 = X[..., :dv], X[..., dv:]
        P = (Q_e @ Ke_i.transpose(-1, -2)).masked_fill(~tril_le, 0.0)

        for n in range(N):
            Sm = S.to(md)
            # Keep the solved coefficients fp32 until the state-dependent subtraction.
            x = (X0[:, n] - X1[:, n] @ Sm.float()).to(md)
            out = Q_e[:, n] @ Sm + P[:, n] @ x
            outs.append(out.permute(0, 2, 1, 3))
            decay = expG[:, n, :, -1]
            K_last = decay.unsqueeze(-2) * Ke_i[:, n]
            S = decay.unsqueeze(-1).float() * S + (K_last.transpose(-1, -2) @ x).float()

    o = torch.cat(outs, dim=1)[:, :T].float()
    if divide_by_sqrt_t:
        denom = torch.sqrt(torch.arange(1, T + 1, device=dev, dtype=torch.float32)).view(1, T, 1, 1)
        o = o / denom
    return o
