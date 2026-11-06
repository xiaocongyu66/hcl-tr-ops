"""Triton 融合递推：把逐 chunk 的 GEMM 与逐元素更新收进一个 kernel。

递推（与 eager 循环逐项一致）：
    x  = X0 − X1·S
    o  = PX0 + QEFF·S
    S' = dec⊙S + K_Lᵀ·x

注意：不能改写成 S' = dec⊙S + (K_LᵀX0) − (K_LᵀX1)·S。当指数因子达到 e^59 时，
矩阵乘后再相减会发生灾难性相消（实测偏差 0.85，而 kernel 本身误差只有 4e-5）。
因此必须在循环内先算 x，再做 K_Lᵀ·x。

所有张量布局为 [B*H, N, C, D]，C 为块大小（要求 2 的幂），N 为块数。
"""

from __future__ import annotations

import torch

try:  # 本地无 triton 时模块仍可导入（测试会跳过）
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _recur_kernel(X0, X1, KL, PX0, QEFF, DEC, O, S_SAVED,
                      N, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      DV_T: tl.constexpr, SAVE: tl.constexpr):
        bh = tl.program_id(0)
        dt = tl.program_id(1)
        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = dt * DV_T + tl.arange(0, DV_T)

        base_v = bh * N * C * DV
        base_k = bh * N * C * DK
        base_s = bh * N * DK * DV

        s = tl.zeros((DK, DV_T), dtype=tl.float32)
        for i in tl.range(0, N):
            if SAVE:
                tl.store(S_SAVED + base_s + i * DK * DV
                         + offs_k[:, None] * DV + offs_v[None, :], s)
            q = tl.load(QEFF + base_k + i * C * DK
                        + offs_c[:, None] * DK + offs_k[None, :])
            px = tl.load(PX0 + base_v + i * C * DV
                         + offs_c[:, None] * DV + offs_v[None, :])
            o = px + tl.dot(q, s, input_precision="ieee")
            tl.store(O + base_v + i * C * DV
                     + offs_c[:, None] * DV + offs_v[None, :], o)

            x0 = tl.load(X0 + base_v + i * C * DV
                         + offs_c[:, None] * DV + offs_v[None, :])
            x1 = tl.load(X1 + base_k + i * C * DK
                         + offs_c[:, None] * DK + offs_k[None, :])
            x = x0 - tl.dot(x1, s, input_precision="ieee")
            kl = tl.load(KL + base_k + i * C * DK
                         + offs_c[:, None] * DK + offs_k[None, :])
            dec = tl.load(DEC + (bh * N + i) * DK + offs_k)
            s = dec[:, None] * s + tl.dot(tl.trans(kl), x, input_precision="ieee")

    @triton.jit
    def _recur_bwd_kernel(X0, X1, KL, QEFF, DEC, S_SAVED, DO,
                          DX0, DX1, DKL, DQEFF, DDEC,
                          N, C: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                          DV_T: tl.constexpr, NT: tl.constexpr):
        bh = tl.program_id(0)
        dt = tl.program_id(1)
        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = dt * DV_T + tl.arange(0, DV_T)

        base_v = bh * N * C * DV
        base_k = bh * N * C * DK
        base_s = bh * N * DK * DV

        ds = tl.zeros((DK, DV_T), dtype=tl.float32)
        for j in tl.range(0, N):
            i = N - 1 - j
            do = tl.load(DO + base_v + i * C * DV
                         + offs_c[:, None] * DV + offs_v[None, :])
            s = tl.load(S_SAVED + base_s + i * DK * DV
                        + offs_k[:, None] * DV + offs_v[None, :])
            q = tl.load(QEFF + base_k + i * C * DK
                        + offs_c[:, None] * DK + offs_k[None, :])
            # 这三个梯度要对整个 dv 求和，每个 dv tile 只能贡献部分和，
            # 因此写到 tile 独立槽位，由调用方求和（直接写会互相覆盖）。
            tl.store(DQEFF + ((bh * N + i) * NT + dt) * C * DK
                     + offs_c[:, None] * DK + offs_k[None, :],
                     tl.dot(do, tl.trans(s), input_precision="ieee"))

            x0 = tl.load(X0 + base_v + i * C * DV
                         + offs_c[:, None] * DV + offs_v[None, :])
            x1 = tl.load(X1 + base_k + i * C * DK
                         + offs_c[:, None] * DK + offs_k[None, :])
            x = x0 - tl.dot(x1, s, input_precision="ieee")
            tl.store(DKL + ((bh * N + i) * NT + dt) * C * DK
                     + offs_c[:, None] * DK + offs_k[None, :],
                     tl.dot(x, tl.trans(ds), input_precision="ieee"))
            tl.store(DDEC + ((bh * N + i) * NT + dt) * DK + offs_k,
                     tl.sum(ds * s, axis=1))

            kl = tl.load(KL + base_k + i * C * DK
                         + offs_c[:, None] * DK + offs_k[None, :])
            dx = tl.dot(kl, ds, input_precision="ieee")
            tl.store(DX0 + base_v + i * C * DV
                     + offs_c[:, None] * DV + offs_v[None, :], dx)
            tl.store(DX1 + ((bh * N + i) * NT + dt) * C * DK
                     + offs_c[:, None] * DK + offs_k[None, :],
                     -tl.dot(dx, tl.trans(s), input_precision="ieee"))

            dec = tl.load(DEC + (bh * N + i) * DK + offs_k)
            ds = (dec[:, None] * ds
                  - tl.dot(tl.trans(x1), dx, input_precision="ieee")
                  + tl.dot(tl.trans(q), do, input_precision="ieee"))


def recurrence_eager(px0, qeff, x0, x1, kl, dec):
    """与 kernel 等价；用于反向重算与 CPU 测试。"""
    batch_heads, n, chunk, dk = qeff.shape
    dv = px0.shape[-1]
    s = torch.zeros(batch_heads, dk, dv, device=px0.device, dtype=px0.dtype)
    outs = []
    for i in range(n):
        outs.append(px0[:, i] + qeff[:, i] @ s)
        x = x0[:, i] - x1[:, i] @ s
        s = dec[:, i].unsqueeze(-1) * s + kl[:, i].transpose(-1, -2) @ x
    return torch.stack(outs, dim=1)


def recurrence_triton(x0, x1, kl, px0, qeff, dec, dv_tile: int = 32,
                      num_warps: int = 8, states=None):
    if not HAS_TRITON:
        raise RuntimeError("triton 不可用")
    if px0.device.type != "cuda":
        raise RuntimeError("Triton 递推需要 CUDA")
    for name, t in (("x0", x0), ("x1", x1), ("kl", kl), ("px0", px0),
                    ("qeff", qeff), ("dec", dec)):
        if not t.is_contiguous():
            raise ValueError(f"recurrence_triton 需要连续张量：{name} 非连续")
    bh, n, chunk, dk = qeff.shape
    dv = px0.shape[-1]
    out = torch.empty_like(px0)
    dv_tile = max(16, min(dv_tile, dv))
    grid = (bh, triton.cdiv(dv, dv_tile))
    save = states is not None
    _recur_kernel[grid](x0, x1, kl, px0, qeff, dec, out,
                        states if save else px0, n, chunk, dk, dv, dv_tile, save,
                        num_warps=num_warps)
    return out


def recurrence_triton_backward(do, x0, x1, kl, qeff, dec, states,
                               dv_tile: int = 32, num_warps: int = 8):
    """反向：基于前向保存的 S_i 逆序递推，一次 kernel 出全部梯度。"""
    for name, t in (("x0", x0), ("x1", x1), ("kl", kl), ("qeff", qeff),
                    ("dec", dec), ("states", states)):
        if not t.is_contiguous():
            raise ValueError(f"recurrence_triton_backward 需要连续张量：{name} 非连续")
    bh, n, chunk, dk = qeff.shape
    dv = x0.shape[-1]
    dv_tile = max(16, min(dv_tile, dv))
    nt = triton.cdiv(dv, dv_tile)
    do = do.contiguous()
    dx0 = torch.empty_like(x0)
    # 跨 dv 求和的三个梯度按 tile 分槽存放，最后求和。
    part = (bh, n, nt, chunk, dk)
    dx1_t = torch.empty(part, device=x0.device, dtype=torch.float32)
    dkl_t = torch.empty(part, device=x0.device, dtype=torch.float32)
    dqeff_t = torch.empty(part, device=x0.device, dtype=torch.float32)
    ddec_t = torch.empty(bh, n, nt, dk, device=x0.device, dtype=torch.float32)
    _recur_bwd_kernel[(bh, nt)](
        x0, x1, kl, qeff, dec, states, do,
        dx0, dx1_t, dkl_t, dqeff_t, ddec_t,
        n, chunk, dk, dv, dv_tile, nt, num_warps=num_warps)
    return (dx0, dx1_t.sum(dim=2), dkl_t.sum(dim=2), dqeff_t.sum(dim=2),
            ddec_t.sum(dim=2))


class _TritonRecurrence(torch.autograd.Function):
    """前向与反向都走融合 kernel；反向使用前向保存的逐 chunk 状态。"""

    @staticmethod
    def forward(ctx, x0, x1, kl, px0, qeff, dec):
        bh, n, chunk, dk = qeff.shape
        dv = px0.shape[-1]
        states = torch.empty(bh, n, dk, dv, device=px0.device, dtype=torch.float32)
        out = recurrence_triton(x0, x1, kl, px0, qeff, dec, states=states)
        ctx.save_for_backward(x0, x1, kl, qeff, dec, states)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x0, x1, kl, qeff, dec, states = ctx.saved_tensors
        dx0, dx1, dkl, dqeff, ddec = recurrence_triton_backward(
            grad_out, x0, x1, kl, qeff, dec, states)
        # d(px0) = dO（逐元素恒等），无需 kernel 写回。
        return dx0, dx1, dkl, grad_out, dqeff, ddec


def recurrence(x0, x1, kl, px0, qeff, dec):
    """自动选择：CUDA 且 fp32 时走 Triton，否则用 eager。"""
    if (HAS_TRITON and px0.is_cuda and px0.dtype == torch.float32
            and torch.is_grad_enabled()):
        return _TritonRecurrence.apply(x0, x1, kl, px0, qeff, dec)
    if HAS_TRITON and px0.is_cuda and px0.dtype == torch.float32:
        return recurrence_triton(x0, x1, kl, px0, qeff, dec)
    return recurrence_eager(px0, qeff, x0, x1, kl, dec)
