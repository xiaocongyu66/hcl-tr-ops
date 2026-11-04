"""严格 CUDA RoPE 与其 PyTorch 参考，不提供静默 fallback。

融合契约：cos/sin 必须为 fp32，q/k 必须同 dtype（fp16/bf16/fp32），
输出跟随 q/k dtype，内部提升为 fp32 计算。不覆盖旧 torch 公式的全 half 缓存或混合 q/k dtype 语义。
bf16 是否可运行取决于 GPU，不代表 T4 支持保证。
"""

from __future__ import annotations

from typing import Any

import torch


_SUPPORTED_FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_contract(q, k, cos, sin, *, reference=False) -> None:
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise TypeError("RoPE 融合契约要求 cos/sin 为 float32；不支持 module.half() 缓存")
    supported = _SUPPORTED_FUSED_DTYPES + ((torch.float64,) if reference else ())
    if q.dtype != k.dtype or q.dtype not in supported:
        raise TypeError("RoPE 融合契约要求 q/k dtype 相同且为 float16/bfloat16/float32；"
                        "reference 额外支持 float64 gradcheck")


def _validate_common(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("q and k must have shape [B, T, H, D]")
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if q.device != k.device:
        raise ValueError("q and k must be on the same device")
    if q.shape[-1] <= 0 or q.shape[-1] % 2:
        raise ValueError("the rotary dimension D must be a positive even number")
    if any(stride == 0 for tensor in (q, k, cos, sin) for stride in tensor.stride()):
        raise ValueError("RoPE 不支持 zero-stride/expanded view 输入")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape [T, D/2]")
    if cos.shape != (q.shape[1], q.shape[3] // 2):
        raise ValueError("cos and sin must have shape [q.shape[1], q.shape[3] // 2]")
    if cos.device != q.device or sin.device != q.device:
        raise ValueError("cos and sin must be on q's device")
    if not q.is_floating_point() or not k.is_floating_point():
        raise TypeError("q and k must have floating-point dtype")
    if not cos.is_floating_point() or not sin.is_floating_point():
        raise TypeError("cos and sin must have floating-point dtype")
    if cos.requires_grad or sin.requires_grad:
        raise RuntimeError("fused_rope does not support gradients for cos or sin")


def reference_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合契约下的参考，并非旧 torch 路径所有 dtype 组合的替代。

    支持非连续布局；fp32 缓存与同 dtype q/k，输出跟随 q/k dtype。
    内部按 fp32 计算后转换回输入 dtype；额外允许同为 fp64 的 q/k 以保留 gradcheck；不支持缓存梯度。
    """
    _validate_common(q, k, cos, sin)
    _validate_contract(q, k, cos, sin, reference=True)
    c, s = cos, sin

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        y1 = x1 * c[None, :, None, :] - x2 * s[None, :, None, :]
        y2 = x1 * s[None, :, None, :] + x2 * c[None, :, None, :]
        return torch.cat((y1, y2), dim=-1).to(dtype=x.dtype)

    return rotate(q), rotate(k)


def _load_triton() -> tuple[Any, Any]:
    try:
        import triton
        import triton.language as tl
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("fused_rope requires Triton, but Triton is not installed") from exc
    return triton, tl


def _make_kernel():
    triton, tl = _load_triton()

    @triton.jit
    def rope_qk_kernel(
        q_ptr, k_ptr, out_q_ptr, out_k_ptr, cos_ptr, sin_ptr,
        q_sb, q_st, q_sh, q_sd, k_sb, k_st, k_sh, k_sd,
        oq_sb, oq_st, oq_sh, oq_sd, ok_sb, ok_st, ok_sh, ok_sd,
        c_st, c_sd, s_st, s_sd, n_time, n_head, n_dim, BLOCK: tl.constexpr, INVERSE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // (n_time * n_head)
        rem = pid - b * n_time * n_head
        t = rem // n_head
        h = rem - t * n_head
        offs = tl.arange(0, BLOCK)
        half = n_dim // 2
        mask = offs < half
        pair = offs % half
        c = tl.load(cos_ptr + t * c_st + pair * c_sd, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + t * s_st + pair * s_sd, mask=mask, other=0.0).to(tl.float32)

        q_base = b * q_sb + t * q_st + h * q_sh
        k_base = b * k_sb + t * k_st + h * k_sh
        oq_base = b * oq_sb + t * oq_st + h * oq_sh
        ok_base = b * ok_sb + t * ok_st + h * ok_sh
        q1 = tl.load(q_ptr + q_base + offs * q_sd, mask=mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + k_base + offs * k_sd, mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(q_ptr + q_base + (offs + half) * q_sd, mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + k_base + (offs + half) * k_sd, mask=mask, other=0.0).to(tl.float32)
        if INVERSE:
            oq1, oq2 = q1 * c + q2 * s, -q1 * s + q2 * c
            ok1, ok2 = k1 * c + k2 * s, -k1 * s + k2 * c
        else:
            oq1, oq2 = q1 * c - q2 * s, q1 * s + q2 * c
            ok1, ok2 = k1 * c - k2 * s, k1 * s + k2 * c
        tl.store(out_q_ptr + oq_base + offs * oq_sd, oq1, mask=mask)
        tl.store(out_k_ptr + ok_base + offs * ok_sd, ok1, mask=mask)
        tl.store(out_q_ptr + oq_base + (offs + half) * oq_sd, oq2, mask=mask)
        tl.store(out_k_ptr + ok_base + (offs + half) * ok_sd, ok2, mask=mask)
    return triton, rope_qk_kernel


def _launch(q, k, cos, sin, out_q, out_k, *, inverse: bool) -> None:
    triton, kernel = _make_kernel()
    d = q.shape[-1]
    block = triton.next_power_of_2(d // 2)
    grid = (q.shape[0] * q.shape[1] * q.shape[2],)
    kernel[grid](
        q, k, out_q, out_k, cos, sin,
        *q.stride(), *k.stride(), *out_q.stride(), *out_k.stride(),
        *cos.stride(), *sin.stride(), q.shape[1], q.shape[2], d, BLOCK=block, INVERSE=inverse,
        num_warps=4, enable_fp_fusion=False,
    )


class _FusedRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        out_dtype = q.dtype
        out_q = torch.empty(q.shape, device=q.device, dtype=out_dtype)
        out_k = torch.empty(k.shape, device=k.device, dtype=out_dtype)
        _launch(q, k, cos, sin, out_q, out_k, inverse=False)
        ctx.save_for_backward(cos, sin)
        ctx.q_shape, ctx.k_shape = q.shape, k.shape
        ctx.q_stride, ctx.k_stride = q.stride(), k.stride()
        ctx.q_dtype, ctx.k_dtype = q.dtype, k.dtype
        return out_q, out_k

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        if torch.is_grad_enabled():
            raise RuntimeError("fused_rope does not support double backward")
        cos, sin = ctx.saved_tensors
        dq = dk = None
        if ctx.needs_input_grad[0]:
            dq = torch.empty_strided(ctx.q_shape, ctx.q_stride, device=grad_q.device, dtype=ctx.q_dtype)
        if ctx.needs_input_grad[1]:
            dk = torch.empty_strided(ctx.k_shape, ctx.k_stride, device=grad_k.device, dtype=ctx.k_dtype)
        if dq is not None or dk is not None:
            # 内核同时写两侧；未请求梯度的一侧使用临时存储，返回仍为 None。
            tq = dq if dq is not None else torch.empty_like(grad_q)
            tk = dk if dk is not None else torch.empty_like(grad_k)
            _launch(grad_q, grad_k, cos, sin, tq, tk, inverse=True)
        return dq, dk, None, None


def fused_rope(q, k, cos, sin):
    """Strict Triton CUDA q/k RoPE.  Never falls back to ``reference_rope``."""
    _validate_common(q, k, cos, sin)
    _validate_contract(q, k, cos, sin)
    if not q.is_cuda:
        raise RuntimeError("fused_rope requires CUDA tensors")
    return _FusedRoPE.apply(q, k, cos, sin)


__all__ = ["fused_rope", "reference_rope"]
