"""融合 chunkwise 实现与现有批量实现的数值一致性（CPU）。"""
import pytest
import torch

from hcl_tr.models.gdn2_chunk_opt import _blocked_solve, chunkwise_fused
from hcl_tr.models.gdn2_ref import gdn2_chunkwise_batched_forward


def _inputs(batch=2, tokens=40, heads=2, dk=8, dv=6, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, tokens, heads, dk)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    v = torch.randn((batch, tokens, heads, dv), generator=gen)
    g = -torch.rand(shape, generator=gen)
    b = torch.rand(shape, generator=gen)
    w = torch.rand((batch, tokens, heads, dv), generator=gen)
    return q, k, v, g, b, w


@pytest.mark.skipif(not torch.cuda.is_available(), reason='需要 CUDA 验证 fp16 autocast')
def test_fused_disables_autocast_on_cuda():
    """回归：autocast(fp16) 下 @ 会被降精度，指数因子溢出成 inf → NaN。

    算子内部必须显式关闭 autocast，输出保持 fp32 且与无 autocast 的结果一致。
    g 放大后单因子达到 ~e^14，远超 fp16 上限 65504，能真实触发溢出。
    """
    from hcl_tr.models.gdn2_chunk_opt import _chunkwise_fused_impl

    q, k, v, g, b, w = _inputs(batch=1, tokens=32, heads=2, dk=8, dv=6, seed=3)
    args = [t.cuda() for t in (q, k, v, g * 3.5, b, w)]
    expected = chunkwise_fused(*args, chunk=8, solve='reference', neg_clamp=80.0)
    with torch.autocast('cuda', dtype=torch.float16):
        raw = _chunkwise_fused_impl(*args, chunk=8, solve='reference', neg_clamp=80.0)
        got = chunkwise_fused(*args, chunk=8, solve='reference', neg_clamp=80.0)
    assert not torch.isfinite(raw).all(), '该用例应能复现 autocast 溢出'
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('solve', ['reference', 'blocked'])
@pytest.mark.parametrize('chunk,padding', [(8, 0), (8, 4), (10, 0)])
def test_matches_reference_implementation(solve, chunk, padding):
    tokens = 40 + padding
    args = _inputs(tokens=tokens)
    expected = gdn2_chunkwise_batched_forward(*args, chunk=chunk, divide_by_sqrt_t=True,
                                              l2norm_qk=True)
    got = chunkwise_fused(*args, chunk=chunk, solve=solve, block=2)
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)


def test_without_l2norm_and_divisor():
    args = _inputs(tokens=24)
    expected = gdn2_chunkwise_batched_forward(*args, chunk=6, divide_by_sqrt_t=False,
                                              l2norm_qk=False)
    got = chunkwise_fused(*args, chunk=6, divide_by_sqrt_t=False, l2norm_qk=False, block=3)
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-5)


def test_fused_backward_matches_reference_for_all_inputs():
    base = _inputs(batch=1, tokens=24, heads=2, dk=4, dv=4, seed=7)
    fused_inputs = [x.detach().clone().requires_grad_(True) for x in base]
    ref_inputs = [x.detach().clone().requires_grad_(True) for x in base]
    got = chunkwise_fused(*fused_inputs, chunk=6, chunk_eff=6, solve='blocked', block=3)
    ref = gdn2_chunkwise_batched_forward(*ref_inputs, chunk=6, chunk_eff_in=6)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=2e-5)
    weight = torch.randn_like(got)
    (got * weight).sum().backward()
    (ref * weight).sum().backward()
    for actual, expected in zip(fused_inputs, ref_inputs):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=2e-5)


def test_blocked_solve_rejects_indivisible_block():
    a = torch.zeros(2, 3, 10, 10)
    rhs = torch.zeros(2, 3, 10, 4)
    with pytest.raises(ValueError, match='整除'):
        _blocked_solve(a, rhs, 4)


def test_blocked_solve_matches_exact_inverse():
    gen = torch.Generator().manual_seed(3)
    c, m = 16, 5
    a = torch.tril(torch.randn(2, c, c, generator=gen), diagonal=-1)
    rhs = torch.randn(2, c, m, generator=gen)
    got = _blocked_solve(a, rhs, block=4)
    lin = torch.eye(c) + a
    want = torch.linalg.solve_triangular(lin, rhs, upper=False)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_invalid_solve_mode():
    args = _inputs(tokens=8)
    with pytest.raises(ValueError, match='solve'):
        chunkwise_fused(*args, chunk=4, solve='magic')
