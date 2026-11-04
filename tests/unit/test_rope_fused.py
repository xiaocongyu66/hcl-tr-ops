import pytest
import torch

from hcl_tr.models.rope_fused import fused_rope, reference_rope


def _cache(t, d, *, device="cpu", dtype=torch.float32):
    angles = torch.randn(t, d // 2, device=device, dtype=torch.float32) * 0.2
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _independent(x, cos, sin):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)


def test_reference_rope_noncontiguous_and_independent_formula():
    torch.manual_seed(1)
    q0 = torch.randn(2, 7, 3, 65, dtype=torch.float64)
    k0 = torch.randn(2, 7, 3, 65, dtype=torch.float64)
    q = q0[..., 1:]
    k = k0[..., :64]
    cos, sin = _cache(7, 64)
    actual_q, actual_k = reference_rope(q, k, cos, sin)
    expected_q = _independent(q, cos.double(), sin.double())
    expected_k = _independent(k, cos.double(), sin.double())
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    assert not q.is_contiguous()
    assert actual_q.dtype == torch.float64


def _cpu_half_conversion_available():
    try:
        torch.ones(1, dtype=torch.float16).float()
        return True
    except RuntimeError as exc:
        if "cpuinfo" in str(exc).lower():
            return False
        raise


@pytest.mark.skipif(not _cpu_half_conversion_available(), reason="PyTorch CPU cpuinfo initialization fails")
def test_reference_promotes_q_fp16_with_fp32_cache_and_preserves_grad_dtype():
    q = torch.randn(1, 3, 2, 8, dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 3, 2, 8, dtype=torch.float16, requires_grad=True)
    cos, sin = _cache(3, 8)
    out_q, out_k = reference_rope(q, k, cos, sin)
    assert out_q.dtype == out_k.dtype == torch.float16
    (out_q.sum() + out_k.sum()).backward()
    assert q.grad.dtype == torch.float16
    assert k.grad.dtype == torch.float16


@pytest.mark.skipif(not _cpu_half_conversion_available(), reason="PyTorch CPU cpuinfo initialization fails")
def test_reference_fp32_cache_promotes_half_inputs():
    q = torch.randn(1, 2, 1, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    cos, sin = _cache(2, 8)
    # 缓存保持 fp32，但输出跟随 q/k，避免 AMP 后续 attention 回退到 fp32。
    assert all(x.dtype == torch.float16 for x in reference_rope(q, k, cos, sin))


def test_reference_gradcheck_and_single_side_gradient():
    q = torch.randn(1, 2, 1, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(1, 2, 1, 4, dtype=torch.float64)
    cos, sin = _cache(2, 4)
    assert torch.autograd.gradcheck(lambda x: reference_rope(x, k, cos, sin)[0], (q,))


def test_rope_backend_is_opt_in_and_half_cache_is_rejected():
    from hcl_tr.models.backbone import HybridBackbone
    with pytest.raises(ValueError, match="RoPE backend"):
        HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4, rope_backend="typo")
    torch.manual_seed(0)
    model = HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4,
                           grad_checkpointing=True, rope_backend="triton")
    attn = next(b for b in model.blocks if b.kind == "attn")
    assert attn.mixer.rope.backend == "triton"
    default = HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4)
    assert all(b.mixer.rope.backend == "torch" for b in default.blocks if b.kind == "attn")
    if _cpu_half_conversion_available():
        model = model.half()
        assert attn.mixer.rope.inv_freq.dtype == torch.float16
        cache = attn.mixer.rope.inv_freq.new_zeros(3, 4)
        q = torch.ones(1, 3, 2, 8, dtype=torch.float16)
        with pytest.raises(TypeError, match=r'module.half\(\)'):
            fused_rope(q, q, cache, cache)


@pytest.mark.parametrize('fn', [reference_rope, fused_rope])
def test_contract_rejects_mixed_inputs_and_non_fp32_cache(fn):
    q = torch.ones(1, 3, 2, 8)
    cos, sin = _cache(3, 8)
    with pytest.raises(TypeError, match='q/k dtype 相同'):
        fn(q, q.double(), cos, sin)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        # 仅分配用于验证元数据，不依赖 CPU half 算术支持。
        cache = torch.empty(3, 4, dtype=dtype)
        with pytest.raises(TypeError, match='cos/sin 为 float32'):
            fn(q, q, cache, sin)
        with pytest.raises(TypeError, match='cos/sin 为 float32'):
            fn(q, q, cos, cache)


@pytest.mark.skipif(not _cpu_half_conversion_available(), reason='PyTorch CPU cpuinfo 初始化失败')
def test_default_torch_keeps_legacy_per_input_promotion():
    from hcl_tr.models.backbone import RotaryEmbedding
    module = RotaryEmbedding(8).half()
    q = torch.randn(1, 3, 2, 8, dtype=torch.float16)
    k = torch.randn(1, 3, 2, 8, dtype=torch.float32)
    got_q, got_k = module(q, k)
    # 旧公式全 half 输出 half，混合 q/k 各自提升；融合契约不覆盖该组合。
    assert got_q.dtype == torch.float16
    assert got_k.dtype == torch.float32


@pytest.mark.parametrize('fn', [reference_rope, fused_rope])
def test_contract_rejects_expanded_and_empty_inputs(fn):
    cos, sin = _cache(3, 4)
    expanded = torch.randn(1, 3, 2, 8).expand(2, -1, -1, -1)
    with pytest.raises(ValueError, match='zero-stride'):
        fn(expanded, expanded, cos, sin)
    with pytest.raises(ValueError, match='zero-stride'):
        fn(torch.randn(1, 3, 2, 8), torch.randn(1, 3, 2, 8), cos[:1].expand(3, -1), sin)
    with pytest.raises(ValueError, match='positive even'):
        fn(torch.empty(1, 3, 2, 0), torch.empty(1, 3, 2, 0),
           torch.empty(3, 0), torch.empty(3, 0))


def test_fused_rope_cpu_raises_instead_of_falling_back():
    q = torch.randn(1, 2, 1, 8)
    cos, sin = _cache(2, 8)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fused_rope(q, q, cos, sin)


def _has_cuda_triton():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize('strided_cache', ['cos', 'sin'])
@pytest.mark.parametrize('inverse', [False, True])
def test_launch_binds_independent_cache_strides(monkeypatch, strided_cache, inverse):
    import inspect
    from types import SimpleNamespace
    import hcl_tr.models.rope_fused as rope

    # 捕获真实内核签名，再将 launch 的位置参数绑定到该签名。
    fake_triton = SimpleNamespace(jit=lambda fn: fn, next_power_of_2=lambda n: 32)
    monkeypatch.setattr(rope, '_load_triton', lambda: (fake_triton, SimpleNamespace(constexpr=int)))
    _, kernel = rope._make_kernel()
    signature = inspect.signature(kernel)
    bindings = []

    class Kernel:
        def __getitem__(self, grid):
            assert grid == (2 * 97 * 3,)

            def launch(*args, **kwargs):
                assert kwargs.pop('num_warps') == 4
                assert kwargs.pop('enable_fp_fusion') is False
                bindings.append(signature.bind(*args, **kwargs).arguments)
            return launch

    monkeypatch.setattr(rope, '_make_kernel', lambda: (fake_triton, Kernel()))
    q = torch.empty(2, 97, 3, 40)
    cos, sin = torch.empty(97, 20), torch.empty(97, 20)
    if strided_cache == 'cos':
        cos = torch.empty(97, 40)[:, ::2]
    else:
        sin = torch.empty(97, 40)[:, ::2]
    rope._launch(q, q, cos, sin, torch.empty_like(q), torch.empty_like(q), inverse=inverse)
    bound = bindings[0]
    assert (bound['c_st'], bound['c_sd']) == cos.stride()
    assert (bound['s_st'], bound['s_sd']) == sin.stride()
    assert (bound['n_time'], bound['n_head'], bound['n_dim']) == (97, 3, 40)
    assert bound['BLOCK'] == 32 and bound['INVERSE'] is inverse


@pytest.mark.skipif(not _has_cuda_triton(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 128])
def test_fused_rope_matches_reference_forward_and_backward(dtype, dim):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('当前 GPU 不支持 bf16；不将其视为 T4 保证')
    torch.manual_seed(2)
    device = "cuda"
    # q/k 使用不同的非连续布局。
    qbase = torch.randn(2, 5, 3, dim * 2, device=device, dtype=dtype, requires_grad=True)
    kbase = torch.randn(2, 5, 3, dim * 2, device=device, dtype=dtype, requires_grad=True)
    q, k = qbase[..., ::2], kbase[..., 1::2]
    cos, sin = _cache(5, dim, device=device)
    actual = fused_rope(q, k, cos, sin)
    expected = reference_rope(q, k, cos, sin)
    tol = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 2e-2}[dtype]
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=tol, rtol=tol)
    gq = torch.randn_like(actual[0])
    gk = torch.randn_like(actual[1])
    actual_grads = torch.autograd.grad(actual, (qbase, kbase), (gq, gk), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (qbase, kbase), (gq, gk))
    for got, want in zip(actual_grads, expected_grads):
        torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@pytest.mark.skipif(not _has_cuda_triton(), reason="requires CUDA and Triton")
@pytest.mark.parametrize('strided_cache', ['cos', 'sin'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_fused_rope_independent_cache_layouts_forward_backward(strided_cache, dtype):
    # D/2=20、BLOCK=32，覆盖 mask；B/T/H 同时覆盖多维寻址。
    q = torch.randn(2, 97, 3, 40, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    cos, sin = _cache(97, 40, device='cuda')
    storage = torch.empty(97, 40, device='cuda')
    cache = cos if strided_cache == 'cos' else sin
    storage[:, ::2] = cache
    if strided_cache == 'cos':
        cos = storage[:, ::2]
    else:
        sin = storage[:, ::2]
    got = fused_rope(q, k, cos, sin)
    want = reference_rope(q, k, cos, sin)
    weights = tuple(torch.randn_like(t) for t in got)
    got_grads = torch.autograd.grad(got, (q, k), weights)
    want_grads = torch.autograd.grad(want, (q, k), weights)
    tol = 2e-5 if dtype == torch.float32 else 3e-3
    for actual, expected in zip(got + got_grads, want + want_grads):
        assert bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.skipif(not _has_cuda_triton(), reason="requires CUDA and Triton")
def test_fused_rope_partial_gradient_and_autocast_dtype():
    q = torch.randn(1, 3, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q)
    cos, sin = _cache(3, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out_q, out_k = fused_rope(q, k, cos, sin)
    assert out_q.dtype == torch.float16
    out_q.sum().backward()
    assert q.grad is not None and k.grad is None
