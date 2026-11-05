"""HybridBackbone fixture：3:1 层比例、QK-norm、消融开关、前向形状。"""

from __future__ import annotations

import torch
import pytest

from hcl_tr.models.backbone import FullAttention, GDN2Layer, HybridBackbone

V, D, L, H = 128, 32, 8, 4


def _model(**kw):
    torch.manual_seed(0)
    return HybridBackbone(vocab_size=V, d_model=D, n_layer=L, n_head=H, **kw)


def test_three_to_one_layer_ratio():
    m = _model(attn_ratio=4)
    kinds = [b.kind for b in m.blocks]
    # 每 4 层最后 1 层是注意力
    assert kinds == ["gdn2", "gdn2", "gdn2", "attn"] * 2


def test_forward_shape_and_loss_backward():
    m = _model()
    ids = torch.randint(0, V, (2, 16), generator=torch.Generator().manual_seed(1))
    logits = m(ids)
    assert logits.shape == (2, 16, V)
    loss = logits.float().pow(2).mean()
    loss.backward()
    assert m.embed.weight.grad is not None


def test_qk_norm_present_on_full_attention():
    m = _model(qk_norm_full_attn=True)
    attn_layers = [b.mixer for b in m.blocks if b.kind == "attn"]
    assert all(isinstance(a, FullAttention) and a.qk_norm for a in attn_layers)


def test_qk_norm_can_be_disabled():
    m = _model(qk_norm_full_attn=False)
    attn_layers = [b.mixer for b in m.blocks if b.kind == "attn"]
    assert all(not a.qk_norm for a in attn_layers)


def test_sla_stacks_on_gdn2_and_keeps_qed():
    """计划书 §9「在 +GDN/QED 上加 SLA」：SLA 叠在 GDN-2 上，可与 QED 共存。"""
    m = _model(use_sla=True, use_qed=True)
    gdn = [b for b in m.blocks if b.kind == "gdn2"]
    assert gdn, "SLA 组不应移除 GDN-2 层"
    assert all(getattr(b.mixer, "use_sla", False) for b in gdn)
    assert all(getattr(b.mixer, "use_qed", False) for b in gdn)
    # 门控参数存在
    assert hasattr(gdn[0].mixer, "sla_gq") and hasattr(gdn[0].mixer, "sla_gk")


def test_attribution_groups_param_aligned():
    """归因组参数量必须接近，否则差异被参数混淆。"""
    import json
    from pathlib import Path
    import yaml
    from hcl_tr.models.backbone import HybridBackbone
    root = Path(__file__).resolve().parents[2]
    groups = json.loads((root / "docs/attribution_groups.json").read_text(encoding="utf-8"))["groups"]
    counts = {}
    for g in groups:
        cfg = yaml.safe_load((root / g["config_path"]).read_text(encoding="utf-8"))["model"]
        m = HybridBackbone(**cfg)
        counts[g["id"]] = sum(p.numel() for p in m.parameters())
    # 所有组都在 ~100M 量级，且极差 < 5%
    lo, hi = min(counts.values()), max(counts.values())
    # 64k 双语词表 + d=640 口径：各组应在 ~95M 附近，极差 <5%
    assert hi / lo < 1.05, f"参数量极差过大: {counts}"
    assert 90e6 <= lo and hi <= 100e6, f"参数量超出 100M 口径: {counts}"


def test_sla_disabled_by_default_no_sla_block():
    m = _model(use_sla=False)
    assert not any(b.kind == "sla" for b in m.blocks)


def test_sla_enabled_keeps_hybrid_structure():
    """SLA 叠加不改变 3:1 混合结构（GDN-2 层保留，只是多了门控）。"""
    m = _model(use_sla=True)
    kinds = [b.kind for b in m.blocks]
    assert kinds == ["gdn2", "gdn2", "gdn2", "attn"] * 2


def test_sla_gate_is_head_softmax_and_trainable():
    """G_Q/G_K 沿 head 归一化，且门控参数可训练。"""
    m = _model(use_sla=True)
    x = torch.randn(2, 8, D)
    gq = torch.softmax(m.blocks[0].mixer.sla_gq(x), dim=-1)
    assert (gq.sum(-1) - 1).abs().max() < 1e-6
    ids = torch.randint(0, V, (2, 8), generator=torch.Generator().manual_seed(2))
    m(ids).pow(2).mean().backward()
    assert m.blocks[0].mixer.sla_gq.weight.grad is not None


def test_qed_switch_changes_path():
    m = _model(use_qed=False)
    gdn_layers = [b.mixer for b in m.blocks if b.kind == "gdn2"]
    assert all(not g.use_qed for g in gdn_layers)

    m2 = _model(use_qed=True)
    gdn_layers2 = [b.mixer for b in m2.blocks if b.kind == "gdn2"]
    assert all(g.use_qed for g in gdn_layers2)
    assert all(hasattr(g, "lam_h_raw") for g in gdn_layers2)


def test_qed_lambda_per_head_and_trainable():
    """λ_h 必须是每头独立向量、非负、且能拿到梯度（不得坍缩成标量）。"""
    import torch.nn.functional as F
    m = _model(use_qed=True)
    g = [b.mixer for b in m.blocks if b.kind == "gdn2"][0]

    # 每头独立参数
    assert g.lam_h_raw.shape == (H,)
    lam = F.softplus(g.lam_h_raw)
    assert lam.shape == (H,)
    assert (lam >= 0).all()

    # 梯度必须能流到 lam_h_raw —— 这是验证代理抓到的真实 bug
    ids = torch.randint(0, V, (1, 4), generator=torch.Generator().manual_seed(7))
    m(ids).float().pow(2).mean().backward()
    assert g.lam_h_raw.grad is not None, "λ_h 梯度为 None：每头强度未被优化"
    assert g.lam_h_raw.grad.abs().sum().item() > 0


def test_qed_per_head_lambda_affects_state_differently():
    """不同头用不同 λ_h 应产生不同结果（证明没有取平均坍缩）。"""
    from hcl_tr.models.qed import gdn2_step_with_qed
    torch.manual_seed(11)
    Hh, dk, dv = 4, 8, 8
    S = torch.randn(Hh, dk, dv)
    q = torch.randn(Hh, dk); k = torch.randn(Hh, dk); v = torch.randn(Hh, dv)
    gdeg = torch.zeros(Hh, dk); b = torch.ones(Hh, dk); w = torch.ones(Hh, dv)
    lam_vec = torch.tensor([0.0, 0.5, 1.0, 2.0])
    lam_scalar = float(lam_vec.mean())
    S_vec, _ = gdn2_step_with_qed(S, q, k, v, gdeg, b, w, lam_vec)
    S_sca, _ = gdn2_step_with_qed(S, q, k, v, gdeg, b, w, lam_scalar)
    assert (S_vec - S_sca).abs().max() > 1e-4, "每头 λ 应不同于统一标量"


def test_output_gate_bias_init_one():
    """输出门 bias 初始化为 1.0（计划书 §5）。

    投影已融合为单个 in_proj（q,k,v,b,w,f,gate 各占 D 段），门对应最后 D 段。
    """
    m = _model(output_gate=True)
    g = [b.mixer for b in m.blocks if b.kind == "gdn2"][0]
    D = g.n_head * g.head_dim
    assert torch.allclose(g.in_proj.bias[-D:], torch.ones(D))
    # 其余 6 段必须是 0（等价于原来的 bias=False）
    assert torch.allclose(g.in_proj.bias[:-D], torch.zeros(6 * D))


def test_gdn2_layer_divide_by_sqrt_t_toggle():
    x = torch.randn(1, 4, D, generator=torch.Generator().manual_seed(3))
    g1 = GDN2Layer(D, H, divide_by_sqrt_t=True)
    g2 = GDN2Layer(D, H, divide_by_sqrt_t=False)
    g2.load_state_dict(g1.state_dict())
    with torch.no_grad():
        o1 = g1(x)["out"]
        o2 = g2(x)["out"]
    assert (o1 - o2).abs().max() > 1e-6   # 归一化确实改变了输出


def test_param_count_matches_plan_spec():
    """计划书 §3 口径：params_total ~100M，params_embed 24.6M，params_non_embed ~75.4M。

    实测 102.47M / 24.58M / 77.89M，落在 ~100M 的合理带宽内。
    若 MLP 扩张改回 4×，non_embed 会到 106M —— 本测试会拦住。
    """
    m = HybridBackbone(vocab_size=32000, d_model=768, n_layer=12, n_head=12)
    emb = sum(p.numel() for p in m.embed.parameters())
    tot = sum(p.numel() for p in m.parameters())
    non_embed = tot - emb
    assert abs(emb / 1e6 - 24.6) < 0.2, f"embed={emb/1e6:.2f}M 应≈24.6M"
    assert 98e6 <= tot <= 108e6, f"total={tot/1e6:.2f}M 应在 100M 口径"
    assert 74e6 <= non_embed <= 82e6, f"non_embed={non_embed/1e6:.2f}M 应≈75.4M"


def test_init_loss_near_uniform_entropy():
    """初始化后 logits 应接近均匀分布：loss ≈ ln(vocab)，而不是爆炸。

    回归保护：nn.Embedding 默认 N(0,1) + RMSNorm 残差曾导致初始 loss≈628。
    """
    import math
    m = HybridBackbone(vocab_size=1000, d_model=64, n_layer=4, n_head=4)
    ids = torch.randint(0, 1000, (2, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        logits = m(ids)
    assert logits.abs().max() < 20, f"初始 logits 过大: {logits.abs().max().item():.1f}"
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 1000), torch.randint(0, 1000, (2 * 64,))
    )
    expected = math.log(1000)
    assert abs(loss.item() - expected) < 2.0, f"init loss={loss.item():.2f}, 预期≈{expected:.2f}"


def test_attention_is_causal():
    """自回归 LM 必须因果：改最后一个 token 不能影响前面位置的输出。

    回归保护：曾因 mask=None 走双向注意力，模型能看到未来 token，loss 会虚假偏低。
    """
    from hcl_tr.models.backbone import FullAttention
    torch.manual_seed(0)
    m = FullAttention(64, 4, qk_norm=True).eval()
    x = torch.randn(2, 16, 64)
    x2 = x.clone()
    x2[:, -1] += 5.0
    with torch.no_grad():
        o1 = m(x)
        o2 = m(x2)
    assert (o1[:, :-1] - o2[:, :-1]).abs().max() < 1e-6, "存在未来信息泄漏"
    assert (o1[:, -1] - o2[:, -1]).abs().max() > 1e-3


@pytest.mark.parametrize("backend", ["serial", "batched", "fused"])
def test_checkpoint_locks_chunk_per_forward_and_matches_gradients(backend):
    import copy
    from torch.utils.checkpoint import checkpoint
    from hcl_tr.models.backbone import GDN2Layer
    torch.manual_seed(12)
    layer = GDN2Layer(16, 2, chunk=32, chunk_backend=backend)
    reference = copy.deepcopy(layer)
    x1 = torch.randn(1, 40, 16, requires_grad=True)
    x2 = (torch.randn(1, 40, 16) * 20).requires_grad_()
    outputs = []
    for x in (x1, x2):
        outputs.append(checkpoint(layer, x, use_reentrant=False,
                                  context_fn=layer.checkpoint_contexts)["out"])
    assert layer._calib_record is None
    sum(o.square().sum() for o in outputs).backward()
    reference_outputs = [reference(x.detach())["out"] for x in (x1, x2)]
    sum(o.square().sum() for o in reference_outputs).backward()
    for actual, expected in zip(outputs, reference_outputs):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(layer.parameters(), reference.parameters()):
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-4)


def test_each_forward_recalibrates_before_operator(monkeypatch):
    import hcl_tr.models.backbone as backbone
    from hcl_tr.models.gdn2_ref import STATS
    layer = backbone.GDN2Layer(16, 2, chunk=32)
    observed = []
    original = backbone.gdn2_chunkwise_forward
    def wrapped(*args, **kwargs):
        observed.append(kwargs["chunk_eff_in"])
        result = original(*args, **kwargs)
        assert STATS["chunk_eff"] == kwargs["chunk_eff_in"]
        return result
    monkeypatch.setattr(backbone, "gdn2_chunkwise_forward", wrapped)
    with torch.no_grad():
        layer.dt_bias.fill_(-8)
        layer(torch.zeros(1, 40, 16))
        layer.dt_bias.fill_(3)
        layer(torch.zeros(1, 40, 16))
    assert observed[0] > observed[1]


def test_nonfinite_decay_fails_instead_of_reusing_cache():
    import pytest
    from hcl_tr.models.backbone import GDN2Layer
    layer = GDN2Layer(16, 2)
    with torch.no_grad():
        layer.dt_bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="NaN/Inf"):
        layer(torch.zeros(1, 4, 16))


def test_fused_widestep_matches_serial_backend():
    """fused 用更大 chunk（safe_exp=80）时，仍应与 serial 数值一致。"""
    from hcl_tr.models.backbone import GDN2Layer
    torch.manual_seed(31)
    serial = GDN2Layer(16, 2, chunk=64, chunk_backend="serial")
    fused = GDN2Layer(16, 2, chunk=64, chunk_backend="fused")
    fused.load_state_dict(serial.state_dict())
    x = torch.randn(1, 96, 16)
    with torch.no_grad():
        expected = serial(x)["out"]
        actual = fused(x)["out"]
    assert fused._chunk_eff >= serial._chunk_eff
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_fused_backend_rejects_unsupported_options():
    import pytest
    with pytest.raises(ValueError, match="只支持 fp32"):
        HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4,
                       gdn2_chunk_backend="fused", gdn2_matmul_dtype="fp16")
    with pytest.raises(NotImplementedError, match="QED/SLA"):
        __import__('hcl_tr.models.backbone', fromlist=['GDN2Layer']).GDN2Layer(
            16, 2, chunk_backend="fused", use_qed=True)


def test_batched_backbone_matches_serial_with_checkpoint():
    torch.manual_seed(24)
    serial = HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4,
                            grad_checkpointing=False, gdn2_chunk=16)
    batched = HybridBackbone(vocab_size=64, d_model=32, n_layer=4, n_head=4,
                             grad_checkpointing=True, gdn2_chunk=16,
                             gdn2_chunk_backend="batched")
    batched.load_state_dict(serial.state_dict())
    ids = torch.randint(0, 64, (2, 35))
    expected, actual = serial(ids), batched(ids)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, a), (_, b) in zip(serial.named_parameters(), batched.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(b.grad, a.grad, atol=1e-5, rtol=1e-4, msg=name)
    assert serial.chunk_stats() == batched.chunk_stats()
