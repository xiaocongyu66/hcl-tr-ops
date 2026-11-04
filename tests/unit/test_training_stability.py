"""端到端数值稳定性：真实骨干跑多步，loss 必须有限且下降。

回归保护：曾因关闭 GDN-2 的 l2norm_qk（状态爆炸 1e9）导致第 2 步 loss=NaN。
"""

from __future__ import annotations

import torch

from hcl_tr.models.backbone import HybridBackbone
from hcl_tr.train.loop import chunked_cross_entropy


def _run_steps(model, steps=40, seq=128, B=4, vocab=256, lr=2e-4, seed=0, fixed=False):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    losses = []
    fixed_ids = torch.randint(0, vocab, (B, seq)) if fixed else None
    for _ in range(steps):
        ids = fixed_ids.clone() if fixed else torch.randint(0, vocab, (B, seq))
        logits = model(ids[:, :-1])
        loss = chunked_cross_entropy(logits, ids[:, 1:], chunk_tokens=64)
        assert torch.isfinite(loss), f"loss 非有限: {loss.item()}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(gn), f"grad_norm 非有限: {gn.item()}"
        opt.step()
        losses.append(loss.item())
    return losses


def test_hybrid_loss_stays_finite_and_decreases():
    """固定小批上应能过拟合：既验证不 NaN，也验证梯度真在学。

    （随机 token 无可学模式，loss 不会降——第一版测试就栽在这。）
    """
    torch.manual_seed(0)
    m = HybridBackbone(vocab_size=256, d_model=64, n_layer=4, n_head=4)
    losses = _run_steps(m, steps=60, fixed=True)
    assert all(map(lambda x: x == x, losses)), "出现 NaN"
    assert losses[-1] < losses[0] * 0.9, f"loss 未下降: 初 {losses[0]:.3f} 末 {losses[-1]:.3f}"


def test_hybrid_with_qed_loss_finite():
    m = HybridBackbone(vocab_size=256, d_model=64, n_layer=4, n_head=4, use_qed=True)
    losses = _run_steps(m, steps=15)
    assert all(map(lambda x: x == x, losses))


def test_hybrid_with_sla_loss_finite():
    m = HybridBackbone(vocab_size=256, d_model=64, n_layer=4, n_head=4, use_sla=True, use_qed=True)
    losses = _run_steps(m, steps=15)
    assert all(map(lambda x: x == x, losses))


def test_gdn2_state_stays_bounded_with_l2norm():
    """l2norm 必须始终开启：关闭后状态幅度差 9 个数量级（1e9 vs 0.6）。"""
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward
    torch.manual_seed(0)
    T, H, dk = 64, 2, 8
    q = torch.randn(1, T, H, dk); k = torch.randn(1, T, H, dk); v = torch.randn(1, T, H, dk)
    g = -torch.rand(1, T, H, dk) * 0.1; b = torch.rand(1, T, H, dk); w = torch.rand(1, T, H, dk)
    o = gdn2_chunkwise_forward(q, k, v, g, b, w, chunk=16, l2norm_qk=True)
    assert o.abs().max().item() < 100, f"l2norm 开启时输出仍有界失败: {o.abs().max().item():.3e}"


def test_qk_norm_flag_raises_instead_of_silently_disabling_stability():
    import pytest
    with pytest.raises(NotImplementedError):
        HybridBackbone(vocab_size=256, d_model=64, n_layer=4, n_head=4,
                       gdn2_internal_qk_norm=True)
