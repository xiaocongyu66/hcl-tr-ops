"""SPAR fixture：门控 KL、false-gate / missed-gate 比例。

SOURCE_LOCK: SPAR LOCKED-PAPER, arXiv:2609.27925
  第 2 页: L = L_CLM + λ_inv L_inv；short-context sufficiency gate；masked KL
计划书 §4.8: 训练中门控阈值固定；false-gate / missed-gate 任一 >5% → 冻结 SPAR
"""

from __future__ import annotations

import torch

from hcl_tr.models.spar import SPARGate, gate_audit, spar_loss

B, T, V = 2, 12, 32
SUFFIX_START = 8


def _logits(seed, sharpen=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, V, generator=g) * sharpen


def _targets():
    g = torch.Generator().manual_seed(99)
    return torch.randint(0, V, (B, T), generator=g)


def test_spar_loss_equals_clm_when_no_mask():
    """mask 全 False 时 L_inv=0，总损失 = L_CLM。"""
    lc, lk = _logits(1), _logits(2)
    t = _targets()
    mask = torch.zeros(B, T, dtype=torch.bool)
    total, l_clm, l_inv = spar_loss(lc, lk, t, mask, lbda_inv=1.0)
    assert abs(l_inv.item()) < 1e-7
    assert abs(total.item() - l_clm.item()) < 1e-6


def test_spar_loss_adds_lambda_times_inv():
    """total = L_CLM + λ_inv * L_inv。"""
    lc, lk = _logits(3, sharpen=2.0), _logits(4, sharpen=0.5)
    t = _targets()
    mask = torch.ones(B, T, dtype=torch.bool)
    lam = 0.5
    total, l_clm, l_inv = spar_loss(lc, lk, t, mask, lbda_inv=lam)
    assert abs(total.item() - (l_clm.item() + lam * l_inv.item())) < 1e-6


def test_kl_nonnegative_and_zero_when_identical():
    """KL(P||P) = 0；KL 恒非负。"""
    lc = _logits(5, sharpen=2.0)
    t = _targets()
    mask = torch.ones(B, T, dtype=torch.bool)
    _, _, l_inv_same = spar_loss(lc, lc, t, mask, lbda_inv=1.0)
    assert l_inv_same.item() < 1e-6

    lk = _logits(6, sharpen=2.0)
    _, _, l_inv = spar_loss(lc, lk, t, mask, lbda_inv=1.0)
    assert l_inv.item() >= 0


def test_gate_selects_only_suffix():
    """门控掩码只在 local suffix 区（pos >= suffix_start）内。"""
    gate = SPARGate(mode="suffix_likelihood", L_suffix=4, nll_threshold=1e9, gain_threshold=1e9)
    lc = _logits(7, sharpen=5.0)   # 高置信 → short NLL 低
    lk = _logits(8, sharpen=5.0)
    t = _targets()
    r = gate.suffix_likelihood_gate(lc, lk, t, SUFFIX_START)
    mask = r["mask"]
    assert mask[:, :SUFFIX_START].sum() == 0, "prefix 区不应被选中"
    # 阈值很宽 → suffix 区应全选
    assert mask[:, SUFFIX_START:].all()


def test_gate_rejects_high_nll():
    """short NLL 高（低置信）时门控拒绝。"""
    gate = SPARGate(mode="suffix_likelihood", L_suffix=4, nll_threshold=0.0, gain_threshold=1e9)
    lc = _logits(9, sharpen=1.0)
    lk = _logits(10, sharpen=1.0)
    t = _targets()
    r = gate.suffix_likelihood_gate(lc, lk, t, SUFFIX_START)
    assert r["mask"].sum() == 0


def test_gate_audit_ratios_and_freeze_flag():
    """self-consistent mask 的 false/missed 为 0；故意扭曲后触发 freeze。"""
    gate = SPARGate(mode="suffix_likelihood", L_suffix=4, nll_threshold=1e9, gain_threshold=1e9)
    lc = _logits(11, sharpen=5.0)
    lk = _logits(12, sharpen=5.0)
    t = _targets()
    r = gate.suffix_likelihood_gate(lc, lk, t, SUFFIX_START)

    audit = gate_audit(r["mask"], r["nll_short"], r["gain"], 1e9, 1e9, suffix_start=SUFFIX_START)
    assert audit["false_gate_ratio"] < 1e-9
    assert audit["missed_gate_ratio"] < 1e-9
    assert not audit["freeze_spar"]

    # 故意把 mask 反转 → 高 false/missed → 触发冻结
    bad_mask = ~r["mask"]
    audit_bad = gate_audit(bad_mask, r["nll_short"], r["gain"], 1e9, 1e9, suffix_start=SUFFIX_START)
    assert audit_bad["freeze_spar"]


def test_gate_selects_only_suffix_semantics():
    """门控 mask 与 suffix 区语义一致：suffix 区内全选，prefix 区不动。"""
    gate = SPARGate(mode="suffix_likelihood", L_suffix=4, nll_threshold=1e9, gain_threshold=1e9)
    lc, lk = _logits(21, sharpen=5.0), _logits(22, sharpen=5.0)
    t = _targets()
    r = gate.suffix_likelihood_gate(lc, lk, t, SUFFIX_START)
    assert r["mask"][:, :SUFFIX_START].sum() == 0
    assert r["mask"][:, SUFFIX_START:].all()


def test_probe_mode_returns_suffix_mask():
    gate = SPARGate(mode="probe", d_model=16, L_suffix=4)
    hidden = torch.randn(B, T, 16, generator=torch.Generator().manual_seed(13))
    m = gate.probe_gate(hidden, SUFFIX_START)
    assert m.shape == (B, T)
    assert m[:, :SUFFIX_START].sum() == 0


def test_corrupt_prefix_keeps_suffix_intact():
    """corrupt 视图必须逐 token 保留 local suffix。"""
    from hcl_tr.models.spar import corrupt_prefix_view
    g = torch.Generator().manual_seed(0)
    ids = torch.arange(600, dtype=torch.long).view(1, -1)
    out = corrupt_prefix_view(ids, suffix_len=128, chunk_size=32, generator=g)
    assert torch.equal(out[:, -128:], ids[:, -128:]), "suffix 被改动"
    assert not torch.equal(out[:, :-128], ids[:, :-128]), "prefix 未被打乱"


def test_corrupt_prefix_preserves_multiset():
    """块重排不得改变 token 多重集（只换位置）。"""
    from hcl_tr.models.spar import corrupt_prefix_view
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 100, (2, 512), generator=g)
    out = corrupt_prefix_view(ids, suffix_len=64, chunk_size=32, generator=g)
    assert torch.equal(out.sort(dim=1).values, ids.sort(dim=1).values)


def test_corrupt_prefix_deterministic_with_generator():
    from hcl_tr.models.spar import corrupt_prefix_view
    ids = torch.randint(0, 100, (1, 400))
    a = corrupt_prefix_view(ids, 64, 32, torch.Generator().manual_seed(7))
    b = corrupt_prefix_view(ids, 64, 32, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_spar_loss_prefers_matching_distributions():
    """clean 与 corrupt 分布相同时 L_inv 应为 0。"""
    import torch
    from hcl_tr.models.spar import spar_loss
    torch.manual_seed(0)
    lg = torch.randn(2, 8, 50)
    tgt = torch.randint(0, 50, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    _, l_clm, l_inv = spar_loss(lg, lg.clone(), tgt, mask)
    assert l_inv.item() < 1e-6
    assert l_clm.item() > 0
