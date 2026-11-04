"""SLA fixture（按论文 arXiv:2602.01744 重写）。

公式依据:
  §4.1 / Eq.(5)-(7): G_Q = softmax_h((QW_GQ)_h)，线性注意力 head 门控更新
  Eq.(8)  Theorem 4.2: lim_{λ→∞} G_Q(λs) = one-hot(argmax_h s_h)
  Eq.(9)  Theorem 4.3: C(λ) = Σ_h G_Qh G_Kh → δ(argmax_h s_Qh, argmax_h s_Kh)

已删除（论文无此内容，不再测试）: 温度退火、熵正则惩罚、θ_range。
头熵仅作监控，不断言"必须 > 0.05"。
"""

from __future__ import annotations

import math

import torch

from hcl_tr.models.sla import (
    SoftmaxLinearAttention,
    head_entropy,
    information_flow_coefficient,
    kronecker_routing,
)

D, H, B, T = 32, 4, 2, 6


def _layer(lam=1.0, seed=0):
    torch.manual_seed(seed)
    return SoftmaxLinearAttention(D, H, lam=lam)


def _x(seed=1):
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


def test_gates_are_softmax_over_heads():
    """G_Q / G_K 沿 head 归一化，Σ_h = 1（论文 §4.1）。"""
    m = _layer()
    gq, gk = m.head_gates(_x(), _x())
    assert gq.shape == (B, T, H)
    assert (gq.sum(-1) - 1).abs().max() < 1e-6
    assert (gk.sum(-1) - 1).abs().max() < 1e-6
    assert (gq >= 0).all() and (gk >= 0).all()


def test_gate_matches_manual_softmax_with_lambda():
    """G_Q = softmax_h(λ (QW_GQ)_h)，λ 是缩放强度。"""
    m = _layer(lam=2.0)
    x = _x()
    gq, _ = m.head_gates(x, x)
    manual = torch.softmax(2.0 * m.w_gq(x), dim=-1)
    assert (gq - manual).abs().max() < 1e-6


def test_lambda_infinity_approaches_onehot_eq8():
    """Eq.(8) / Theorem 4.2: λ→∞ 时 G_Q → one-hot(argmax_h s_h)。"""
    x = _x(seed=3)
    base = _layer(lam=1.0)
    s = base.w_gq(x)                      # 未缩放的 head 打分 [B,T,H]
    argmax = s.argmax(-1)

    for lam in [1.0, 10.0, 100.0, 1000.0]:
        m = SoftmaxLinearAttention(D, H, lam=lam)
        m.load_state_dict(base.state_dict())
        gq, _ = m.head_gates(x, x)
        # 最大值与 one-hot 对应位置一致
        assert (gq.argmax(-1) == argmax).all(), f"lam={lam} argmax 偏移"

    # 用「非 argmax 头的残余概率」衡量收敛：max 会在 float32 下提前饱和到 1.0，
    # 不足以看到单调性；min-non-argmax 能一路衰减到 0。
    residual = []
    for lam in [1.0, 10.0, 100.0, 1000.0]:
        m = SoftmaxLinearAttention(D, H, lam=lam)
        m.load_state_dict(base.state_dict())
        gq, _ = m.head_gates(x, x)
        # 非 argmax 位置的概率之和
        mask = torch.nn.functional.one_hot(argmax, H).bool()
        residual.append(gq.masked_fill(mask, 0.0).sum().item())
    assert all(residual[i] > residual[i + 1] for i in range(len(residual) - 1)), residual
    assert residual[0] > 0.1
    assert residual[-1] < 1e-6, f"λ=1000 应近乎 one-hot，残余 {residual[-1]}"


def test_entropy_decreases_with_lambda():
    """Theorem 4.2: 缩放强度 λ 增大 → 门控分布熵下降。"""
    x = _x(seed=4)
    base = _layer()
    ents = []
    for lam in [0.1, 1.0, 10.0, 100.0]:
        m = SoftmaxLinearAttention(D, H, lam=lam)
        m.load_state_dict(base.state_dict())
        gq, _ = m.head_gates(x, x)
        ents.append(head_entropy(gq).mean().item())
    assert all(ents[i] > ents[i + 1] for i in range(len(ents) - 1)), ents


def test_information_flow_kronecker_eq9():
    """Eq.(9): head 一致时 C(λ)→1，不一致时 →0（λ 大）。"""
    # 构造确定性打分：Q 与 K 在 head 2 上同为最大
    m = SoftmaxLinearAttention(D, H, lam=100.0)
    torch.nn.init.zeros_(m.w_gq.weight); torch.nn.init.zeros_(m.w_gq.bias)
    torch.nn.init.zeros_(m.w_gk.weight); torch.nn.init.zeros_(m.w_gk.bias)
    with torch.no_grad():
        m.w_gq.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))
        m.w_gk.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))
    x = torch.randn(1, 1, D)
    gq, gk = m.head_gates(x, x)
    C = information_flow_coefficient(gq, gk)
    assert C.item() > 0.99, f"共识 head 时 C 应趋近 1，实际 {C.item()}"
    assert kronecker_routing(gq, gk).item() is True or kronecker_routing(gq, gk).all()

    # 不一致：K 的最大在 head 0
    with torch.no_grad():
        m.w_gk.bias.copy_(torch.tensor([5.0, 0.0, 0.0, 0.0]))
    gq, gk = m.head_gates(x, x)
    C = information_flow_coefficient(gq, gk)
    assert C.item() < 1e-6, f"不共识时 C 应趋近 0，实际 {C.item()}"
    assert not kronecker_routing(gq, gk).all()


def test_forward_output_shape_and_gradient():
    m = _layer()
    x = _x(seed=5)
    r = m(x)
    assert r["out"].shape == (B, T, D)
    r["out"].pow(2).mean().backward()
    assert m.w_gq.weight.grad is not None
    assert m.w_gk.weight.grad is not None
    assert m.q_proj.weight.grad is not None


def test_head_entropy_monitor_values():
    """头熵监控：均匀 = log H；one-hot = 0。仅监控，非损失。"""
    uniform = torch.full((1, 1, H), 1.0 / H)
    assert abs(head_entropy(uniform).item() - math.log(H)) < 1e-5
    onehot = torch.zeros(1, 1, H); onehot[..., 0] = 1.0
    assert head_entropy(onehot).item() < 1e-6


def test_linear_complexity_prefix_state():
    """状态是前缀累加（Eq.5），第 t 步只依赖前 t 步。"""
    m = _layer(lam=1.0)
    x = _x(seed=6)
    full = m(x)["out"]
    # 截断到前 3 个 token 的前向，应与全长前向的前 3 个输出一致（因果性）
    part = m(x[:, :3])["out"]
    assert (full[:, :3] - part).abs().max() < 1e-5
