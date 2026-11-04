"""When2Think 三机制 fixture（严格按论文 arXiv:2609.19671v2）。

公式依据:
  Eq.(1) PDF p.3  IS 重要性比
  Eq.(2) PDF p.3  PPO clipped surrogate（IS 越界的梯度截断语义）
  Eq.(5) PDF p.5  IDAC: λ = exp(-T·α/τ) if Think else 1
  Eq.(6) PDF p.5  r = V·(1+λ·δ) - α_i    ← 注意 V=0 时 r=-α_i，不是 0
  Eq.(7)(8) PDF p.5  BWS: 按 trajectory index k 跨 M 个 instance 标准化
"""

from __future__ import annotations

import math

import pytest

from hcl_tr.models.idac import (
    IDACReward,
    ISMonitor,
    RewardConfig,
    bws_standardize,
    ppo_clipped_objective,
    weighted_objective,
)

R = IDACReward()


# ---------- Eq.(5) IDAC 缩放因子 ----------

def test_lambda_formula_eq5():
    """λ = exp(-T·α/τ)，与手算一致。"""
    T, alpha, tau = 200, 0.6, 100.0
    lam = R.lambda_scale(T, alpha, tau, thought=True)
    expected = math.exp(-T * alpha / tau)
    assert abs(lam - expected) < 1e-12


def test_lambda_nothink_is_one():
    """NoThink 的 λ = 1，不受 token 数影响。"""
    for T in [0, 100, 10000]:
        assert R.lambda_scale(T, 0.9, 50.0, thought=False) == 1.0


def test_lambda_monotonic_decreasing_in_tokens():
    """Think 轨迹 λ 随 token 数单调不增。"""
    prev = 2.0
    for T in [0, 10, 50, 100, 500, 2000]:
        lam = R.lambda_scale(T, 0.5, 100.0, thought=True)
        assert lam <= prev + 1e-12, f"λ 非单调不增 @ T={T}"
        prev = lam


def test_lambda_higher_alpha_decays_faster():
    """α_i（参考成功率）越高衰减越快 —— 计划书 §4.7 边界条件。"""
    T, tau = 200, 100.0
    lam_low = R.lambda_scale(T, alpha_i=0.1, tau_i=tau, thought=True)
    lam_high = R.lambda_scale(T, alpha_i=0.9, tau_i=tau, thought=True)
    assert lam_high < lam_low


def test_lambda_rejects_nonpositive_tau():
    with pytest.raises(ValueError):
        R.lambda_scale(10, 0.5, 0.0, thought=True)


# ---------- Eq.(6) 最终奖励 ----------

def test_wrong_answer_reward_is_minus_alpha():
    """Eq.(6): V=0 → r = -α_i（**不是 0**，与计划书简化不同）。"""
    r = R.reward(correct=False, thought=False, think_tokens=0, alpha_i=0.4, tau_i=100.0)
    assert abs(r - (-0.4)) < 1e-12
    r2 = R.reward(correct=False, thought=True, think_tokens=500, alpha_i=0.4, tau_i=100.0)
    assert abs(r2 - (-0.4)) < 1e-12


def test_correct_nothink_reward_full():
    """V=1, NoThink → λ=1 → r = 1 + δ - α_i。"""
    cfg = RewardConfig(delta=1.0)
    r = IDACReward(cfg)
    got = r.reward(correct=True, thought=False, think_tokens=0, alpha_i=0.3, tau_i=100.0)
    assert abs(got - (1.0 + 1.0 - 0.3)) < 1e-12


def test_correct_think_reward_monotonic():
    """正确 THINK 奖励随 token 单调不增。"""
    prev = 1e9
    for T in [0, 10, 100, 500, 2000]:
        r = R.reward(correct=True, thought=True, think_tokens=T, alpha_i=0.5, tau_i=100.0)
        assert r <= prev + 1e-12
        prev = r


def test_correct_think_bounded_by_nothink():
    """Think 奖励不超过 NoThink（λ<=1，δ>0）。"""
    cfg = RewardConfig(delta=2.0)
    r = IDACReward(cfg)
    nothink = r.reward(True, False, 0, 0.5, 100.0)
    for T in [1, 100, 1000]:
        think = r.reward(True, True, T, 0.5, 100.0)
        assert think <= nothink + 1e-12


# ---------- Eq.(7)(8) BWS ----------

def test_bws_standardizes_across_instances_per_k():
    """Eq.(7)(8): μ_k、σ_k 按 trajectory index k 跨 M 个 instance 求。"""
    rewards = {0: [1.0, 2.0, 3.0], 1: [0.5, 0.5, 0.5]}
    adv = bws_standardize(rewards)
    # k=0: mean=2, std=sqrt(2/3)
    mu0 = 2.0
    sig0 = math.sqrt(((1-2)**2 + (2-2)**2 + (3-2)**2) / 3)
    expect0 = [(x - mu0) / sig0 for x in rewards[0]]
    for a, e in zip(adv[0], expect0):
        assert abs(a - e) < 1e-5
    # k=1: 全相同 → adv 全 0
    assert all(abs(a) < 1e-5 for a in adv[1])


def test_bws_not_pooled_across_k():
    """关键：不能把不同 k 混在一起求均值（论文是按 k 分组）。"""
    rewards = {0: [10.0, 10.0, 10.0], 1: [0.0, 0.0, 0.0]}
    adv = bws_standardize(rewards)
    # 按 k 分组 → 每组内部无方差 → 全 0；若错误地跨 k 拉平，会得到非零
    assert all(abs(a) < 1e-5 for a in adv[0])
    assert all(abs(a) < 1e-5 for a in adv[1])


def test_bws_advantage_bounded():
    cfg = RewardConfig(advantage_bound=10.0)
    rewards = {0: [0.0, 0.0, 0.0, 0.0, 100.0]}
    adv = bws_standardize(rewards, cfg)
    assert all(abs(a) <= 10.0 + 1e-9 for a in adv[0])


def test_bws_zero_mean_per_k():
    adv = bws_standardize({0: [1.0, 2.0, 3.0, 4.0]})
    assert abs(sum(adv[0])) < 1e-5


# ---------- Eq.(2) PPO clipped surrogate / IS 越界 ----------

def test_ppo_in_range_not_clipped():
    obj, clipped = ppo_clipped_objective(rho=1.1, adv=2.0)
    assert not clipped
    assert abs(obj - 2.2) < 1e-12


def test_ppo_out_of_range_is_clipped():
    """ρ 越界 → clipped=True，目标值不再随 ρ 增长（梯度截断）。"""
    cfg = RewardConfig(ppo_clip_eps=0.2)
    obj_low, c1 = ppo_clipped_objective(rho=0.5, adv=1.0, cfg=cfg)
    obj_high, c2 = ppo_clipped_objective(rho=3.0, adv=1.0, cfg=cfg)
    assert c1 and c2
    # ρ=3 时被裁到 1.2，objective = min(3, 1.2)*1 = 1.2
    assert abs(obj_high - 1.2) < 1e-12
    # ρ=0.5 时被裁到 0.8，objective = min(0.5, 0.8)*1 = 0.5
    assert abs(obj_low - 0.5) < 1e-12


def test_ppo_negative_advantage_clipping():
    """A<0 时 min 语义：越界后取被裁剪支，避免过度惩罚。"""
    cfg = RewardConfig(ppo_clip_eps=0.2)
    obj, clipped = ppo_clipped_objective(rho=3.0, adv=-1.0, cfg=cfg)
    # min(3*-1, 1.2*-1) = min(-3, -1.2) = -3  → 不裁剪这一侧（PPO 的悲观下界）
    assert clipped
    assert abs(obj - (-3.0)) < 1e-12


def test_weighted_objective_is_negated():
    o, _ = ppo_clipped_objective(1.0, 2.0)
    w, _ = weighted_objective(1.0, 2.0)
    assert abs(w + o) < 1e-12


# ---------- IS 监控 / 停训 ----------

def test_is_monitor_clip_fraction():
    m = ISMonitor(RewardConfig(ppo_clip_eps=0.2), max_clip_frac=0.5)
    for rho in [1.0, 1.1, 0.9]:
        m.observe(rho)
    assert m.clip_fraction() == 0.0
    assert not m.should_stop()

    m.observe(2.0)   # 越界
    assert abs(m.clip_fraction() - 0.25) < 1e-12
    assert not m.should_stop()


def test_is_monitor_stops_when_clip_fraction_high():
    m = ISMonitor(RewardConfig(ppo_clip_eps=0.1), max_clip_frac=0.5)
    for _ in range(3):
        m.observe(5.0)
    assert m.should_stop()


def test_is_out_of_bounds_never_silently_passes():
    """计划书 §4.7 红线：IS 越界不得悄悄 clipping 而不记录。"""
    m = ISMonitor()
    clipped = m.observe(10.0)
    assert clipped is True
    assert m.n_clipped == 1
