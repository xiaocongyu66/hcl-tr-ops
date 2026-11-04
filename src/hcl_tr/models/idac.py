"""When2Think 的 IDAC / BWS / IS 三项机制。

SOURCE_LOCK: When2Think LOCKED
  论文 arXiv:2609.19671v2（PDF 归档 sha256 c2f4ff4a…，页码已核）；权重
  junshim/When2Think-1.5B @ 69547b99…

公式依据（严格按论文，**不按计划书简化**）:
  Eq.(1) PDF p.3  IS 重要性比:
      ρ_{i,k} = π_θ(o_{i,k}|x_i) / π_IS(o_{i,k}|x_i)
      π_IS 以等概率选 THINK/NOTHINK 初始 mode token，后续 token 由 π_θ 生成
  Eq.(2) PDF p.3  PPO clipped surrogate:
      L = -E[min(ρ A, clip(ρ, 1-ε, 1+ε) A)]
      "IS 越界停训"落在 clip 语义上：ρ 偏离 1 超 ε 时 min 取被裁剪支，梯度截断
  Eq.(5) PDF p.5  IDAC 缩放因子:
      λ̃_{i,k} = exp(-T(o_{i,k})·α_i / τ_i),  λ = λ̃ if Think else 1
      α_i = 参考策略在实例 i 的经验成功率；τ_i = 平均参考 token 数；
      T(o) = 轨迹 token 数
  Eq.(6) PDF p.5  最终实例自适应奖励:
      r_{i,k} = V(x_i,o_{i,k})·(1 + λ_{i,k}·δ) - α_i
      V ∈ {0,1} verifier 正确性；δ>0 效率 bonus 量级
      **注意：V=0 时 r = -α_i，不是 0**（与计划书原"错误=0"不同，以论文为准）
  Eq.(7)(8) PDF p.5  BWS 批次标准化:
      μ_k = (1/M)Σ_i r_{i,k},  σ_k = sqrt((1/M)Σ_i (r_{i,k}-μ_k)²)
      A_{i,k} = (r_{i,k} - μ_k) / (σ_k + ε)
      **按 trajectory index k 跨 M 个 instance 求**，同一轨迹所有 token 共享 A
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RewardConfig:
    delta: float = 1.0          # Eq.(6) 效率 bonus 量级 δ > 0
    bws_eps: float = 1e-8       # Eq.(8) 数值稳定 ε
    ppo_clip_eps: float = 0.2   # Eq.(2) PPO 裁剪 ε
    advantage_bound: float = 10.0  # 实现侧安全上界（论文未给，标注为工程约束）


class IDACReward:
    """IDAC 奖励塑形（Eq.5 + Eq.6）。"""

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()

    def lambda_scale(self, think_tokens: int, alpha_i: float, tau_i: float, thought: bool) -> float:
        """Eq.(5): λ = exp(-T·α/τ) if Think else 1。"""
        if not thought:
            return 1.0
        if tau_i <= 0:
            raise ValueError(f"τ_i 必须 > 0（平均参考 token 数），got {tau_i}")
        return math.exp(-think_tokens * alpha_i / tau_i)

    def reward(
        self,
        correct: bool,
        thought: bool,
        think_tokens: int,
        alpha_i: float,
        tau_i: float,
    ) -> float:
        """Eq.(6): r = V·(1 + λ·δ) - α_i。

        correct=False → r = -α_i（论文语义；不是 0）。
        correct=True, NoThink → λ=1 → r = 1 + δ - α_i。
        correct=True, Think   → r = 1 + λ·δ - α_i，λ 随 token 单调递减。
        """
        lam = self.lambda_scale(think_tokens, alpha_i, tau_i, thought)
        V = 1.0 if correct else 0.0
        return V * (1.0 + lam * self.cfg.delta) - alpha_i


def bws_standardize(rewards_by_k: dict[int, list[float]], cfg: RewardConfig | None = None) -> dict[int, list[float]]:
    """Eq.(7)(8): 按 trajectory index k 跨 instance 求 μ_k、σ_k，再标准化。

    rewards_by_k[k][i] = r_{i,k}，i 遍历 M 个 instance。
    返回同结构的 advantage，同一 (i,k) 的 advantage 在其轨迹内所有 token 共享。
    """
    cfg = cfg or RewardConfig()
    out: dict[int, list[float]] = {}
    for k, rs in rewards_by_k.items():
        M = len(rs)
        mu = sum(rs) / M
        var = sum((r - mu) ** 2 for r in rs) / M
        sigma = math.sqrt(var)
        adv = [(r - mu) / (sigma + cfg.bws_eps) for r in rs]
        bound = cfg.advantage_bound
        out[k] = [max(-bound, min(bound, a)) for a in adv]
    return out


def ppo_clipped_objective(rho: float, adv: float, cfg: RewardConfig | None = None) -> tuple[float, bool]:
    """Eq.(2): min(ρA, clip(ρ,1-ε,1+ε)A)。

    返回 (objective, clipped)。clipped=True 表示 ρ 越界、梯度被截断——
    这就是"IS 越界停训/冻结梯度"的实际语义（隐式约束，不是显式 clipping 操作）。
    """
    cfg = cfg or RewardConfig()
    eps = cfg.ppo_clip_eps
    clipped_rho = max(1.0 - eps, min(1.0 + eps, rho))
    obj = min(rho * adv, clipped_rho * adv)
    is_clipped = abs(rho - 1.0) > eps
    return obj, is_clipped


def weighted_objective(rho: float, adv: float, cfg: RewardConfig | None = None) -> tuple[float, bool]:
    """返回实际进入损失的价值 = -objective（PPO 是最大化 objective）。"""
    obj, clipped = ppo_clipped_objective(rho, adv, cfg)
    return -obj, clipped


class ISMonitor:
    """IS ratio 监控：ρ 越出 [1-ε, 1+ε] 即标记梯度截断，累计越界比例超阈即停训。

    论文 Eq.(2) 本身用 clip 保证有界；这里额外提供停训判据，供 fixture 与训练循环用。
    """

    def __init__(self, cfg: RewardConfig | None = None, max_clip_frac: float = 0.5):
        self.cfg = cfg or RewardConfig()
        self.max_clip_frac = max_clip_frac
        self.n = 0
        self.n_clipped = 0

    def observe(self, rho: float) -> bool:
        """返回该样本是否被 clip。"""
        self.n += 1
        _, clipped = ppo_clipped_objective(rho, 1.0, self.cfg)
        if clipped:
            self.n_clipped += 1
        return clipped

    def clip_fraction(self) -> float:
        return self.n_clipped / self.n if self.n else 0.0

    def should_stop(self) -> bool:
        return self.clip_fraction() > self.max_clip_frac
