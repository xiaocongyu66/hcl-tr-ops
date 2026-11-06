"""SPAR：选择性前缀抗干扰（Selective Anti-Interference Pretraining）。

SOURCE_LOCK: SPAR LOCKED-PAPER, arXiv:2609.27925
  第 2 页 eq: L = L_CLM + λ_inv · L_inv
  第 2 页 Figure 1 机制:
    - corrupt-prefix view：仅在 far prefix 内 shuffle chunk，local suffix 逐 token 保留
    - short-context sufficiency gate：选出 "high short-context likelihood" 且
      "small full–short likelihood gain" 的 suffix token 作为门控标签
    - 门控掩码把 clean→corrupt 的 KL 匹配限制在这些 locally sufficient token 上
    - CLM 项全程有效

计划书 §4.8 预注册二选一的门控标签来源：
  (1) held-out probe：小校准集上训练短上下文充分性探针
  (2) teacher-forced suffix likelihood：用最后 L_suffix token 的 NLL，低于阈值判为 "本地信息足够"
本实现支持 mode ∈ {"suffix_likelihood", "probe"}；训练中阈值固定、不调。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPARGate(nn.Module):
    """short-context sufficiency gate 的标签生成。

    mode="suffix_likelihood": 直接用 teacher-forced NLL 判据（无额外参数）。
    mode="probe": 训练一个小 MLP 探针输出 sufficient 概率。
    """

    def __init__(
        self,
        mode: str = "suffix_likelihood",
        d_model: int = 768,
        L_suffix: int = 512,
        nll_threshold: float = 1.0,
        gain_threshold: float = 0.1,
    ):
        super().__init__()
        assert mode in {"suffix_likelihood", "probe"}
        self.mode = mode
        self.L_suffix = L_suffix
        self.nll_threshold = nll_threshold      # "本地信息足够"判据：short NLL < 阈值
        self.gain_threshold = gain_threshold    # full−short likelihood gain 上限

        if mode == "probe":
            self.probe = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
            )
        else:
            self.probe = None

    def suffix_likelihood_gate(
        self,
        logits_clean: torch.Tensor,     # [B, T, V] clean view 的 logits
        logits_corrupt: torch.Tensor,   # [B, T, V] corrupt-prefix view 的 logits
        targets: torch.Tensor,          # [B, T]
        suffix_start: int,              # local suffix 起始位置
        logits_short: torch.Tensor | None = None,   # [B, L_suffix-1, V] 短上下文前向
        ignore_index: int = -100,
    ) -> dict:
        """按 teacher-forced suffix likelihood 生成门控掩码。

        返回:
          mask:        [B, T] bool，True 表示该位置参与 L_inv 的 KL 匹配
          nll_short:   [B, T] clean 在每个位置上的 NLL
          kl:          [B, T] clean→corrupt 的逐位置 KL 上界（用 -log q_corrupt[target] 近似计数）
        """
        if logits_short is None:
            raise ValueError(
                "SPAR 需要真正的短上下文 logits（对 suffix 单独前向）；"
                "不能用全上下文 NLL 近似")
        T = targets.shape[1]
        covered = logits_short.shape[1]
        if suffix_start + covered > T:
            raise ValueError(
                f"短上下文长度 {covered} 超出 suffix 区（T={T}, suffix_start={suffix_start}）")
        # short-context likelihood：只来自短上下文前向，覆盖 suffix 区
        nll_short = torch.full((targets.shape[0], T), float("inf"),
                               device=targets.device, dtype=torch.float32)
        nll_short[:, suffix_start:suffix_start + covered] = F.cross_entropy(
            logits_short.transpose(1, 2),
            targets[:, suffix_start:suffix_start + covered],
            reduction="none", ignore_index=ignore_index).float()
        # corrupt 视角下的 target 似然（用于 full−short gain 近似）
        nll_corrupt = F.cross_entropy(
            logits_corrupt.transpose(1, 2), targets, reduction="none", ignore_index=ignore_index
        )
        gain = nll_corrupt - nll_short    # full−short likelihood gain（越负说明前缀有害）

        # 在 suffix 区内：short NLL 低（本地充分）且 gain 小（前缀几乎无贡献）→ 选中
        pos = torch.arange(targets.shape[1], device=targets.device).view(1, -1)
        in_suffix = pos >= suffix_start
        valid = targets != ignore_index
        mask = in_suffix & valid & (nll_short < self.nll_threshold) & (gain.abs() < self.gain_threshold)

        # clean→corrupt KL 的逐位置近似（one-hot target 下 KL = log p_clean - log p_corrupt at target）
        logp_c = F.log_softmax(logits_clean, dim=-1)
        logp_k = F.log_softmax(logits_corrupt, dim=-1)
        safe_t = targets.clamp_min(0)
        lp_c = logp_c.gather(-1, safe_t.unsqueeze(-1)).squeeze(-1)
        lp_k = logp_k.gather(-1, safe_t.unsqueeze(-1)).squeeze(-1)
        kl = (lp_c - lp_k).clamp_min(0.0)

        return {"mask": mask, "nll_short": nll_short, "nll_corrupt": nll_corrupt,
                "gain": gain, "kl": kl}

    def probe_gate(self, hidden: torch.Tensor, suffix_start: int) -> torch.Tensor:
        """mode="probe"：用隐藏态探针判定本地充分性。"""
        logit = self.probe(hidden).squeeze(-1)      # [B, T]
        prob = torch.sigmoid(logit)
        pos = torch.arange(hidden.shape[1], device=hidden.device).view(1, -1)
        return (pos >= suffix_start) & (prob > 0.5)


def spar_loss(
    logits_clean: torch.Tensor,
    logits_corrupt: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,          # [B, T] bool
    lbda_inv: float = 1.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L = L_CLM + λ_inv · L_inv。

    L_CLM    : clean 视角下的常规交叉熵
    L_inv    : 被 mask 选中的位置上，clean 与 corrupt 的分布 KL（用 KLDiv 对 logits）
    """
    # L_CLM
    l_clm = F.cross_entropy(logits_clean.transpose(1, 2), targets, ignore_index=ignore_index)

    # L_inv：逐位置 KL(P_clean || P_corrupt)，仅 mask 位置平均
    logp_clean = F.log_softmax(logits_clean, dim=-1)
    logp_corrupt = F.log_softmax(logits_corrupt, dim=-1)
    # KL(P||Q) = Σ P (logP - logQ)；用 logp_clean 的 softmax 作为 P
    p_clean = logp_clean.exp()
    kl = (p_clean * (logp_clean - logp_corrupt)).sum(dim=-1)     # [B, T]

    m = mask.to(kl.dtype)
    denom = m.sum().clamp_min(1.0)
    l_inv = (kl * m).sum() / denom

    total = l_clm + lbda_inv * l_inv
    return total, l_clm, l_inv


@torch.no_grad()
def gate_audit(mask: torch.Tensor, nll_short: torch.Tensor, gain: torch.Tensor,
               threshold_nll: float, threshold_gain: float,
               suffix_start: int = 0) -> dict:
    """审计 false-gate / missed-gate 比例（计划书 §4.8：任一 >5% 冻结 SPAR）。

    truth 判据只在 local suffix 区（>= suffix_start）定义——prefix 区本就不应被门控选中，
    把它纳入 truth 会产生虚假的 missed-gate。
    """
    pos = torch.arange(nll_short.shape[-1], device=nll_short.device).view(1, -1)
    in_suffix = pos >= suffix_start
    truth = in_suffix & (nll_short < threshold_nll) & (gain.abs() < threshold_gain)
    false_gate = (mask & ~truth).sum().item() / max(1, mask.numel())
    missed_gate = (~mask & truth).sum().item() / max(1, truth.numel())
    return {"false_gate_ratio": false_gate, "missed_gate_ratio": missed_gate,
            "freeze_spar": max(false_gate, missed_gate) > 0.05}


def corrupt_prefix_view(input_ids: torch.Tensor, suffix_len: int = 512,
                        chunk_size: int = 64, generator: torch.Generator | None = None,
                        ) -> torch.Tensor:
    """构造 corrupt-prefix 视图（论文 §2 Figure 1）。

    - local suffix（最后 suffix_len 个 token）逐 token 原样保留；
    - far prefix 内按 chunk_size 切块，只打乱**块顺序**（块内 token 不变），
      从而破坏前缀的远距离依赖、同时保留局部连续性。
    """
    B, T = input_ids.shape
    keep = min(suffix_len, T)
    p = T - keep
    if p <= chunk_size:
        return input_ids.clone()

    n_chunk = p // chunk_size
    usable = n_chunk * chunk_size
    out = input_ids.clone()
    pre = input_ids[:, :usable].reshape(B, n_chunk, chunk_size)
    perm = torch.randperm(n_chunk, generator=generator, device=input_ids.device)
    out[:, :usable] = pre[:, perm].reshape(B, usable)
    return out


def spargate_suffix_mask(logits_clean: torch.Tensor, logits_corrupt: torch.Tensor,
                         targets: torch.Tensor, suffix_start: int,
                         logits_short: torch.Tensor | None = None,
                         nll_threshold: float = 1.0, gain_threshold: float = 0.1,
                         ignore_index: int = -100) -> torch.Tensor:
    """无参数版的 short-context sufficiency gate（mode="suffix_likelihood"）。

    ``logits_short`` 必须来自**只保留 suffix 上下文**的独立前向。用全上下文
    logits 代替会让 "short-context likelihood" 退化成近似，计划书 §4.8 不允许。
    """
    if logits_short is None:
        raise ValueError(
            "SPAR 需要真正的短上下文 logits（对 suffix 单独前向）；"
            "不能用全上下文 NLL 近似")
    T = targets.shape[1]
    covered = logits_short.shape[1]
    if suffix_start + covered > T:
        raise ValueError(
            f"短上下文长度 {covered} 超出 suffix 区（T={T}, suffix_start={suffix_start}）")
    nll_short = torch.full((targets.shape[0], T), float("inf"),
                           device=targets.device, dtype=torch.float32)
    short_targets = targets[:, suffix_start:suffix_start + covered]
    nll_short[:, suffix_start:suffix_start + covered] = F.cross_entropy(
        logits_short.transpose(1, 2), short_targets,
        reduction="none", ignore_index=ignore_index).float()
    nll_k = F.cross_entropy(logits_corrupt.transpose(1, 2), targets,
                            reduction="none", ignore_index=ignore_index)
    gain = nll_k - nll_short
    pos = torch.arange(T, device=targets.device).view(1, -1)
    return (pos >= suffix_start) & (targets != ignore_index) & \
           torch.isfinite(nll_short) & \
           (nll_short < nll_threshold) & (gain.abs() < gain_threshold)
