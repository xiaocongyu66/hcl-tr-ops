"""QED：第二擦除方向（Query-derived Erase Direction）。

公式依据（SOURCE_LOCK: QED LOCKED-PAPER, arXiv:2608.13668）：
  第 3 页 eq.(6):
      a_t = b_t ⊙ k_t + λ_h · P⊥k_t(b_t ⊙ q_t)
      P⊥k(x) = x − k (k^T x)          # 论文假设 ‖k‖₂ = 1，故无 1/‖k‖² 因子
  第 3 页 eq.(8):  GDN-2 更新（见 gdn2_ref.gdn2_step）
  第 4 页 eq.(11): S_t^QED = S_t^GDN2 − λ_h · k_t (S̄_t^T d_t)^T
  第 4 页 eq.(12): o_t^QED = o_t^GDN2 − λ_h · ρ_t · (S̄_t^T d_t)
      d_t = P⊥k_t(b_t ⊙ q_t),  ρ_t = q_t^T k_t

关键约束（计划书 §4.2 / §0 红线）：
  - 不改变 GDN-2 主状态更新公式；QED 只是在 erase 向量 a_t 上加正交附加项。
  - 方向由 query 导出，且与 key 正交（eq.6 声称保持 I − k a^T 的非平凡特征值不变）。
"""

from __future__ import annotations

import torch


def project_out_key(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """P⊥k(x) = x − k (k^T x)，k 已 L2 归一化（‖k‖₂=1）。

    x: [H, d]，k: [H, d] → [H, d]
    """
    return x - k * torch.sum(k * x, dim=-1, keepdim=True)


def qed_erase_direction(
    q: torch.Tensor,        # [H, dk]  query（已 L2 归一化）
    k: torch.Tensor,        # [H, dk]  key（已 L2 归一化）
    b: torch.Tensor,        # [H, dk]  erase gate
) -> torch.Tensor:
    """d_t = P⊥k_t(b_t ⊙ q_t) —— 第 3 页 eq.(6)。"""
    return project_out_key(b * q, k)


def gdn2_step_with_qed(
    S_prev: torch.Tensor,   # [H, dk, dv]
    q: torch.Tensor,        # [H, dk]
    k: torch.Tensor,        # [H, dk]
    v: torch.Tensor,        # [H, dv]
    g: torch.Tensor,        # [H, dk]
    b: torch.Tensor,        # [H, dk]
    w: torch.Tensor,        # [H, dv]
    lam_h: float | torch.Tensor,  # λ_h >= 0，每头可学习强度（标量或 [H]）
    l2norm_qk: bool = True,
) -> tuple[torch.Tensor, dict]:
    """带 QED 的单步更新，返回 (S_t, aux)。

    实现严格按 eq.(8)→(11)：
        S̄_t   = Diag(exp(g)) S_{t-1}
        r     = S̄_t^T (b ⊙ k)
        S^GDN2= S̄_t + k (w⊙v − r)^T
        d     = P⊥k(b ⊙ q)
        S^QED = S^GDN2 − λ_h k (S̄_t^T d)^T
    并额外返回 aux 以便 fixture 校验 eq.(12) 的读出恒等式。
    """
    dt = torch.float32
    S_prev = S_prev.to(dt); q = q.to(dt); k = k.to(dt); v = v.to(dt)
    g = g.to(dt); b = b.to(dt); w = w.to(dt)
    if l2norm_qk:
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    S_bar = g.exp().unsqueeze(-1) * S_prev

    r = torch.einsum("hk,hkv->hv", b * k, S_bar)
    delta = w * v - r
    S_gdn2 = S_bar + k.unsqueeze(-1) * delta.unsqueeze(-2)

    d = qed_erase_direction(q, k, b)                 # [H, dk]
    # 正交性：d 与 k 内积应为 0（数值）
    s_bar_d = torch.einsum("hk,hkv->hv", d, S_bar)   # S̄_t^T d_t  [H, dv]
    # λ_h 支持标量或每头向量 [H]（论文 §3.1：λ_h 为 learned per-head strength）
    if isinstance(lam_h, torch.Tensor):
        lam = lam_h.to(dt).view(-1, 1, 1)            # [H,1,1] 广播到 [H,dk,dv]
    else:
        lam = lam_h
    S_qed = S_gdn2 - lam * k.unsqueeze(-1) * s_bar_d.unsqueeze(-2)

    rho = torch.sum(q * k, dim=-1)                    # [H]
    aux = {
        "S_bar": S_bar,
        "S_gdn2": S_gdn2,
        "d": d,
        "s_bar_d": s_bar_d,
        "rho": rho,
        "k": k,
        "q": q,
        "o_gdn2": torch.einsum("hk,hkv->hv", q, S_gdn2),
        "o_qed": torch.einsum("hk,hkv->hv", q, S_qed),
        "qed_read_correction": (lam_h.view(-1, 1) if isinstance(lam_h, torch.Tensor) else lam_h)
                                * rho.unsqueeze(-1) * s_bar_d,  # eq.(12) 减去的项
        "lam_h": lam_h,
    }
    return S_qed, aux


def qed_output_identity_err(aux: dict, lam_h: float) -> torch.Tensor:
    """校验 eq.(12): o_qed = o_gdn2 − λ_h ρ (S̄^T d)。

    返回两侧的最大绝对误差。
    """
    lhs = aux["o_qed"]
    lam = lam_h.view(-1, 1) if isinstance(lam_h, torch.Tensor) else lam_h
    rhs = aux["o_gdn2"] - lam * aux["rho"].unsqueeze(-1) * aux["s_bar_d"]
    return (lhs - rhs).abs().max()
