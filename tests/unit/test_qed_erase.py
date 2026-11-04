"""QED 数值 fixture：正交性与读出恒等式。

SOURCE_LOCK: QED LOCKED-PAPER, arXiv:2608.13668
  第 3 页 eq.(6):  a_t = b_t ⊙ k_t + λ_h P⊥k(b_t ⊙ q_t),  P⊥k(x)=x−k(k^T x)
  第 4 页 eq.(11): S^QED = S^GDN2 − λ_h k (S̄^T d)^T
  第 4 页 eq.(12): o^QED = o^GDN2 − λ_h ρ (S̄^T d)

容差（计划书 §10）:
  - QED 正交性 < 1e-6
  - 状态/读出 max_abs_err <= 1e-5 (fp32 reference)
"""

from __future__ import annotations

import pytest
import torch

from hcl_tr.models.qed import (
    gdn2_step_with_qed,
    project_out_key,
    qed_erase_direction,
    qed_output_identity_err,
)

SEED = 4321
TOL_ORTHO = 1e-6
TOL_FP32 = 1e-5
DTYPE_INFO = {"dtype": "float32", "kernel": "pytorch-reference"}


def _rand(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32) * scale


def _l2norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def test_project_out_key_is_orthogonal():
    """P⊥k(x) 与 k 正交。"""
    H, d = 8, 32
    k = _l2norm(_rand((H, d), seed=1))
    x = _rand((H, d), seed=2)
    y = project_out_key(x, k)
    dot = (y * k).sum(dim=-1).abs().max().item()
    assert dot < TOL_ORTHO, {**DTYPE_INFO, "max_abs_dot": dot}


def test_qed_direction_orthogonal_to_key():
    """eq.(6) 的第二项 d_t 与 k_t 正交。"""
    H, dk = 8, 32
    q = _l2norm(_rand((H, dk), seed=3))
    k = _l2norm(_rand((H, dk), seed=4))
    b = torch.rand((H, dk), generator=torch.Generator().manual_seed(5))
    d = qed_erase_direction(q, k, b)
    dot = (d * k).sum(dim=-1).abs().max().item()
    assert dot < TOL_ORTHO, {**DTYPE_INFO, "max_abs_dot": dot}


def test_qed_preserves_non_trivial_eigenvalue():
    """eq.(6) 声称 I − k a^T 的非平凡特征值不变；即 k^T a 不变。

    k^T a_new = k^T(b⊙k) + λ_h k^T d = k^T(b⊙k) + 0.
    """
    H, dk, dv = 4, 16, 16
    S_prev = _rand((H, dk, dv), seed=6)
    q = _l2norm(_rand((H, dk), seed=7))
    k = _l2norm(_rand((H, dk), seed=8))
    v = _rand((H, dv), seed=9)
    g = -torch.rand((H, dk), generator=torch.Generator().manual_seed(10))
    b = torch.rand((H, dk), generator=torch.Generator().manual_seed(11))
    w = torch.rand((H, dv), generator=torch.Generator().manual_seed(12))

    lam = 0.7
    S, aux = gdn2_step_with_qed(S_prev, q, k, v, g, b, w, lam, l2norm_qk=False)

    k_final = aux["k"]
    a_gdn2 = b * k_final
    a_qed = a_gdn2 + lam * aux["d"]
    err = (torch.sum(k_final * a_qed, -1) - torch.sum(k_final * a_gdn2, -1)).abs().max().item()
    assert err < TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_qed_lam_zero_recovers_gdn2():
    """λ_h=0 时 QED 退回 GDN-2。"""
    H, dk, dv = 4, 16, 16
    S_prev = _rand((H, dk, dv), seed=13)
    q = _l2norm(_rand((H, dk), seed=14))
    k = _l2norm(_rand((H, dk), seed=15))
    v = _rand((H, dv), seed=16)
    g = -torch.rand((H, dk), generator=torch.Generator().manual_seed(17))
    b = torch.rand((H, dk), generator=torch.Generator().manual_seed(18))
    w = torch.rand((H, dv), generator=torch.Generator().manual_seed(19))

    S, aux = gdn2_step_with_qed(S_prev, q, k, v, g, b, w, lam_h=0.0, l2norm_qk=False)
    err = (S - aux["S_gdn2"]).abs().max().item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_qed_eq12_output_identity():
    """eq.(12) 读出恒等式成立。"""
    H, dk, dv = 6, 24, 24
    S_prev = _rand((H, dk, dv), seed=20, scale=0.5)
    q = _l2norm(_rand((H, dk), seed=21))
    k = _l2norm(_rand((H, dk), seed=22))
    v = _rand((H, dv), seed=23)
    g = -torch.rand((H, dk), generator=torch.Generator().manual_seed(24))
    b = torch.rand((H, dk), generator=torch.Generator().manual_seed(25))
    w = torch.rand((H, dv), generator=torch.Generator().manual_seed(26))

    lam = 1.3
    S, aux = gdn2_step_with_qed(S_prev, q, k, v, g, b, w, lam, l2norm_qk=True)
    err = qed_output_identity_err(aux, lam).item()
    assert err <= TOL_FP32, {**DTYPE_INFO, "max_abs_err": err}


def test_qed_erases_query_direction_old_state():
    """QED 的核心效果：减去 query 方向上的旧状态分量。

    构造对称场景：旧状态恰好有沿 q 方向的分量，验证读出在 σ 方向减小。
    """
    H, dk, dv = 1, 8, 8
    torch.manual_seed(SEED)
    # 构造 ρ = q·k ≠ 0：q 与 k 同向，故 delta 更新沿 q 能到达读出。
    # 但 QED 的擦除方向 d = P⊥k(b⊙q) 在 q∥k 时为 0，故改用 q 含 k 正交分量：
    dk_dim = dk
    q = torch.zeros(H, dk_dim); q[0, 0] = 1.0; q[0, 1] = 1.0
    k = torch.zeros(H, dk_dim); k[0, 1] = 1.0      # ρ = q·k = 1 ≠ 0
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    # 旧状态：沿 q 方向有旧内容（通过 v=0 的 key 轴 e0 分量存 S[0, e0, :]）
    S_prev = torch.zeros(H, dk_dim, dv)
    S_prev[0, 0, 0] = 5.0
    v = torch.zeros(H, dv)
    g = torch.zeros(H, dk_dim)
    b = torch.ones(H, dk_dim)
    w = torch.zeros(H, dv)

    # GDN-2 读出（λ=0）：o_gdn2 = q^T S_gdn2
    _, aux0 = gdn2_step_with_qed(S_prev, q, k, v, g, b, w, lam_h=0.0, l2norm_qk=False)
    o0 = (q.unsqueeze(-1) * aux0["o_gdn2"]).sum().item()

    # QED 读出（λ=1）：ρ≠0 且 d≠0，eq.(12) 的修正项非零
    _, aux1 = gdn2_step_with_qed(S_prev, q, k, v, g, b, w, lam_h=1.0, l2norm_qk=False)
    o1 = (q.unsqueeze(-1) * aux1["o_qed"]).sum().item()

    assert abs(aux1["rho"].item()) > 1e-6, "构造应满足 ρ≠0"
    assert o0 > 1.0, f"baseline read should see old content, got {o0}"
    assert o1 < o0, f"QED should reduce old-state read along q: {o0} -> {o1}"
