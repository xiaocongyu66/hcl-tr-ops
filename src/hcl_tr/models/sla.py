"""SLA：Softmax Linear Attention 的 head 维门控。

SOURCE_LOCK: SLA LOCKED-PAPER, arXiv:2602.01744（PDF sha256 87ba301c…）
  注意：thu-ml 的同名 SLA (arXiv:2509.24006) 是扩散模型稀疏注意力，与本模块无关。

公式依据（论文页码已核）:
  §4.1, PDF p.4:
      G_Q = softmax_h((Q W_GQ)_h)
      G_K = softmax_h((K W_GK)_h)
  线性注意力更新（Eq.5-7, PDF p.3-4）：
      S_{h,t} = S_{h,t-1} + (G_K_{h,t} ⊙ φ(k_{h,t}))^T v_{h,t}
      y_{h,t} = (G_Q_{h,t} ⊙ φ(q_{h,t})) S_{h,t}
      o_t     = Concat_{h=1..H}[y_{h,t}] W_O
  Eq.(8), PDF p.5, Theorem 4.2（magnitude sensitivity）：
      lim_{λ→∞} G_Q(λ s) = one-hot(argmax_h s_h)
  Eq.(9), PDF p.5, Theorem 4.3（Kronecker delta 路由）：
      C(λ) = Σ_h G_Qh(λ s_Q) · G_Kh(λ s_K)
      lim_{λ→∞} C(λ) = δ(argmax_h s_Qh, argmax_h s_Kh)

λ 是 query/key 的缩放强度，**不是**温度退火参数。论文未给出 λ 的调度策略，
因此实现采用固定值；不发明退火。头熵仅作监控，不作损失项 / 不作 Go/NoGo 条件。

已删除（计划书原 §4.3，论文无此内容）：α_h = softmax(s_h/t_h) 输出头竞争、
温度退火 1.5→0.5、熵正则 θ_init/θ_range/γ/λ、H_head<0.05 塌缩阈值。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """φ(·)：论文用 ReLU 型齐次特征映射（§4.1 Proposition 4.1 假设齐次）。"""
    return F.relu(x)


class SoftmaxLinearAttention(nn.Module):
    """head 维 softmax 门控的线性注意力层（论文 §4.1 + Eq.5-7）。

    Args:
        d_model: 输入维度
        n_head: 头数 H
        lam: λ 缩放强度（论文未定调度 → 固定值，默认 1.0）
    """

    def __init__(self, d_model: int, n_head: int, lam: float = 1.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.lam = float(lam)

        # Q/K/V 投影 + 门控投影（论文 W_GQ / W_GK）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.w_gq = nn.Linear(d_model, n_head, bias=True)   # (Q W_GQ)_h → per-head scalar
        self.w_gk = nn.Linear(d_model, n_head, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def head_gates(self, x_q: torch.Tensor, x_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """G_Q = softmax_h(λ·(Q W_GQ)_h)，G_K 同理（论文 §4.1 / Eq.8）。

        x_q, x_k: [B, T, d_model] → G_Q, G_K: [B, T, H]，沿 head 归一化（Σ_h = 1）。
        """
        s_q = self.w_gq(x_q) * self.lam      # [B,T,H]
        s_k = self.w_gk(x_k) * self.lam
        G_q = torch.softmax(s_q, dim=-1)
        G_k = torch.softmax(s_k, dim=-1)
        return G_q, G_k

    def forward(self, x: torch.Tensor) -> dict:
        """x: [B,T,d_model] → {out: [B,T,d_model], G_q, G_k, head_entropy}"""
        B, T, D = x.shape
        H, dh = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, H, dh)
        k = self.k_proj(x).view(B, T, H, dh)
        v = self.v_proj(x).view(B, T, H, dh)
        G_q, G_k = self.head_gates(x, x)                 # [B,T,H]

        phi_q = feature_map(q)                            # [B,T,H,dh]
        phi_k = feature_map(k)

        # 逐时间去递推状态 S_{h,t} ∈ R^{dh×dh}（Eq.5），训练/推理统一用前缀和形式
        gk = G_k.unsqueeze(-1) * phi_k                    # [B,T,H,dh] 门控后的 key
        states = torch.cumsum(
            gk.unsqueeze(-1) * v.unsqueeze(-2), dim=1
        )                                                 # [B,T,H,dh,dh] 前缀外积累加
        gq = G_q.unsqueeze(-1) * phi_q                    # [B,T,H,dh]
        y = torch.einsum("bthk,bthkv->bthv", gq, states)  # [B,T,H,dh] (Eq.6)
        o = y.reshape(B, T, D)
        out = self.o_proj(o)                              # Eq.7

        with torch.no_grad():
            head_entropy = -(G_q * (G_q + 1e-9).log()).sum(-1).mean()

        return {"out": out, "G_q": G_q, "G_k": G_k,
                "head_entropy": head_entropy}


def information_flow_coefficient(G_q: torch.Tensor, G_k: torch.Tensor) -> torch.Tensor:
    """Eq.(9) 的信息流系数 C(λ) = Σ_h G_Qh · G_Kh，逐 (B,T) 返回。"""
    return (G_q * G_k).sum(-1)


def kronecker_routing(G_q: torch.Tensor, G_k: torch.Tensor) -> torch.Tensor:
    """Eq.(9) 的极限形式：δ(argmax_h G_Qh, argmax_h G_Kh)。

    返回 bool：query 与 key 选中的 head 是否一致（winner-take-all 共识）。
    """
    return G_q.argmax(-1) == G_k.argmax(-1)


def head_entropy(G: torch.Tensor) -> torch.Tensor:
    """头选择熵（监控用，非损失项）。均匀=log H，one-hot=0。"""
    return -(G * (G + 1e-9).log()).sum(-1)


def gate_sharpness(G: torch.Tensor) -> torch.Tensor:
    """G 的最大值，用于判断是否已接近 one-hot（监控用）。"""
    return G.max(dim=-1).values
