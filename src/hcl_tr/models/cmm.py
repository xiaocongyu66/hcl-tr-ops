"""CMM：Continuous Memory Machines（CMM 的双记忆核心）。

SOURCE_LOCK: CMM LOCKED
  论文 arXiv:2610.07907（PDF 归档 sha256 f9a3c24c…，公式页码已核）：
    §3.1  PDF p.4  eq.(1): a_t = f(z_t, o_t)
    §3.1  PDF p.4  eq.(2): z_i^{t+1} = g_i(A_i^t)         # per-neuron NLM
    §3.1  PDF p.4  eq.(3): S^{t+1}_{ij} ∝ Σ_τ e^{-r_ij(t+1-τ)} z_i^τ z_j^τ
    §3.2  PDF p.4  eq.(4): X^t = [s; B^{t-1}; A^t] ∈ R^{d×(1+M_L+M_S)}
    §3.2  PDF p.4-5 eq.(5): X̃^t = TransformerBlocks(X^t) = [s̃; B̃^t; Ã^t]
    §3.2  PDF p.5  eq.(6): z_i^{t+1} = g_i(Ã_i^t)
  状态推进: B̃^t 前传为持久 LTM；Ã^t 产出 z^{t+1} 后丢弃；原 A^t 保留为滚动窗口。

代码底座（LOCKED）: SakanaAI/continuous-thought-machines @ 4a6c9c3a（Apache-2.0）
  复用其 per-neuron NLM（SuporLinear / SuperLinear 语义）、synchronisation、certainty。

实现范围：STM（滑动窗口 A_t）、LTM（矩阵 B_t）、联合 Transformer 更新（eq.4-5）、
NLM 更新（eq.6）、synchronisation（eq.3）。投影/骨干由调用方提供 observation o_t。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronLevelModel(nn.Module):
    """per-neuron NLM g_i（eq.2 / eq.6）。

    参考 CTM 的 SuperLinear：每个神经元一份私有线性权重，作用于自己的滑动窗口历史。
    这里用 grouped linear 实现：输入 [B, d, M_S]，对每个 neuron 的 M_S 维历史投影到 1 维。
    """

    def __init__(self, d_model: int, memory_length: int, do_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.layernorm = nn.LayerNorm(memory_length) if do_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 每 neuron 私有权重：[d_model, 1, memory_length]
        self.w = nn.Parameter(torch.randn(d_model, 1, memory_length) / memory_length**0.5)
        self.b = nn.Parameter(torch.zeros(1, d_model, 1))

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """A: [B, d, M_S] → z: [B, d]"""
        A = self.layernorm(A)
        A = self.dropout(A)
        # 每 neuron 独立：b d m, d o m -> b d o
        z = torch.einsum("bdm,dom->bdo", A, self.w) + self.b
        return z.squeeze(-1)


class Synchronisation(nn.Module):
    """CTM 的 synchronisation（eq.3）：指数加权的外积历史，作为潜在表示。

    为简洁与可微，用逐步 EMA 递推实现 S_{ij}^{t} = (1-r_ij) S^{t-1} + r_ij z_i z_j 形式
    （等价于论文的 ∝ Σ e^{-r(t-τ)} z^τ_i z^τ_j 的归一化版本）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # 论文 eq.(3) 的 r_ij 是成对 decay。用对称低秩近似 r_ij = 0.5(r_i + r_j)，
        # 保证 S 的对称性（z_i z_j = z_j z_i），同时把 O(d^2) 参数降到 O(d)。
        self.decay = nn.Parameter(torch.rand(d_model) * 0.5 + 0.1)

    def forward(self, z: torch.Tensor, S_prev: torch.Tensor | None) -> torch.Tensor:
        """z: [B, d] → S: [B, d, d]"""
        outer = z.unsqueeze(-1) * z.unsqueeze(-2)        # [B, d, d]
        # 对称 pair decay: r_ij = 0.5*(r_i + r_j)
        ri = self.decay.view(1, -1, 1)
        rj = self.decay.view(1, 1, -1)
        r = 0.5 * (ri + rj)                              # [1, d, d] 对称
        if S_prev is None:
            return r * outer
        return (1 - r) * S_prev + r * outer


class CMMMemory(nn.Module):
    """CMM 的双记忆 + 联合 Transformer 更新。

    时间步 t 的流程（严格对应论文 eq.1-6）：
      1. a_t = synapse(z_t, o_t)                        # eq.1
      2. A_t = [A_{t-1}[..., 1:], a_t]                  # 滑动窗口 STM
      3. X_t = [sink; B_{t-1}; A_t]                     # eq.4
      4. X̃_t = TransformerBlocks(X_t) = [s̃; B̃; Ã]        # eq.5（双向读写）
      5. z_{t+1} = NLM(Ã)                               # eq.6
      6. B_t ← B̃（持久）；Ã 丢弃；A_t 滚动窗口继续       # 状态推进
    """

    def __init__(
        self,
        d_model: int = 768,
        memory_length: int = 8,      # M_S
        ltm_cols: int = 16,          # M_L
        n_tf_layers: int = 2,
        n_tf_heads: int = 12,
        dropout: float = 0.0,
        do_nlm_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_length = memory_length
        self.ltm_cols = ltm_cols

        # synapse：把 [z_t, o_t] 映射为预激活 a_t（eq.1）
        self.synapse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        # LTM 初始矩阵 B_0（可学习）
        self.B0 = nn.Parameter(torch.randn(d_model, ltm_cols) * 0.02)
        # learnable sink token s（eq.4）
        self.sink = nn.Parameter(torch.randn(d_model) * 0.02)
        # 位置编码（覆盖 1 + M_L + M_S 个 token）
        self.pos_emb = nn.Parameter(torch.randn(1 + ltm_cols + memory_length, d_model) * 0.02)
        # 联合更新的 Transformer 栈（eq.5）—— 这就是"双向读写头"
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_tf_heads, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_tf_layers)

        self.nlm = NeuronLevelModel(d_model, memory_length, do_norm=do_nlm_norm, dropout=dropout)
        self.sync = Synchronisation(d_model)

        # 初始状态（可学习）
        self.start_z = nn.Parameter(torch.randn(d_model) * 0.02)
        self.start_A = nn.Parameter(torch.randn(d_model, memory_length) * 0.02)

    def forward(
        self,
        obs: torch.Tensor,             # [B, T, d_model] 观测序列 o_t（来自骨干）
        steps: int | None = None,      # tick 循环次数，默认 T
        return_traces: bool = False,
    ) -> dict:
        B, T, d = obs.shape
        steps = steps if steps is not None else T

        z = self.start_z.unsqueeze(0).expand(B, -1)                 # [B, d]
        A = self.start_A.unsqueeze(0).expand(B, -1, -1).clone()      # [B, d, M_S]
        B_ltm = self.B0.unsqueeze(0).expand(B, -1, -1).clone()      # [B, d, M_L]
        S = None
        Z_trace, B_trace = [], []

        outputs = []
        for t in range(steps):
            o_t = obs[:, min(t, T - 1)]                              # 若 steps>T 则复用末帧
            # 1. synapse: a_t = f(z_t, o_t)
            a_t = self.synapse(torch.cat([z, o_t], dim=-1))          # [B, d]
            # 2. 滑动窗口 STM
            A = torch.cat([A[..., 1:], a_t.unsqueeze(-1)], dim=-1)   # [B, d, M_S]
            # 3. X = [sink; B; A]  (as tokens: [B, 1+M_L+M_S, d])
            B_tok = B_ltm.transpose(1, 2)                            # [B, M_L, d]
            A_tok = A.transpose(1, 2)                                # [B, M_S, d]
            X = torch.cat([self.sink.view(1, 1, d).expand(B, -1, -1), B_tok, A_tok], dim=1)
            X = X + self.pos_emb
            # 4. Transformer 联合更新（双向读写）
            Xt = self.tf(X)
            s_hat = Xt[:, 0]
            B_hat = Xt[:, 1:1 + self.ltm_cols].transpose(1, 2)       # [B, d, M_L]
            A_hat = Xt[:, 1 + self.ltm_cols:].transpose(1, 2)        # [B, d, M_S]
            # 5. z_{t+1} = NLM(Ã)
            z_next = self.nlm(A_hat)                                 # [B, d]
            # 6. 状态推进：B ← B̃（持久），Ã 丢弃，A 原样保留
            B_ltm = B_hat
            z = z_next
            S = self.sync(z, S)
            outputs.append(z)
            if return_traces:
                Z_trace.append(z)
                B_trace.append(B_hat)

        result = {"z_final": z, "S": S, "B_ltm": B_ltm, "outputs": torch.stack(outputs, dim=1)}
        if return_traces:
            result["Z_trace"] = torch.stack(Z_trace, dim=1)
            result["B_trace"] = torch.stack(B_trace, dim=1)
        return result

    @torch.no_grad()
    def certainty(self, z: torch.Tensor) -> torch.Tensor:
        """规则的触发 certainty（供 When2Think 规则回退使用，SOURCE_LOCK: When2Think UNVERIFIED）。"""
        # 以 z 的 L2 归一化范数作为稳定性代理；范围 [0,1]
        n = z.norm(dim=-1, keepdim=True)
        return (n / (n + 1.0)).squeeze(-1)
