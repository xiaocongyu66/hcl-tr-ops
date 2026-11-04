"""HybridBackbone：GDN-2 线性层 + 3:1 全注意力的混合骨干（MVP-1a）。

结构（计划书 §1、§4.1、§12 第 4 步）：
  每 4 层为一个 group：3 层 GDN-2 + 1 层全注意力（比例 3:1）。
  - 全注意力层：per-head RMSNorm（RoPE 之前）QK-norm，符合 §5 qk_norm.full_attention=true
  - GDN-2 层：复用 gdn2_ref 的 recurrence；支持可选 QED 与 SLA 消融
  - SLA：头级竞争门控，默认关闭（MVP-1b 才开）
  - divide_by_sqrt_t 序列长度归一化 + sigmoid output gate

本文件提供 **纯 PyTorch 参考实现**，用于 CPU 上的形状/梯度/配置校验与 fixture；
生产训练可切换 vendor/gdn2 的 Triton 算子（同一公式，由 fixture 保证数值一致）。
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gdn2_ref import (gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward,
                       gdn2_recurrent_forward, gdn2_step)
from .sla import SoftmaxLinearAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, backend: str = "torch"):
        super().__init__()
        if backend not in ("torch", "triton"):
            raise ValueError(f"不支持的 RoPE backend: {backend}")
        self.backend = backend
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q, k):
        if self.backend == "triton":
            from .rope_fused import fused_rope
            if q.device.type != "cuda" or k.device.type != "cuda":
                raise RuntimeError("Triton RoPE 需要 CUDA 张量")
            if q.device != k.device or q.shape != k.shape or q.shape[-1] % 2:
                raise ValueError("Triton RoPE 输入 shape/device 不匹配或 head_dim 为奇数")
            t = torch.arange(q.shape[1], device=q.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(q.device))
            cos, sin = freqs.cos(), freqs.sin()
            return fused_rope(q, k, cos, sin)
        T = q.shape[1]
        t = torch.arange(T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)          # [T, head_dim/2]
        freqs = torch.cat([freqs, freqs], dim=-1)                  # [T, head_dim]（rotate_half 配对）
        cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class FullAttention(nn.Module):
    """带 per-head QK-norm（RoPE 之前）的全注意力层。"""

    def __init__(self, d_model: int, n_head: int, qk_norm: bool = True,
                 rope_backend: str = "torch"):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # q/k/v 融合成一次 GEMM（同上：减少 kernel 数、增大 N 维）
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, backend=rope_backend)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.in_proj(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # RMSNorm 权重通常保持 fp32；AMP 下 q/k 必须回到 v 的计算 dtype，
        # 否则 SDPA 会被 fp32 q/k 强制拖出半精度路径。
        if self.rope.backend == "triton" and q.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
        # RoPE 在 qk-norm 之后
        q, k = self.rope(q.transpose(1, 2), k.transpose(1, 2))   # [B,T,H,d]
        q, k = q.transpose(1, 2), k.transpose(1, 2)              # [B,H,T,d]
        # SDPA：flash/memory-efficient 后端不物化 [B,H,T,T]（8K 下手写 softmax 需 6GB），
        # 且 is_causal=True 提供因果掩码——自回归 LM 绝不能看到未来 token。
        if mask is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, D)                    # [B,T,D]
        return self.o_proj(o)


class GDN2Layer(nn.Module):
    """纯 PyTorch GDN-2 层（可选 QED）。

    注：SLA 不在本层内。按论文 arXiv:2602.01744，SLA 是**线性注意力层的 Q/K head 门控**，
    是独立的混合层类型（见 SLA 层 / HybridBlock 的 "sla" kind），不是 GDN-2 的输出门控。
    """

    def __init__(self, d_model: int, n_head: int, use_qed: bool = False,
                 divide_by_sqrt_t: bool = True, output_gate: bool = True,
                 qk_norm: bool = False, output_gate_bias_init: float = 1.0,
                 chunk: int = 64, use_chunkwise: bool = True,
                 use_sla: bool = False, sla_lam: float = 1.0,
                 matmul_dtype: str = "fp32", fused_proj: bool = True,
                 sub_chunk: int = 64, chunk_override: int | None = None,
                 solve_mode: str = "solver", dt_min: float = 1e-3,
                 dt_max: float = 1e-1, a_log_min: float = 1.0,
                 a_log_max: float = 16.0, chunk_backend: str = "serial"):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        dk = dv = self.head_dim
        self.fused_proj = fused_proj
        if fused_proj:
            # 7 个投影融合成一次 GEMM（q,k,v,b,w,f 各 d_model，外加输出门）
            self.in_proj = nn.Linear(d_model, 7 * d_model, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, n_head * dk, bias=False)
            self.k_proj = nn.Linear(d_model, n_head * dk, bias=False)
            self.v_proj = nn.Linear(d_model, n_head * dv, bias=False)
            self.b_proj = nn.Linear(d_model, n_head * dk, bias=False)
            self.w_proj = nn.Linear(d_model, n_head * dv, bias=False)
            self.f_proj = nn.Linear(d_model, n_head * dk, bias=False)
            self.out_gate = nn.Linear(d_model, n_head * dv, bias=True)
        self.o_proj = nn.Linear(n_head * dv, d_model, bias=False)
        # A_log ~ log-uniform(a_log_min, a_log_max)（计划书为 (1,16)）。
        # 它决定每 token 衰减的上界 exp(A_log)·softplus(...)，进而决定精确 chunk。
        self.A_log = nn.Parameter(torch.log(torch.empty(n_head).uniform_(a_log_min, a_log_max)))
        # dt_bias 用标准 SSM 初始化（计划书未规定；见 SOURCE_LOCK.md 记录）。
        # 初始 dt=softplus(dt_bias) ~ log-uniform(0.001, 0.1)，
        # 使初始每 token 衰减 exp(A_log)*dt 落在 [0.001, 1.6]，
        # 而不是 dt_bias=0 时的 [0.69, 11]（那样初始记忆不足 1 token，
        # 且会让 chunkwise 因指数范围过大而失真）。
        _dt = torch.exp(torch.rand(n_head * dk) * (math.log(dt_max) - math.log(dt_min))
                        + math.log(dt_min))
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))

        self.use_qed = use_qed
        self.divide_by_sqrt_t = divide_by_sqrt_t
        self.output_gate = output_gate
        # l2norm_qk 是 GDN-2 数值稳定的必要条件（关闭后状态爆炸 ~1e9，fp16 下变 NaN）。
        # 它不是消融项——计划书 §4.1 的"内部 QK-norm 默认关闭"指的是另一回事，
        # 此前把两者接到一起导致训练第 2 步就 NaN。
        if qk_norm:
            raise NotImplementedError(
                "gdn2_internal_qk_norm=True 尚未作为独立消融实现；"
                "GDN-2 的 l2norm_qk 始终开启（稳定性必需）。"
            )
        self.qk_norm = True
        self.chunk = chunk
        self.sub_chunk = sub_chunk
        self._chunk_eff: int | None = None      # 缓存的自适应 chunk
        self._calib_record = None
        self._last_dmax: float = 0.0
        self.chunk_override = chunk_override
        self.solve_mode = solve_mode
        if chunk_backend not in ("serial", "batched", "fused"):
            raise ValueError(f"不支持的 GDN2 chunk backend: {chunk_backend}")
        if chunk_backend == "fused" and (use_qed or use_sla):
            raise NotImplementedError("fused GDN2 backend 暂不支持 QED/SLA")
        self.chunk_backend = chunk_backend
        self.use_chunkwise = use_chunkwise
        # SLA 叠在 GDN-2 上（计划书 §9「在 +GDN/QED 上加 SLA」）：
        # G_Q/G_K 是 head 维 softmax 门控，乘在 l2norm 后的 q/k 上（论文 arXiv:2602.01744 §4.1）
        self.use_sla = use_sla
        self.sla_lam = sla_lam
        # 矩阵乘精度：T4 上 fp16 张量核是 fp32 的 8 倍；状态始终 fp32
        self.matmul_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                             "bf16": torch.bfloat16}[matmul_dtype]
        if self.chunk_backend == "fused" and self.matmul_dtype != torch.float32:
            raise ValueError("fused GDN2 backend 当前只支持 fp32 matmul")
        if use_sla:
            self.sla_gq = nn.Linear(d_model, n_head, bias=True)
            self.sla_gk = nn.Linear(d_model, n_head, bias=True)

        if use_qed:
            # λ_h 每头可学习强度（QED §3.1 eq.6）。存 raw 参数，前向用 softplus 保证 >=0。
            self.lam_h_raw = nn.Parameter(torch.zeros(n_head))
        self.output_gate_bias_init = output_gate_bias_init

    @contextmanager
    def chunk_calibration_context(self, record):
        previous = self._calib_record
        self._calib_record = record
        try:
            yield
        finally:
            self._calib_record = previous

    def checkpoint_contexts(self):
        # 每次 checkpoint 独占记录；后续前向不能覆盖本次反向重算的分块。
        record = {}
        return self.chunk_calibration_context(record), self.chunk_calibration_context(record)

    def forward(self, x, collect_sla=False):
        B, T, D = x.shape
        H, dh = self.n_head, self.head_dim
        D = H * dh
        if self.fused_proj:
            q, k, v, b, w, pre, gate = self.in_proj(x).split(D, dim=-1)
            gate = gate.view(B, T, H, dh)
        else:
            q, k, v, b, w, pre = (p(x) for p in
                                  (self.q_proj, self.k_proj, self.v_proj,
                                   self.b_proj, self.w_proj, self.f_proj))
            gate = self.out_gate(x).view(B, T, H, dh)
        q = q.view(B, T, H, dh)
        k = k.view(B, T, H, dh)
        v = v.view(B, T, H, dh)
        b = b.view(B, T, H, dh).sigmoid()
        w = w.view(B, T, H, dh).sigmoid()
        pre = pre.view(B, T, H, dh)
        # g = -exp(A_log) * softplus(pre + dt_bias)
        g = -self.A_log.exp().view(1, 1, H, 1) * F.softplus(pre + self.dt_bias.view(1, 1, H, dh))

        # QED 的额外 erase 项只有逐 token 版支持；非 QED 走 chunkwise 并行形式，
        # Python 迭代从 T 降到 T/chunk（逐 token 实测仅 ~1.5k tok/s，10B 不可行）。
        gq = gk = None
        if self.use_sla:
            # Eq.(4)/§4.1：G_Q = softmax_h(λ (x W_GQ)_h)，G_K 同理
            gq = torch.softmax(self.sla_gq(x) * self.sla_lam, dim=-1)   # [B,T,H]
            gk = torch.softmax(self.sla_gk(x) * self.sla_lam, dim=-1)

        if self.use_qed:
            lam = F.softplus(self.lam_h_raw)          # [H]，每头独立
            o = gdn2_recurrent_forward(
                q, k, v, g, b, w,
                divide_by_sqrt_t=self.divide_by_sqrt_t,
                l2norm_qk=True, lam_h=lam, gq=gq, gk=gk,
            )
        elif self.use_chunkwise:
            record = self._calib_record
            if record is not None and "chunk_eff" in record:
                chunk_eff, dmax = record["chunk_eff"], record["dmax"]
            else:
                with torch.no_grad():
                    dmax = float((-g.float()).amax()) if g.numel() else 0.0
                if not math.isfinite(dmax):
                    raise ValueError("GDN2 衰减包含 NaN/Inf，停止训练")
                safe = 10.0 if self.matmul_dtype == torch.float16 else 40.0
                chunk_eff = self.chunk if dmax <= 0 else max(1, min(self.chunk, int(safe / dmax)))
                if self.chunk_override is not None:
                    chunk_eff = max(1, min(self.chunk, self.chunk_override))
                if record is not None:
                    record.update(chunk_eff=chunk_eff, dmax=dmax)
            self._chunk_eff, self._last_dmax = chunk_eff, dmax
            if self.chunk_backend == "fused":
                from .gdn2_chunk_opt import chunkwise_fused
                o = chunkwise_fused(
                    q, k, v, g, b, w, chunk=self.chunk,
                    divide_by_sqrt_t=self.divide_by_sqrt_t, l2norm_qk=True,
                    matmul_dtype=self.matmul_dtype, chunk_eff=self._chunk_eff,
                )
            else:
                chunk_forward = (gdn2_chunkwise_batched_forward if self.chunk_backend == "batched"
                                 else gdn2_chunkwise_forward)
                o = chunk_forward(
                    q, k, v, g, b, w, chunk=self.chunk,
                    divide_by_sqrt_t=self.divide_by_sqrt_t, l2norm_qk=True,
                    gq=gq, gk=gk, matmul_dtype=self.matmul_dtype,
                    sub_chunk=self.sub_chunk, chunk_override=self.chunk_override,
                    chunk_eff_in=self._chunk_eff, solve_mode=self.solve_mode,
                )
        else:
            o = gdn2_recurrent_forward(
                q, k, v, g, b, w,
                divide_by_sqrt_t=self.divide_by_sqrt_t, l2norm_qk=True,
                gq=gq, gk=gk,
            )   # [B,T,H,dh]

        result = {"y": o}
        if self.output_gate:
            o = o * torch.sigmoid(gate)
        o = o.reshape(B, T, H * dh).to(x.dtype)
        result["out"] = self.o_proj(o)
        return result


class HybridBlock(nn.Module):
    MLP_MULT = 2

    def __init__(self, d_model, n_head, kind: str, **kw):
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(d_model)
        if kind == "gdn2":
            self.mixer = GDN2Layer(d_model, n_head, **kw)
        elif kind == "attn":
            self.mixer = FullAttention(d_model, n_head, qk_norm=kw.get("qk_norm", True),
                                       rope_backend=kw.get("rope_backend", "torch"))
        elif kind == "sla":
            # 论文 arXiv:2602.01744：head 维 softmax 门控的线性注意力层
            self.mixer = SoftmaxLinearAttention(d_model, n_head, lam=kw.get("sla_lam", 1.0))
        elif kind == "mlp":
            self.mixer = None
        self.norm2 = RMSNorm(d_model)
        # MLP 扩张系数 2（而非 4）：混合骨干里 GDN-2 层本身参数重，
        # 4× 会把 non-embed 推到 106M（计划书 §3 口径要求 ~75.4M）。
        # 2× 实测 non_embed=77.9M / total=102.4M，符合计划书 ~100M。
        self.mlp = nn.Sequential(
            nn.Linear(d_model, self.MLP_MULT * d_model, bias=False),
            nn.GELU(),
            nn.Linear(self.MLP_MULT * d_model, d_model, bias=False),
        )

    def forward(self, x, collect_sla=False):
        if self.mixer is not None:
            if self.kind == "gdn2":
                x = x + self.mixer(self.norm1(x), collect_sla=collect_sla)["out"]
            else:
                x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HybridBackbone(nn.Module):
    """3:1 GDN-2 : FullAttention 交错骨干预训练模型。

    Args:
        attn_ratio: 每 (attn_ratio) 层中 1 层全注意力（默认 4 → 3 层 GDN-2 + 1 层注意力）。
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        attn_ratio: int = 4,
        use_qed: bool = False,
        use_sla: bool = False,
        qk_norm_full_attn: bool = True,
        rope_backend: str = "torch",
        divide_by_sqrt_t: bool = True,
        output_gate: bool = True,
        tie_embeddings: bool = True,
        grad_checkpointing: bool = False,
        gdn2_internal_qk_norm: bool = False,   # 计划书 §4.1：默认关闭的独立消融开关
        output_gate_bias_init: float = 1.0,    # 计划书 §5 bias_init
        use_spar: bool = False,                # §9 归因组开关（训练目标未实现时大声报错）
        use_idac: bool = False,                # 同上
        sla_lam: float = 1.0,                  # SLA 门控缩放强度（论文未定调度 → 固定）
        gdn2_matmul_dtype: str = "fp32",       # GDN-2 矩阵乘精度（fp16 用张量核）
        gdn2_fused_proj: bool = True,          # 7 投影融合（A/B 可关）
        gdn2_sub_chunk: int = 64,              # 保留字段
        gdn2_chunk_override: int | None = None,  # 基准用：强制 chunk
        gdn2_solve_mode: str = "solver",         # "solver" | "neumann"（可图捕获）
        gdn2_dt_min: float = 1e-3,               # dt ~ log-uniform[dt_min, dt_max]（SSM 标准）
        gdn2_dt_max: float = 1e-1,
        gdn2_a_log_min: float = 1.0,             # A_log ~ log-uniform[min,max]（计划书 1..16）
        gdn2_a_log_max: float = 16.0,
        gdn2_chunk_backend: str = "serial",
        gdn2_chunk: int = 64,                  # chunkwise GDN-2 的块大小
        use_chunkwise: bool = True,            # 逐 token 循环太慢，默认用 chunkwise
    ):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing
        self.gdn2_internal_qk_norm = gdn2_internal_qk_norm
        self.use_spar = use_spar
        self.use_idac = use_idac
        self.gdn2_chunk = gdn2_chunk
        self.use_chunkwise = use_chunkwise
        assert n_layer % attn_ratio == 0 or attn_ratio == 1
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(0.0)

        blocks = []
        for i in range(n_layer):
            is_attn = (i % attn_ratio == attn_ratio - 1)
            if is_attn:
                blocks.append(HybridBlock(d_model, n_head, "attn", qk_norm=qk_norm_full_attn,
                                           rope_backend=rope_backend))
            else:
                # SLA 叠在 GDN-2 上（+SLA 组 = GDN-2 + QED + SLA 门控），
                # 参数量与 +GDN/QED 对齐，只多每层的 2×d_model×n_head 门控参数。
                blocks.append(HybridBlock(
                    d_model, n_head, "gdn2",
                    use_qed=use_qed,
                    divide_by_sqrt_t=divide_by_sqrt_t, output_gate=output_gate,
                    qk_norm=gdn2_internal_qk_norm,
                    output_gate_bias_init=output_gate_bias_init,
                    chunk=gdn2_chunk, use_chunkwise=use_chunkwise,
                    use_sla=use_sla, sla_lam=sla_lam,
                    matmul_dtype=gdn2_matmul_dtype, fused_proj=gdn2_fused_proj,
                    sub_chunk=gdn2_sub_chunk, chunk_override=gdn2_chunk_override,
                    solve_mode=gdn2_solve_mode, dt_min=gdn2_dt_min, dt_max=gdn2_dt_max,
                    a_log_min=gdn2_a_log_min, a_log_max=gdn2_a_log_max,
                    chunk_backend=gdn2_chunk_backend,
                ))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # GPT-2 风格缩放初始化：默认 nn.Embedding 是 N(0,1)，配合 RMSNorm+残差会让
        # logits 爆炸（实测初始 loss≈628，而 ln(32000)≈10.4）。统一用 0.02。
        self.apply(self._init_weights)
        # 残差输出投影按 1/sqrt(2·n_layer) 再缩一次，抑制深残差累积
        for blk in self.blocks:
            if getattr(blk, "mixer", None) is not None and hasattr(blk.mixer, "o_proj"):
                nn.init.normal_(blk.mixer.o_proj.weight, std=0.02 / (2 * n_layer) ** 0.5)
            nn.init.normal_(blk.mlp[2].weight, std=0.02 / (2 * n_layer) ** 0.5)
            # output gate 的 bias 必须恢复为 1.0（计划书 §5 bias_init=1.0），
            # 上面的 _init_weights 会把它清零
            mixer = getattr(blk, "mixer", None)
            if (mixer is not None and getattr(mixer, "fused_proj", False)
                    and getattr(getattr(mixer, "in_proj", None), "bias", None) is not None):
                # 融合投影的 bias 前 6D 保持 0，最后 D 段是输出门（bias_init=1.0）
                with torch.no_grad():
                    mixer.in_proj.bias[-d_model:].fill_(1.0)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def chunk_stats(self):
        out = {}
        for i, blk in enumerate(self.blocks):
            mixer = getattr(blk, "mixer", None)
            if mixer is not None and hasattr(mixer, "_chunk_eff"):
                out[i] = (mixer._chunk_eff, round(mixer._last_dmax, 3))
        return out

    def forward(self, input_ids):
        x = self.drop(self.embed(input_ids))
        # 按层梯度检查点：反向时重算每层，避免全部层激活同时驻留。
        # 8K 序列下全注意力层的激活是显存大头（3 层 × ~3GB），不开必 OOM。
        use_ckpt = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                if isinstance(blk.mixer, GDN2Layer):
                    x = checkpoint(blk, x, use_reentrant=False,
                                   context_fn=blk.mixer.checkpoint_contexts)
                else:
                    x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)

    def forward_with_sla_entropy(self, input_ids):
        """训练监控专用：返回 logits 与每层头熵（计划书 §5 monitor.head_entropy）。

        头熵只作监控，不作损失项、不作 Go/NoGo 条件（SOURCE_LOCK: SLA 已知边界）。
        """
        x = self.drop(self.embed(input_ids))
        entropies = []
        for blk in self.blocks:
            if blk.kind == "sla":
                res = blk.mixer(blk.norm1(x))
                x = x + res["out"]
                x = x + blk.mlp(blk.norm2(x))
                entropies.append(res["head_entropy"].detach())
            else:
                x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x), entropies
