# SOURCE_LOCK

本文件是 HCL-TR v0.4 的唯一外部依赖事实源。CI 闸门 `scripts/lock_check.py`
会解析本文件，并强制校验：代码中每个外部 `import`、每个 `torch.load` /
`from_pretrained`、每次 RULER/S-NIAH-1 评估，都能映射到这里的条目。
**本文件中的 URL、commit、license、公式页码全部于 2026-10-29 实测核验，未核验项一律标注 `UNVERIFIED` 并冻结对应模块。**

状态图例：
- `LOCKED` — 已实测 URL + commit/revision + license，可用于实现。
- `LOCKED-PAPER` — 论文已实测存在，按公式自实现，无官方代码可复用。
- `UNVERIFIED` — 网络不可达或源不存在，**对应模块冻结**，`train.py` 不得启动。

---

## 模块：GDN-2
- Status: `LOCKED`
- URL: https://github.com/NVlabs/GatedDeltaNet-2
- Commit: `de8476de0fecae08fc16f0bf3154900244c61c73` (branch `main`, pushed 2026-10-11)
- LICENSE: NVIDIA Source Code License-NC (`NOASSERTION` on GitHub; 原文已读)
  - **限制：仅限非商业研究/评估用途（§3.3）。商业使用禁止。**
  - 复用要求：分发须附带完整 license 副本、保留版权声明（§3.1）。
- 接口签名:
  - `GatedDeltaNet2(hidden_size, expand_v=1, head_dim, num_heads, num_v_heads=None, mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-5)`
  - `forward(hidden_states[B,T,H], attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False) -> (o[B,T,H], None, cache)`
  - 算子入口：`lit_gpt.gdn2_ops.chunk_gdn2.chunk_gdn2(q,k,v,g,b,w,A_log,dt_bias,initial_state,output_final_state,use_qk_l2norm_in_kernel,use_gate_in_kernel,cu_seqlens)`
- 公式依据: 论文 `arXiv:2411.12537`（negative eigenvalues）在代码 docstring 引用；GDN-2 状态转移原文见
  `vendor/gdn2/lit_gpt/gdn2.py:57-62`：
  `S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t ⊙ v_t)^T`
  b_t ∈ R^{d_k} 通道级 erase 门，w_t ∈ R^{d_v} 通道级 write 门。
- 依赖: `torch`, `einops`, `fla` (flash-linear-attention, PyPI `flash-linear-attention`), `transformers`
- 复用范围: **状态转移与门构造直接复用官方 `lit_gpt/gdn2.py` + `gdn2_ops/`**。
- 自行实现部分（官方无对应开关，见 §4.1 超出官方实现的要求）:
  - `divide_by_sqrt_t` 序列长度归一化（官方仅有 SiLU-gated RMSNorm）
  - output gate 的 `bias_init=1.0` 配置（官方 `o_norm` 无此初始化口径）
  - `gdn2_internal_qk_norm` 独立消融开关（官方固定 `use_qk_l2norm_in_kernel=True`）
- 依赖外部 revision 校验: vendor 目录 commit 必须等于 `de8476d...`，否则 `lock_check.py` 失败。
- 核验日期: 2026-10-29
- 核验人: (待填)
- 对应 fixture: `tests/unit/test_gdn2_state.py`（公式逐行自实现 vs 官方算子数值对比，fp32 max_abs_err<=1e-5）

---

## 模块：CMM (Continuous Memory Machines)
- Status: `LOCKED`（论文）+ `LOCKED`（底座代码）
- 论文依据: `arXiv:2610.07907` — "Continuous Memory Machines" (2026-10-06, Regan, Arulkumaran, Darlow, Druga, ...)
- 官方代码: **不存在**。`github.com/SakanaAI/continuous-memory-machines` 于 2026-10-29 实测 **HTTP 404**；全 GitHub 搜索无非官方镜像托管。
- 现阶段实现基础（代码底座）:
  - URL: https://github.com/SakanaAI/continuous-thought-machines
  - Commit: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466` (branch `main`, pushed 2025-12-29)
- LICENSE: Apache-2.0（可商用、可修改、须保留声明）—— 底座仓库 SakanaAI/continuous-thought-machines
  - 代码复用状态（**如实记录**，2026-10-29 修正）:
    - **当前未 import CTM 代码**。`src/hcl_tr/models/cmm.py` 是独立实现：per-neuron NLM 用
      grouped einsum 自写（`NeuronLevelModel`），同步机制自写（`Synchronisation`），
      未调用 `vendor/ctm` 的 `SuperLinear` / `ContinuousThoughtMachine`。
    - `vendor/ctm` 作为**参考实现与设计依据**保留（tick 循环结构、per-neuron 参数化思路、
      synchronisation 语义），并作为后续「真正复用审计」的对照物。
    - 计划书 §2.1 要求「能复用官方源码直接复用」。CMM 官方源不存在，故以 CTM 为底座；
      但「复用」目前是**语义对齐而非代码 import**——这是一处已知偏离，记录在此。
      若要转为真正 import 复用，需在 GPU 环境验证 CTM 的 `SuperLinear` 与当前
      `NeuronLevelModel` 数值等价后替换，并用 `test_cmm_memory.py` 回归。
- 需严格按 CMM 论文 `arXiv:2610.07907` 自行实现的部分（PDF 已归档，sha256=`f9a3c24c…09d7`）:
  - 矩阵值短期记忆 STM = 预激活滑动窗口 `A_t ∈ R^{d×M_S}` —— §3.1 eq.(1)(2)，**PDF 第 4 页**
    `a_t = f(z_t, o_t)`（eq.1）；`z_i^{t+1} = g_i(A_i^t)`（eq.2）
  - 长期记忆 LTM 矩阵 `B_t ∈ R^{d×M_L}`，以可学习 `B_0` 初始化 —— §3.2 "Long-Term Memory"，**PDF 第 4 页**
  - 联合更新（即"Transformer 双向读写头"）—— §3.2 eq.(4)(5)(6)，**PDF 第 4–5 页**
    eq.(4) `X^t = [s; B^{t-1}; A^t] ∈ R^{d×(1+M_L+M_S)}`（sink token + LTM + STM 拼接）
    eq.(5) `X̃^t = TransformerBlocks(X^t) = [s̃; B̃^t; Ã^t]`
    eq.(6) `z_i^{t+1} = g_i(Ã_i^t)`（NLM 作用在已与 LTM 交互后的 STM 上）
    状态推进：`B̃^t` 作为持久 LTM 前传；`Ã^t` 产出 `z^{t+1}` 后丢弃；原 `A^t` 作为滚动窗口保留
  - 同步矩阵 `S^{t+1}` 与输出 `y^{t+1}` 沿用 CTM eq.(3)，**PDF 第 4 页**
  - 关键澄清：论文的"Transformer 双向读写"**就是** eq.(4)-(5) 的 self-attention over [sink; LTM; STM]，
    不存在额外的 cross-attention 头。CTM 底座里的 `self.attention`（MultiheadAttention）即复用点。
- 依赖: `torch`, CTM 的 `utils/`、`models/modules.py`
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_cmm_memory.py`（STM/LTM 读写一致性、双向读写头形状与梯度流）

---

## 模块：QED (Query knows what to forget)
- Status: `LOCKED-PAPER`
- 论文: `arXiv:2608.13668` — "The Query Knows What to Forget: A Second Erase Direction for Linear Attention" (2026-08-13, Gupta, Das, Gupta)
- 官方代码: 无（`arXiv` 页面实测 200；未声明官方仓库）
- 实现: 按公式自实现，位置在 GDN-2 erase 步骤内部，作为 `b_t ⊙ k_t` 之外的正交附加分支；方向由 query 导出且与 key 正交；**不改变 GDN-2 主状态更新公式**。
- LICENSE: 论文 arXiv 默认授权（PDF 已归档，sha256=`38632a28…2a16`）
- 公式页码（**逐条已核**）:
  - §3.1 Motivation，PDF **第 3 页**：读出为衰减外积之和 `o_t = S_t^T q_t = Σ_{j≤t}(q_t^T k̃_j)ṽ_j`（eq.5），干扰项沿 q_t
  - §3.1 核心式，PDF **第 3 页 eq.(6)**：
    `a_t = b_t ⊙ k_t + λ_h · P⊥kt(b_t ⊙ q_t)`，其中 `P⊥k(x) = x − k(k^T x)`，`λ_h ≥ 0` 为每头可学习强度
    — 注意 P⊥k 里**没有** `1/‖k‖²` 因子（论文假设 `‖k‖₂ = 1`，见 eq.9-10 上下文）
  - §3.2，PDF **第 3 页 eq.(8)**：GDN-2 更新 `S_t = S̄_t + k_t[w_t ⊙ v_t − S̄_t^T(b_t ⊙ k_t)]^T`
  - §3.2 精确恒等式，PDF **第 4 页 eq.(11)(12)**：
    `S_t^QED = S_t^GDN2 − λ_h k_t(S̄_t^T d_t)^T`
    `o_t^QED = o_t^GDN2 − λ_h ρ_t · S̄_t^T d_t`，其中 `d_t = P⊥kt(b_t ⊙ q_t)`，`ρ_t = q_t^T k_t`
  - 正交性证据：eq.(6) 后文"keeps the new component of the erase vector key-orthogonal and leaves the
    nontrivial eigenvalue of `I − k_t a_t^T` unchanged"
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_qed_erase.py`（正交性 <1e-6；query 方向旧状态抵消）

---

## 模块：SLA
- Status: `LOCKED-PAPER`（**已解冻**，2026-10-29 按用户确认来源）
- 论文: `arXiv:2602.01744` — "Softmax Linear Attention: Reclaiming Global Competition" (2026-02-02, Xu, Lin, Guo, ...)
  PDF 归档 `vendor/papers/arXiv-2602.01744-SLA.pdf`，sha256=`87ba301c…211a`（11 页）
- **缩写陷阱（已排除）**: thu-ml 的 SLA = "Beyond Sparsity in Diffusion Transformers via Fine-Tunable
  Sparse-Linear Attention"（arXiv:2509.24006, 2025-09-28），面向扩散模型，**与本模块无关**。
  `2509.24006` 实测标题已核，确认不是同一机制。
- 官方代码: 无（论文未附代码链接）
- **机制（按论文，非计划书原 §4.3）**:
  - 门控位置在**线性注意力的 Q/K head 投影**上，而非输出头聚合
  - §4.1，PDF p.4: `G_Q = softmax_h((Q W_GQ)_h)`，`G_K = softmax_h((K W_GK)_h)`
  - Eq.(5)(6)(7)，PDF p.3-4:
    `S_{h,t} = S_{h,t-1} + (G_K_{h,t} ⊙ φ(k_{h,t}))^T v_{h,t}`
    `y_{h,t} = (G_Q_{h,t} ⊙ φ(q_{h,t})) S_{h,t}`
    `o_t = Concat_{h=1..H}[y_{h,t}] W_O`
  - Eq.(8)，PDF p.5，Theorem 4.2（magnitude sensitivity）:
    `lim_{λ→∞} G_Q(λ s) = one-hot(argmax_h s_h)` —— winner-take-all 极限
  - Eq.(9)，PDF p.5，Theorem 4.3（Kronecker delta 路由）:
    `C(λ) = Σ_h G_Qh(λ s_Q) G_Kh(λ s_K)`，`lim_{λ→∞} C(λ) = δ(argmax_h s_Qh, argmax_h s_Kh)`
  - λ 是 query/key 的**缩放强度**，不是温度退火参数
- **已删除（论文无此内容）**: 计划书原 §4.3 的 `α_h = softmax(s_h/t_h)` 输出头竞争、
  温度退火 1.5→0.5、熵正则 `θ_init=0.5/θ_range=[0.3,0.8]/γ=0.1/λ=1e-3`、
  `H_head<0.05` 塌缩阈值作为 Go/NoGo 条件。**这些参数无论文支撑，不写入实现。**
- **已知边界（诚实标注）**: λ 的调度策略论文未明确。实现采用固定值（未指定则 1.0）；
  若训练中出现门控过早 one-hot，记录为已知风险，**不自行发明退火策略**。
  头熵保留为**监控指标**（论文 §4.1 讨论了 entropy 与 magnitude sensitivity），
  但**不作损失项、不作 Go/NoGo 硬条件**。
- **接入方式（2026-10-30 修正）**: 计划书 §9 原文是「在 +GDN/QED 上加 SLA」，因此 SLA
  **叠加在 GDN-2 层上**（G_Q/G_K 乘在 l2norm 之后的 q/k），而不是替换 GDN-2 层。
  参数量 102.63M vs +GDN/QED 102.47M（只多门控参数），归因不被参数混淆。
  **门控必须在 l2norm 之后**——l2norm 尺度不变，先乘会被归一化约掉。
- 已知限制: `sla.py` 的 SoftmaxLinearAttention 用 cumsum 存 [B,T,H,dh,dh]，8K 下约 29GB，
  独立层版本不可用于 8K；叠加版走 GDN-2 的 chunkwise，无此问题。
- 依赖: `torch`
- 核验日期: 2026-10-29（机制）/ 2026-10-30（接入方式）
- 对应 fixture: `tests/unit/test_sla_gating.py`、`test_backbone.py::test_sla_stacks_on_gdn2_and_keeps_qed`
  （G_Q/G_K softmax 归一化、λ→∞ one-hot 极限 Eq.(8)、Kronecker delta 路由 Eq.(9)、头熵监控日志）

---

## 模块：SPAR (Selective prefix anti-interference)
- Status: `LOCKED-PAPER`
- 论文: `arXiv:2609.27925` — "Learning When Not to Listen: Selective Anti-Interference Pretraining for Language Models" (2026-08-22, Zhu, He, Ding, Fu, ...)
- 官方代码: 无
- 实现: 按公式自实现；门控标签来源预注册二选一（held-out probe / teacher-forced suffix likelihood）
- 公式页码（**逐条已核**，PDF 已归档 sha256=`8589d1b1…81d6`）:
  - 总目标，PDF **第 2 页** Figure 1 下的 `L = L_CLM + λ_inv · L_inv`
  - 两项机制，PDF **第 2 页** Figure 1 说明：
    - "short-context sufficiency gate selects suffix tokens with **high short-context likelihood** and a
      **small full–short likelihood gain**"
    - "the resulting mask restricts **clean-to-corrupt-prefix KL** matching while CLM remains active throughout"
  - corrupt-prefix 构造，PDF **第 2 页**：仅 shuffle far prefix 内的 chunk，local suffix 逐 token 保留
  - 门控标签定义（"sufficiency"）见 §3 正文（PDF 第 2–3 页）；具体阈值与 `L_suffix` 未在正文给常数 →
    按计划书 §4.8 预注册二选一实现，阈值固定不调
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_spar_gate.py`（门控 KL；false-gate/missed-gate 比例 >5% 冻结）

---

## 模块：MoE-nD
- Status: `LOCKED-PAPER`
- 论文: `arXiv:2604.17695` — "MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression" (2026-04-20, Sun, He, Harn, Qin)
- 官方代码: 无
- 实现: 按公式自实现；**仅部署期验收，不参与训练期 S-NIAH-1 归因**
- 路由表 schema: 见计划书 §4.6
- 公式页码（**逐条已核**，PDF 已归档 sha256=`82f2f1a8…abea`）:
  - 方法总述，PDF **第 2 页** "Contribution"：每个压缩轴（eviction ratio、K-precision、V-precision）作为 expert，
    离线校准 router 在单一全局 memory budget 下逐层选择混合
  - **core lever**，PDF **第 2 页**：per-layer **eviction** routing 是新颖增益来源；
    per-layer **quantization** routing 单独加约 0 pts（消融 2d uniform → 2d → 2d hetero）
  - router 算法 = "greedy solver over an offline-measured per-layer sensitivity table"（PDF 第 2 页）
  - 校准数据：LongBench-v1 的 4-task 子集，16k 输入（PDF 第 2 页）；主模型 DeepSeek-R1-Distill-Qwen-7B，
    仅评 `</think>` 之后文本、max_gen×16（PDF 第 2 页）
  - 具体算法伪代码见 §3（PDF 第 3–4 页）；实现时以 `§3 + 附录` 为准
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_moe_nd_router.py`（贪心校准小样本 + schema）

---

## 模块：DASC
- Status: `UNVERIFIED` — **模块冻结，部署验收 NoGo**（2026-10-29 二次检索后维持）
- 论文: `arXiv:2608.30386` — "DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving"
  (2026-08-31, Yu, Sun, Tan, ...) — PDF 归档 sha256=`7ecb14ff…ef61`
- 论文中的实现声明（PDF p.2, p.6）: "We design and implement **DASC in SGLang**"，
  实验用 TP8、BF16 unquantized，recurrent state 为 FP32/BF16 mixed precision。
  **但论文未给出任何 PR 编号、release tag 或代码 URL。**
- **2026-10-29 检索记录（全部实测，经可用网络出口）**:
  | 检索式 | 结果 |
  |---|---|
  | `repo:sgl-project/sglang dasc` | total_count = **0** |
  | `repo:sgl-project/sglang decay_aware` | total_count = **0** |
  | `repo:sgl-project/sglang decay-aware` | 18 条，**全部无关**（KV-split cap、radix eviction、speculative decoding 等） |
  | `repo:sgl-project/sglang ragged_state` | total_count = **0** |
  | sglang 1917 个 head 分支名含 dasc | **无** |
- 论文覆盖范围（比原判断宽，记录备查）:
  - Kimi-Linear（channel-wise **KDA**）与 **Qwen3-Next（head-wise GDN）** 均验证
  - 因此 DASC 的目标**包含 GDN 类 recurrent state**，与计划书 §4.5「GDN-2/CMM recurrent checkpoint」方向一致
  - 但依赖 SGLang 的 `RadixAttention` 前缀缓存与默认 mixed-precision recurrent state，属**推理服务框架内实现**
- 结论: 论文声明存在实现，但**公开信息无法定位到可锁定的 commit/API**。
  按计划书 §0 红线：**不得现场发明压缩格式；冻结模块并 NoGo。**
- 解冻条件（二选一）:
  1. 提供 SGLang 中包含 DASC 的精确 tag 或 PR merge commit SHA；
  2. 或论文作者公开代码仓库。
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_dasc_roundtrip.py`（**contract 测试通过，数值测试 skip**）

---

## 模块：When2Think
- Status: `LOCKED`（**已解冻**，2026-10-29 按用户确认来源 + HF 实测核验）
- 论文依据: `arXiv:2609.19671v2` — "When2Think: Learning When and How Much to Reason"
  (2026-09-17 v1 / v2, Shim Jaejun, Kim HyunJin, Kim Young Jin, Bak JinYeong)
  PDF 归档 `vendor/papers/arXiv-2609.19671-When2Think.pdf`，sha256=`c2f4ff4a…45d2`（37 页）
- **标题版本说明**: v1 标题为 "Learning Difficulty-Aware Length Control for Efficient Hybrid
  Reasoning Models"；v2 改为 "Learning When and How Much to Reason"。**同一篇论文**，
  作者一致。本锁按 **v2** 取公式（编号在 v2 稳定）。
- HF 权重（实测 via `huggingface.co/api/models`，2026-10-29）:
  - `junshim/When2Think-1.5B` @ **`69547b99750d1cb82d75ea82c3afb48454a75627`**
    (lastModified 2026-10-07, license MIT, base_model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`)
  - 消融对照: `junshim/When2Think-ThinkOnly-1.5B` @ **`ebf78883335ba11c8c9821fb84a79422470f5e30`**
    （对应论文 "w/o IS（仅 IDAC+BWS）" 变体；供 §9 "+When2Think only" 诊断组）
  - **权重仅作接口行为参考，不作公式依据**（计划书 §2.2 要求分两行）
- 公式依据（**逐条已核**，v2 PDF）:
  - **IS / importance ratio — Eq.(1)，PDF p.3**:
    `ρ_{i,k} = π_θ(o_{i,k}|x_i) / π_IS(o_{i,k}|x_i)`；
    π_IS 以**等概率**选择初始 mode token（THINK/NOTHINK）强制均衡探索，
    后续 token 由 π_θ 生成。推理时移除 π_IS。
  - **PPO clipped surrogate — Eq.(2)，PDF p.3**:
    `L = -E[min(ρ A, clip(ρ,1-ε,1+ε) A)]`
    **"IS 越界停训"的红线落在 Eq.(2) 的 clip 语义上**：ρ 偏离 1 超过 ε 时 min 取被裁剪的一支，
    梯度被截断——这是隐式约束，不是显式 clipping 操作。fixture 须按 clipped surrogate 的实际行为写。
  - **IDAC 缩放因子 — Eq.(5)，PDF p.5**:
    `λ̃_{i,k} = exp(-T(o_{i,k})·α_i / τ_i)`；`λ_{i,k} = λ̃ if Think else 1`
    `α_i` = 参考策略在实例 i 上的经验成功率（难度代理，越大越简单）；
    `τ_i` = 平均参考 token 数（成本尺度）；`T(o)` = 轨迹 token 数。
    边界：Think 奖励随 token 单调不增；α_i 越高衰减越快；NoThink 的 λ=1 不受 token 数影响。
  - **最终实例自适应奖励 — Eq.(6)，PDF p.5**:
    `r_{i,k} = V(x_i,o_{i,k})·(1 + λ_{i,k}·δ) - α_i`
    `V ∈ {0,1}` 为 verifier 正确性；`δ>0` 控制效率 bonus 量级；`α_i` 作参考基线归一化。
    **错误回答时 V=0 → r = -α_i**（注意：不是 0；fixture 需按论文，不按计划书"错误=0"的简化）。
  - **BWS 批次标准化 — Eq.(7)(8)，PDF p.5**:
    `μ_k = (1/M)Σ_i r_{i,k}`，`σ_k = sqrt((1/M)Σ_i (r_{i,k}-μ_k)^2)`
    `A_{i,k} = (r_{i,k} - μ_k) / (σ_k + ε)`
    **关键：μ_k、σ_k 是按 trajectory index k 跨 M 个 instance 求的**（不是整个 mini-batch 拉平），
    同一轨迹内所有 token 共享同一 advantage。
- 依赖: `torch`, `transformers`（加载参考权重时须 `revision=` 上述 sha）
- 核验日期: 2026-10-29
- 对应 fixture: `tests/unit/test_idac_reward.py`
  （Eq.(5) 单调性/α_i 依赖、Eq.(6) 错误项 = -α_i、Eq.(7)(8) 跨 instance 标准化、
   Eq.(2) clip 越界梯度截断）

---

## 模块：S-NIAH-1 / RULER
- Status: `LOCKED`（基准定义）+ 预注册 YAML 待生成
- 基准: RULER — https://github.com/NVIDIA/RULER（探针评估脚本，公开基准）
- 评估口径: `configs/sniah1_ruler_prereg@1.0.yaml`（预注册，见 §8.1）
- 说明: 训练期评估必须读取预注册 YAML；禁止运行时改评测长度/重定义可用长度
- 核验日期: 2026-10-29

---

## 论文 PDF 归档（审计线索）

以下 PDF 已下载并归档至 `vendor/papers/`，为公式页码的唯一事实来源：
| 模块 | 文件 | 页数 | sha256（前 8） |
|---|---|---|---|
| CMM | `arXiv-2610.07907-CMM.pdf` | 37 | `f9a3c24c` |
| QED | `arXiv-2608.13668-QED.pdf` | 10 | `38632a28` |
| SPAR | `arXiv-2609.27925-SPAR.pdf` | 9 | `8589d1b1` |
| MoE-nD | `arXiv-2604.17695-MoE-nD.pdf` | 9 | `82f2f1a8` |
| SLA | `arXiv-2602.01744-SLA.pdf` | 11 | `87ba301c` |
| When2Think | `arXiv-2609.19671-When2Think.pdf` | 37 | `c2f4ff4a` |
| DASC | `arXiv-2608.30386-DASC.pdf` | — | `7ecb14ff` |

## 公式页码状态

| 模块 | arXiv ID | 状态 | 关键页码 |
|---|---|---|---|
| CMM | 2610.07907 | ✅ 已提取 | p.4 eq.(1)(2)(3)；p.4–5 eq.(4)(5)(6) |
| QED | 2608.13668 | ✅ 已提取 | p.3 eq.(6)(8)；p.4 eq.(11)(12) |
| SPAR | 2609.27925 | ✅ 已提取 | p.2 `L = L_CLM + λ_inv L_inv` + Figure 1 门控定义 |
| MoE-nD | 2604.17695 | ✅ 已提取 | p.2 方法总述 + greedy router 定义；§3 伪代码 |
| SLA | 2602.01744 | ✅ 已提取（2026-10-29 解冻） | p.4 §4.1 G_Q/G_K；p.3-4 eq.(5)(6)(7)；p.5 eq.(8)(9) |
| When2Think | 2609.19671v2 | ✅ 已提取（2026-10-29 解冻） | p.3 eq.(1)(2)；p.5 eq.(5)(6)(7)(8) |
| DASC | 2608.30386 | ⚠️ 论文有公式，但**代码不可锁** | 论文声明"implemented in SGLang"，无 PR/tag |

**唯一剩余 BLOCKER**: DASC。论文公式可读，但公开信息无法定位 SGLang 中的实现 commit。
按 §0 红线，**不自行发明压缩格式**，模块冻结至提供精确 tag/PR SHA 或作者公开代码。

**已解除的 BLOCKER（2026-10-29）**:
- SLA：论文 `2602.01744` 经用户确认，并与 thu-ml 的同名 `2509.24006` 明确区分。
  计划书原 §4.3 的温度退火/熵正则无论文支撑，已删除，改为论文的 G_Q/G_K 机制。
- When2Think：论文 `2609.19671v2` 公式已逐条核实；HF 权重 sha 已实测锁定。

## CI 闸门规则（`scripts/lock_check.py` 实现）

1. 每个外部 `import`（`fla`, `transformers`, CTM vendor 等）必须在本文件有条目。
2. 每个 `torch.load` / `from_pretrained` 必须携带 `revision=` 且等于本文件记录值。
3. 每个 RULER/S-NIAH-1 评估入口必须加载预注册 YAML。
4. vendor 目录 commit 必须等于本文件 `LOCKED` 值，漂移即失败。
5. `UNVERIFIED` 模块被 import 即失败（冻结）。
6. 任一检查失败 → `train.py` 不启动。

---

## 协议偏离记录（2026-10-30）

以下偏离计划书原文，**经用户明确决策**，必须随任何结果一并披露：

| 项 | 计划书原文 | 实际 | 原因 |
|---|---|---|---|
| Tokenizer | TinyLlama v1.1（32k，英文导向） | **自训 64k 中英双语 BPE** | 用户要求中英双语；TinyLlama 下中文约 2–3 token/字，效率过低 |
| 语言 mix | 英文 FineWeb-Edu | **英 85% / 中 15%**（C4 + FineWeb-Edu + chinese-fineweb-edu-v2） | 同上 |
| d_model | 768 | **640**（n_head=10，head_dim=64） | 64k 词表下守住 ~100M（实测 95.06M）；768 会到 ~128M |
| 参数量 | ~100M | **95.06M 混合骨干 / 93.41M Base-T** | 同上 |
| 数据源 | HuggingFace 流式 | **Kaggle 数据集挂载**（C4 19.3GB + FW-Edu 21GB） | 免下载、版本可 pin、便于算 data_mix_hash |

**连带影响**：baseline_100m_transformer 的旧口径（TinyLlama/768/英文）**全部作废**，必须按新口径重训重冻；
Go/NoGo 的「相对 baseline ≥80%」需在新 baseline 上重新建立。

**未偏离**：S-NIAH-1 主指标、长度网格、Wilson CI、可用长度判定规则、global_batch_tokens=262144、
seq_len=8192、grad_clip=1.0、warmup=1e9、各归因组除机制外训练配置一致。
