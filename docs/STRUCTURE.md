# HCL-TR v0.4 目录结构与类名

## 顶层

```
SOURCE_LOCK.md              外部依赖唯一事实源（CI 闸门校验）
train.py                    训练入口（启动前强制 SOURCE_LOCK 闸门 + 阶段冻结检查）
pytest.ini                  testpaths=src pythonpath；markers: cpu/gpu
requirements.txt            GPU 训练依赖
requirements-dev.txt        CPU fixture 依赖
.pre-commit-config.yaml     pre-commit / pre-push 闸门
.github/workflows/gates.yml CI：source-lock → unit-fixtures-cpu
```

## scripts/

| 脚本 | 作用 |
|---|---|
| `lock_check.py` | SOURCE_LOCK 闸门：import 映射 / revision / 冻结 / vendor commit 漂移 |
| `check_load_revision.py` | 禁止无 revision 的 `from_pretrained` / 无 sha256 的 `torch.load` |
| `freeze_prereg.py` | 冻结预注册 YAML 与 baseline 三项哈希（防覆盖） |
| `fixture_report.py` | 生成 `docs/fixture_report.json`（dtype/kernel/容差） |

## src/hcl_tr/models/

| 文件 | 类 / 函数 | SOURCE_LOCK 状态 |
|---|---|---|
| `gdn2_ref.py` | `gdn2_step`, `gdn2_chunk`, `gdn2_output` | GDN-2 LOCKED |
| `qed.py` | `project_out_key`, `qed_erase_direction`, `gdn2_step_with_qed`, `qed_output_identity_err` | QED LOCKED-PAPER |
| `sla.py` | `SoftmaxLinearAttention`, `head_gates`, `information_flow_coefficient`, `kronecker_routing` | SLA LOCKED-PAPER（arXiv:2602.01744，G_Q/G_K 门控） |
| `spar.py` | `SPARGate`, `spar_loss`, `gate_audit` | SPAR LOCKED-PAPER |
| `cmm.py` | `NeuronLevelModel`, `Synchronisation`, `CMMMemory` | CMM LOCKED |
| `idac.py` | `IDACReward`, `bws_standardize`, `ppo_clipped_objective`, `ISMonitor` | When2Think LOCKED（arXiv:2609.19671v2） |
| `moe_nd.py` | `RoutingEntry`, `greedy_calibrate`, `apply_atomically` | MoE-nD LOCKED-PAPER（仅部署） |
| `dasc.py` | `DASC`, `DASCNotAvailableError`, `CompressionReport` | DASC **UNVERIFIED（冻结）** |
| `backbone.py` | `RMSNorm`, `RotaryEmbedding`, `FullAttention`, `GDN2Layer`, `HybridBlock`, `HybridBackbone` | 组合层 |

## src/hcl_tr/eval/

| 文件 | 函数 | 作用 |
|---|---|---|
| `gonogo.py` | `usable_context_length`, `check_gonogo`, `build_attribution_json`, `diagnose` | 自动 NoGo 三条件 + 归因 |

## src/hcl_tr/configs/

| 文件 | 内容 |
|---|---|
| `sniah1_ruler_prereg_v1.0.yaml` | 预注册评估口径（已冻结 hash `a1b30654…`） |
| `baseline_100m_transformer.json` | 冻结 baseline（checkpoint/tokenizer/data hash 待实测填入） |
| `stage0_mvp1a.yaml` | MVP-1a：3:1 混合骨干，SLA/QED 关闭 |
| `stage1_cmm.yaml` | CMM 中间训练配置 |

## tests/unit/（全部 CPU 可跑）

| 文件 | fixture | 数量 |
|---|---|---|
| `test_gdn2_state.py` | GDN-2 状态转移（闭式对照/衰减/外积/分块一致） | 4 + 1 gpu-skip |
| `test_qed_erase.py` | 正交性 / 特征值不变 / eq.(12) 恒等式 / 擦除有效 | 6 |
| `test_sla_gating.py` | G_Q/G_K 归一化 / λ→∞ one-hot (Eq.8) / Kronecker 路由 (Eq.9) / 因果性 | 8 |
| `test_spar_gate.py` | 门控 / KL / false-missed 比例 | 8 |
| `test_cmm_memory.py` | eq.4-6 token 数 / LTM 持久 / 梯度 / 对称同步 | 9 |
| `test_idac_reward.py` | Eq.(5) λ 单调 / Eq.(6) 错误=-α / Eq.(7)(8) 按k分组 / Eq.(2) clip | 20 |
| `test_moe_nd_router.py` | schema / 贪心预算 / 原子回滚 | 7 |
| `test_dasc_roundtrip.py` | 冻结契约（数值测试 skip） | 4 + 1 skip |
| `test_backbone.py` | 3:1 比例 / QK-norm / 消融开关 / SLA 替换 GDN-2 | 11 |
| `test_gonogo.py` | 三条件 / Wilson / 归因 JSON | 11 |
| `test_lock_check.py` | 闸门自测（防假绿） | 5 |

## vendor/（锁定源码）

| 目录 | 仓库 | commit | license |
|---|---|---|---|
| `vendor/gdn2` | NVlabs/GatedDeltaNet-2 | `de8476de…` | NVIDIA Source Code License-NC |
| `vendor/ctm` | SakanaAI/continuous-thought-machines | `4a6c9c3a…` | Apache-2.0 |
| `vendor/papers/` | 七篇 arXiv PDF（含 SLA/When2Think/DASC） | — | arXiv 默认 |

## Exit Criteria 映射

| 条件 | 状态 |
|---|---|
| CI 全绿 | ✅ lock_check + 102 unit tests |
| SOURCE_LOCK 无缺口 | ⚠️ 仅 DASC 为 BLOCKER（论文有公式但 SGLang 无可锁 commit） |
| 预注册 YAML | ✅ hash `a1b30654…` |
| baseline checkpoint 冻结 | ⚠️ 需实跑后填 checkpoint_sha256 |
| 6 模块 fixture 按 dtype 记录 | ✅ `docs/fixture_report.json` |
| DASC/MoE-nD 仅部署验收 | ✅ DASC 冻结（contract 测通过，数值 skip）；MoE-nD 仅部署 |
| 主 4 组 + 诊断 2 组配置 | ✅ `configs/attribution/` 6 组已生成（见 §9 映射） |
| NoGo 白名单生效 | ✅ |
| 外部 revision 漂移冻结 | ✅ lock_check vendor commit 校验 |
