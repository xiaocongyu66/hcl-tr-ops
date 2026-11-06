# HCL-TR v0.4 — Hybrid CMM-Linear with Learned Rethink Triggering

按《HCL-TR v0.4 执行计划书》落地的工程实现。核心原则：**不降级、不近似**，每个红灯指向某个 commit、某条公式页码、某个 fixture 或某份预注册 YAML。

## 快速开始（CPU，只跑数值 fixture）

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install pytest numpy PyYAML

python scripts/lock_check.py                      # SOURCE_LOCK 闸门
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q
.venv/bin/python scripts/fixture_report.py        # 生成 docs/fixture_report.json
python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml --dry-run
```

## 目录

见 `docs/STRUCTURE.md`。

## 依赖状态（SOURCE_LOCK.md 为唯一事实源）

| 模块 | 状态 | 说明 |
|---|---|---|
| GDN-2 | LOCKED | NVlabs/GatedDeltaNet-2 @ `de8476de`，**NVIDIA NC license（仅研究）** |
| CMM | LOCKED | 论文 arXiv:2610.07907 + 底座 SakanaAI/continuous-thought-machines @ `4a6c9c3a`（Apache-2.0） |
| QED | LOCKED-PAPER | arXiv:2608.13668，eq.(6)(11)(12) 页码已核 |
| SPAR | LOCKED-PAPER | arXiv:2609.27925，`L = L_CLM + λ_inv L_inv` |
| MoE-nD | LOCKED-PAPER | arXiv:2604.17695，仅部署验收 |
| SLA | LOCKED-PAPER | arXiv:2602.01744，G_Q/G_K head 门控（非计划书原温度退火版本） |
| When2Think | LOCKED | arXiv:2609.19671v2；HF `69547b99…` / 消融 `ebf78883…` |
| DASC | **UNVERIFIED（冻结）** | 论文 arXiv:2608.30386 声明实现于 SGLang，但无 PR/tag 可锁 |

## 开 10B 主训前的硬门槛

`train.py` 启动即跑 `scripts/lock_check.py`；任一缺口即退出。UNVERIFIED 模块所在的训练阶段被硬拦截
（当前 `--stage deploy` 因 DASC 冻结而拒绝启动，exit 4）。**目前只有 `--stage 0` 有实际实现**，
其余阶段直接 exit 6，避免静默训练同一个骨干。

## 近期验证状态

- **Brahma + Neon 数据集训练（2026-11-06）是管线验证，不是 stage 0**：296M token 语料、约 33M 参数、2×T4；跑到 step 140 因每 token 衰减失控（dmax 6→42）导致 chunk 塌缩与 loss 跳变而中止。
- 参数口径以 `--dry-run` 输出为准：`params.total / embed / non_embed / tokens_per_non_embed_param`。
- 数据摄取默认走「规范化 → 精确去重 → 长度过滤 → MinHash → 全局打散」，并支持 holdout 验证集、按权重混合与去污染（`scripts/ingest_text.py`）。

## 未完成项（需外部输入，不可自行编造）

1. **DASC 的 SGLang tag / PR merge commit** —— 已检索 `dasc`/`decay_aware`/`ragged_state` 全部 0 命中；需精确 SHA 才能解冻。
2. **baseline_100m.pt 实际 checkpoint 与 tokenizer 文件** —— 填写三个 hash 后才能冻结 baseline、计算 Go/NoGo。
3. **远端 GPU 环境** —— 本机无 GPU、磁盘 4.8G。
