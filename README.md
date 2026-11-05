# HCL-TR Ops

Hybrid CMM-Linear with Learned Rethink Triggering —— **混合骨干（3:1 GDN-2 + 全注意力）+ 可训练流程 + 算子基准工具**。

这个仓库的重点不是论文复现的完整清单，而是三件能直接跑起来的事：

1. **一个可训练的混合骨干**：3 层 GDN-2 chunkwise 线性层 + 1 层全注意力，fp16 AMP、梯度累积、DDP、断点续训齐全。
2. **一套可复现的算子基准**：GDN-2 chunkwise 的逐阶段计时、RoPE 融合 kernel 对照、TileLang 可行性探测。
3. **一组硬门禁**：`SOURCE_LOCK` 校验、数值 fixture、失败即报错（不静默回退、不把 skip 当通过）。

## 已验证的事实（T4 / SM75 实测）

| 项目 | 结果 |
|---|---|
| GDN-2 chunkwise 与独立 fp64 实现的最大误差 | `1.55e-07` |
| 现有批量实现（B=1,H=10,dk=dv=64）2K / 8K 单次前向 | 20.5 ms / 105.0 ms（CUDA event） |
| 融合后端（中点平移，chunk=86）2K / 8K | 5.6 ms / 19.1 ms（**3.6× / 5.6×**） |
| 融合后端对 fp64 真值误差（chunk 21/43/86） | `7.6e-08` / `1.3e-07` / `2.1e-07` |
| 融合后端梯度对准确基准的最大误差 | `7.2e-07`（chunk 86，全部输入） |
| autocast 下的融合算子 | **必须显式关闭**：fp16 会把 `e^80` 级因子溢出成 NaN（已修复，有 CUDA 回归测试） |
| 融合 RoPE（Triton）数值 | 输出与输入梯度均通过（fp32 容差 1e-5，fp16 容差 5e-3） |
| 融合 RoPE 端到端收益 | **没有**：2K AMP 12.35 ms vs torch 6.28 ms |
| TileLang 0.1.15 在 SM75 | 编译并运行成功 |
| FlashQLA | **不支持 T4**：仅 sm90/100/103/120/121 |

融合后端的三处关键改动：A/P 合并为一次 GEMM；UT 变换把递推循环中每 chunk 的
GEMM 从 4 个降到 3 个；按块内中点平移指数因子，使 `exp(±(G−c))` 的幅度上界从
`e^{chunk×dmax}` 降到 `e^{chunk×dmax/2}`，把安全 chunk 从 21 提到 86。

上述"没有收益"和"不支持"都是实测结论，仓库不把它们包装成优点。

## 安装

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# 只跑测试（CPU 即可）
.venv/bin/pip install -r requirements-dev.txt
```

`flash-linear-attention` 是可选依赖；不安装时 GDN-2 走仓库内的纯 PyTorch chunkwise 实现。

## 快速开始（CPU，只跑数值 fixture）

```bash
python scripts/lock_check.py
PYTHONPATH=src .venv/bin/python -m pytest tests -q
.venv/bin/python scripts/fixture_report.py
python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml --dry-run
```

## 训练

### 1. 准备数据分片

训练消费的是已经分词好的 `shard_*.npy`（`uint16`）+ 同名 `.ready` 戳记：

```bash
python -m hcl_tr.data.prefetch \
  --cache-dir /data/shards \
  --dataset HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --max-shards 60
```

这个进程可以持续往目录里追加分片，训练侧边写边读。

### 2. 启动训练

```bash
# 单卡
python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml \
  --data-dir /data/shards --out-dir out/run1

# 双卡 DDP（自动 spawn 两个进程）
python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml \
  --data-dir /data/shards --out-dir out/run1 --n-gpu 2

# 断点续训（必须提供 checkpoint 的 SHA256，防止静默加载错权重）
python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml \
  --data-dir /data/shards --out-dir out/run1 \
  --resume-from out/run1/ckpt_00000000100000000.pt --ckpt-sha256 <sha256>
```

T4 没有 bf16 硬件支持，训练固定走 fp16 + GradScaler。8K 序列、`micro_batch=2` 是 16GB 卡的起点；OOM 就降 micro batch，用 `grad_accum` 补回 global batch。

### 3. 归因组配置

```bash
python scripts/gen_attribution_groups.py   # 生成 configs/attribution/*.yaml
```

## 算子基准

```bash
# GDN-2 chunkwise：自定义 chunk/窗口，逐例 JSON，带精度门禁
python scripts/benchmark_gdn2.py --mode profile --output-dir results/profile \
  --profile-backends serial batched --profile-steps 1

# 需要更细的 trace 时才开（体积会大很多）
python scripts/benchmark_gdn2.py --mode profile --output-dir results/profile \
  --profile-shapes --profile-memory
```

计时来自 CUDA event，精度检查与计时分离。`profile` 只验证完整层数值有限，不是吞吐验收；判吞吐要看独立的重复计时。

## 已知限制

- **只在 T4（SM75）上实测**。FlashQLA 直接拒绝 SM75，不能用；TileLang 可用但完整 GDN kernel 尚未移植。
- **融合 RoPE 目前是负收益**，保持 opt-in，不做默认路径。
- 训练入口要求真实 GPU；CPU 上只能跑 fixture 和 dry-run。
- `vendor/` 下的第三方源码**不在本仓库内**（见下）。

## 第三方代码与许可证

本仓库不包含任何第三方 vendored 源码。`SOURCE_LOCK.md` 记录了各模块的外部来源、commit 与许可证，供审计使用。

特别说明：**NVlabs/GatedDeltaNet-2 采用 NVIDIA Source Code License-NC（仅限研究用途），不可随本仓库再分发**。需要对照官方实现时请自行按其许可证获取。`scripts/lock_check.py` 在 `vendor/` 目录不存在时会跳过 vendor commit 校验，因此本仓库开箱即可通过门禁。

## 许可证

[Blue Oak Model License 1.0.0](LICENSE)。
