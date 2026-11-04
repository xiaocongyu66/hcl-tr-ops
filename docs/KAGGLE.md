# Kaggle 双卡 T4 训练指南

## 环境实测（2026-10-29）

| 项 | 值 |
|---|---|
| GPU | Tesla T4 × 2，各 14.6GB，**sm_75** |
| torch | 2.11.0+cu128（Kaggle 预装） |
| 混合精度 | **必须 fp16 + GradScaler**（T4 无 bf16 硬件支持） |
| 模型 | 102.47M 参数 / non-embed 77.89M（符合计划书 §3 口径） |
| 初始 loss | ≈10.5（≈ln(32000)），已修掉曾出现的 628 爆炸 |
| HF 连通 | ✅ 可达（fineweb-edu 流式加载正常） |
| 首分片就绪 | ~27s |

## 训练配置（双卡）

```
micro_batch_per_gpu = 2      # 8K 序列下全注意力 attn 矩阵 B×H×T×T×2B，mb=4 会 OOM
grad_accum          = 8      # 2×8×8192×2 = 262144 global tokens
world_size          = 2      # DDP，每卡独立进程
```

单卡会浪费一半算力——`HybridBackbone` 已支持 `train_ddp()` 用 `torch.multiprocessing.spawn` 起两进程，
`ShardStream(rank=r, world_size=2)` 按 `index % world_size` 分片，两卡样本不重复。

## 踩过的坑（都已修）

1. **`enable_gpu: true` 不等于双卡**：CLI 推送后默认可能是单卡。必须在网页端
   Settings → Accelerator 选 **GPU T4 x2**。notebook 里有 `assert n>=2` 会硬失败提示。
2. **dataset 挂载路径不可靠**：CLI 的 `dataset_sources` 绑不上时 `/kaggle/input/<slug>` 不存在。
   最终改为**自包含 notebook**（base64 内嵌代码包，运行时解压），不依赖 dataset。
3. **输出下载会截断**：`kaggle kernels output` 对大文件（如 final.pt）常下成 0 字节，
   诊断时看不到日志。用 `--file-pattern ".*run\.log$"` 只拉小日志文件才可靠。
   notebook 因此把所有阶段状态写进 `/kaggle/working/run.log`。
4. **初始 loss 628**：`nn.Embedding` 默认 `N(0,1)` + RMSNorm 残差 → logits 爆炸。
   已改 GPT-2 风格 std=0.02 + 残差投影 1/sqrt(2·n_layer)。
5. **`self.apply(_init_weights)` 覆盖 out_gate bias**：计划书要求 `bias_init=1.0`，
   初始化后需显式恢复（已在代码里处理，并有测试锁定）。

## 运行

```bash
# 推送（本地）
kaggle kernels push -p /tmp/kaggle_pkg

# 监控
kaggle kernels status congyu6666/hcl-tr-v04-short-run

# 只拉日志（避免大文件截断）
kaggle kernels output congyu6666/hcl-tr-v04-short-run -p /tmp/out \
  --file-pattern ".*run\.log$"
```

## 数据策略（按用户要求：用到哪下到哪，删旧的）

`ShardStream(delete_consumed=True, max_cache_bytes=8GB)`：消费完的分片立即删除，
缓存超过 8GB 时删最老分片。分片 20M tokens ≈ 40MB，55 片 ≈ 1.1B tokens。

## 吞吐与显存（实测 + 修正记录）

| 问题 | 现象 | 修复 |
|---|---|---|
| GDN-2 逐 token 循环的 autograd 图 | 8K 序列 OOM（每步保留中间量 ≈12GB） | `gdn2_recurrent_forward(ckpt_chunk=256)` 分块梯度检查点 |
| 全注意力层激活全部驻留 | 3 层 × ~3GB | `HybridBackbone(grad_checkpointing=True)` 按层重算 |
| logits 转 fp32 的 2GB 峰值 | 2×8192×32000×4B | `chunked_cross_entropy(chunk_tokens=2048)` |

**吞吐瓶颈**：`gdn2_recurrent_forward` 仍是 Python 级 `for t in range(T)`，9 个 GDN-2 层 × 8192 步
≈ 7.4 万次迭代/前向。估算全局 ~3k tok/s，50M token ≈ 4h。
**10B 全程必须换成 chunkwise 并行形式或官方 Triton kernel**（`fla`），否则不可行——已记录为已知限制。

## 训练目标分级

| stage | tokens | 目的 | 预计耗时（双 T4） |
|---|---|---|---|
| smoke | 8K | 管线连通性 | 分钟级 |
| 50m | 50M | loss 曲线 + 完整存档 | ~4h |
| 1B | 1B | short run 筛选 lr/OOM | 超出单次会话，需断点续训 |
| 10B | 10B | 主训 | 需 chunkwise kernel |

## 已修复的正确性缺陷（审计记录）

| 缺陷 | 影响 | 修复 |
|---|---|---|
| `FullAttention` 默认 `mask=None` → 双向注意力 | 自回归 LM 能看到未来 token，loss 虚假偏低，模型学不到预测 | 改 `F.scaled_dot_product_attention(is_causal=True)`，并测「改末位 token，前面输出不变」 |
| 初始 loss 628 | `nn.Embedding` N(0,1) → logits 爆炸 | GPT-2 风格 std=0.02 + 残差 1/sqrt(2·n_layer) |
| `self.apply(_init_weights)` 清零 out_gate bias | 违反计划书 §5 `bias_init=1.0` | 初始化后显式恢复 |
| DDP 未包装（if 块合并） | 两卡梯度不同步 | 初始化与 `DDP()` 包装拆开 |
| `consumed` 统计本地 token 比全局目标 | 双卡训 2 倍量 | `local_budget = total // world` |

CMM 的双向注意力作用于**单步内的记忆槽**（`[sink; LTM; STM]`），时间由外层循环推进，
不构成跨时间泄漏，无需修改。
