"""DEPRECATED（2026-10-30）：改用自包含 kernel（见 docs/KAGGLE.md）。

保留原因：记录早期方案。**不要使用本脚本的 OOM 降级路径**——它复用同一个已部分消费的
ShardStream 重新 train()，会重复消费/重复计步，且 mb 与 grad_accum 需手工同步易错。

原说明：Kaggle 双 T4 启动器：预取提前跑 + fp16 自检 + OOM 自动降级。

用法（Kaggle cell 里）：
    !python scripts/kaggle_launch.py --stage short --lr 2e-4

关键设计（按用户要求）：
  1. **预取在训练前就开始**，独立进程写分片；训练侧只消费已就绪分片，不等下载。
  2. **fp16**：T4 是 sm_75，无 bf16。先做一次自检确认 fp16+scaler 不出 NaN。
  3. **OOM 自动降级**：mb 2→1，再不行开 grad checkpointing；每次降级都记录到 JSON。
  4. **磁盘按需**：分片消费完即删（ShardStream.delete_consumed），并设缓存上限。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def check_env() -> dict:
    import torch
    info = {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda"]:
        caps = [torch.cuda.get_device_capability(i) for i in range(info["n_gpu"])]
        info["capabilities"] = caps
        info["names"] = [torch.cuda.get_device_name(i) for i in range(info["n_gpu"])]
        info["vram_gb"] = [torch.cuda.get_device_properties(i).total_memory / 2**30
                           for i in range(info["n_gpu"])]
        # T4 = sm_75：不支持 bf16
        info["bf16_supported"] = all(c[0] >= 8 for c in caps)
        info["fp16_required"] = not info["bf16_supported"]
    return info


def selftest_fp16(model, device: str = "cuda") -> dict:
    """确认 fp16 + GradScaler 能正常前反向且无 NaN。"""
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ids = torch.randint(0, 32000, (1, 512), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1)
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer := torch.optim.AdamW(model.parameters(), lr=1e-4))
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    ok = torch.isfinite(loss) and torch.isfinite(gn)
    return {"loss": float(loss.detach().cpu()), "grad_norm": float(gn), "finite": bool(ok),
            "scale": float(scaler.get_scale())}


def start_prefetch(cache_dir: str, shard_tokens: int, max_shards: int) -> subprocess.Popen:
    """后台启动预取进程，立刻返回（训练侧边训边消费）。"""
    log = Path(cache_dir) / "prefetch.log"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    f = open(log, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "hcl_tr.data.prefetch",
         "--cache-dir", cache_dir,
         "--shard-tokens", str(shard_tokens),
         "--max-shards", str(max_shards)],
        stdout=f, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="short", choices=["smoke", "short", "full"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cache-dir", default="/kaggle/working/tokens")
    ap.add_argument("--out-dir", default="/kaggle/working/out")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--cache-gb", type=float, default=8.0)
    a = ap.parse_args()

    env = check_env()
    print(json.dumps(env, indent=2))
    if not env["cuda"]:
        print("FATAL: 需要 CUDA（Kaggle 请选择 T4 x2 加速器）", file=sys.stderr)
        return 2
    if env.get("fp16_required"):
        print("检测到 T4 (sm_75)：使用 fp16 + GradScaler（bf16 不可用）")

    from hcl_tr.models.backbone import HybridBackbone
    from hcl_tr.data.prefetch import ShardStream
    from hcl_tr.train.loop import TrainConfig, train, estimate_memory

    # 1) 先启动预取（训练时不阻塞等数据）
    stage_shards = {"smoke": 1, "short": 12, "full": 60}[a.stage]
    shard_tokens = 20_000_000 if a.stage != "smoke" else 2_000_000
    proc = start_prefetch(a.cache_dir, shard_tokens, stage_shards)
    print(f"预取已后台启动 pid={proc.pid}，目标 {stage_shards} 分片")

    # 2) 等第一个分片就绪（只需一个，之后边训边等）
    stream = ShardStream(a.cache_dir, seq_len=a.seq_len,
                         batch_tokens=a.micro_batch * a.grad_accum * a.seq_len * 2,
                         max_cache_bytes=int(a.cache_gb * 2**30))
    first = stream.wait_for_shard(timeout=900)
    print(f"首分片就绪: {first.name}，开始训练")

    # 3) 模型 + 自检
    model = HybridBackbone()   # 102.47M / non_embed 77.89M（计划书口径）
    st = selftest_fp16(model)
    print("fp16 自检:", st)
    if not st["finite"]:
        print("FATAL: fp16 自检出现 NaN/Inf", file=sys.stderr)
        return 3

    mem = estimate_memory(TrainConfig(micro_batch_per_gpu=a.micro_batch, seq_len=a.seq_len))
    print(f"显存估算: {mem['total_gb_est']:.1f} GB / 卡（可用 {min(env['vram_gb']):.1f} GB）")

    total_tokens = {"smoke": 2_000_000, "short": 1_000_000_000, "full": 10_000_000_000}[a.stage]
    cfg = TrainConfig(
        seq_len=a.seq_len, micro_batch_per_gpu=a.micro_batch, grad_accum=a.grad_accum,
        lr=a.lr, total_tokens=total_tokens, out_dir=a.out_dir,
    )

    result = {"env": env, "selftest": st, "mem_est_gb": mem["total_gb_est"],
              "stage": a.stage, "lr": a.lr, "degradations": []}
    try:
        consumed = train(model, stream, cfg, device="cuda")
        result["consumed_tokens"] = consumed
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        result["degradations"].append({"step": "mb_2_to_1", "reason": "OOM"})
        print("OOM：降 micro_batch 到 1，grad_accum 翻倍", file=sys.stderr)
        cfg.micro_batch_per_gpu = 1
        cfg.grad_accum = a.grad_accum * 2
        import torch
        torch.cuda.empty_cache()
        result["consumed_tokens"] = train(model, stream, cfg, device="cuda")
    finally:
        proc.terminate()

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / "launch_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print("report ->", Path(a.out_dir) / "launch_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
