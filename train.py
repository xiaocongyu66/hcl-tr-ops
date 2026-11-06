"""train.py —— HCL-TR 训练入口。启动前强制通过 SOURCE_LOCK CI 闸门。

用法:
  # 1) 先把语料分词写成分片（另开一个终端，可持续追加）
  python -m hcl_tr.data.prefetch --cache-dir /data/shards --dataset HuggingFaceFW/fineweb-edu
  # 2) 训练
  python train.py --stage 0 --config src/hcl_tr/configs/stage0_mvp1a.yaml --data-dir /data/shards

红线（计划书 §2.3 / §12 第 1 步）：
  - 锁不齐 → train.py 不启动
  - UNVERIFIED 模块（DASC / When2Think）不得进入训练路径
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_lock_gate() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "lock_check.py")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("SOURCE_LOCK 闸门失败，train.py 不启动：", file=sys.stderr)
        print(proc.stdout + proc.stderr, file=sys.stderr)
        sys.exit(3)
    print("SOURCE_LOCK 闸门通过")


FREEZE_STAGES = {
    # 冻结模块 -> 禁止启动的训练阶段（SOURCE_LOCK 状态驱动）
    # 2026-10-29：When2Think 已解冻（arXiv:2609.19671v2 公式已核 + HF sha 已锁），
    # 阶段 2 解禁；DASC 仍无 SGLang 可锁定 commit，部署阶段继续冻结。
    "DASC": ["deploy"],
}


def check_stage_allowed(stage: str) -> None:
    source_lock = (REPO_ROOT / "SOURCE_LOCK.md").read_text(encoding="utf-8")
    for module, blocked in FREEZE_STAGES.items():
        if stage in blocked:
            # 检查该模块在 SOURCE_LOCK 中是否仍为 UNVERIFIED
            block = source_lock.split(f"## 模块：{module}")[-1][:400]
            if "UNVERIFIED" in block:
                print(
                    f"阶段 {stage} 依赖模块 {module}，但其状态为 UNVERIFIED（冻结）。"
                    f"train.py 拒绝启动。",
                    file=sys.stderr,
                )
                sys.exit(4)


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_kwargs_from_config(cfg: dict) -> dict:
    """把 YAML 的 model 段映射到 HybridBackbone 的构造参数。

    只透传 HybridBackbone 真正接受的键，避免配置文件里的说明性字段
    在构造时被静默忽略或抛错。
    """
    from hcl_tr.models.backbone import HybridBackbone
    import inspect

    accepted = set(inspect.signature(HybridBackbone.__init__).parameters) - {"self"}
    section = cfg.get("model") or {}
    unknown = sorted(set(section) - accepted)
    if unknown:
        raise SystemExit(f"配置中 model 段含未知参数: {unknown}")
    return dict(section)


def train_config_from_config(cfg: dict, *, lr: float | None, out_dir: str,
                             resume_from: str | None, ckpt_sha256: str | None):
    from hcl_tr.train.loop import TrainConfig, grad_accum_for_global_batch

    batch = cfg.get("batch") or {}
    n_gpu = int(batch.get("n_gpu") or max(1, _cuda_count()))
    micro = int(batch.get("micro_batch_per_gpu", 2))
    seq_len = int(batch.get("seq_len") or cfg.get("seq_len") or 8192)
    global_tokens = int(batch.get("global_batch_tokens") or 262144)
    grad_accum = int(batch.get("grad_accum") or
                     grad_accum_for_global_batch(micro, seq_len, n_gpu, global_tokens))
    lr_cfg = cfg.get("lr") or {}
    if lr is None:
        sweep = lr_cfg.get("sweep") or []
        lr = float(sweep[len(sweep) // 2]) if sweep else 3e-4
    optim = cfg.get("optimizer") or {}
    return TrainConfig(
        seq_len=seq_len,
        micro_batch_per_gpu=micro,
        grad_accum=grad_accum,
        global_batch_tokens=global_tokens,
        lr=float(lr),
        weight_decay=float(optim.get("weight_decay", 0.1)),
        betas=tuple(optim.get("betas", (0.9, 0.95))),
        eps=float(optim.get("eps", 1e-8)),
        grad_clip=float(cfg.get("grad_clip", 1.0)),
        log_every=int(cfg.get("log_every", 10)),
        ckpt_every_tokens=int(cfg.get("ckpt_every_tokens", 100_000_000)),
        warmup_tokens=int(lr_cfg.get("warmup_tokens", 1_000_000_000)),
        total_tokens=int(cfg.get("train_tokens", 1_000_000_000)),
        out_dir=out_dir,
        seed=int(cfg.get("seed", 0)),
        resume_from=resume_from,
        ckpt_sha256=ckpt_sha256,
        fields={"config": cfg},
    )


def _cuda_count() -> int:
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="训练阶段：0/1/1.5/2/3/4/deploy")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", help="含 shard_*.npy/.ready 的分片缓存目录")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--lr", type=float, help="覆盖配置中的 lr.sweep（默认取中位值）")
    ap.add_argument("--n-gpu", type=int, help="DDP 进程数；>=2 时走 train_ddp")
    ap.add_argument("--resume-from")
    ap.add_argument("--ckpt-sha256")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_lock_gate()
    check_stage_allowed(args.stage)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"配置不存在: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_yaml(cfg_path)
    print(f"stage={args.stage} config={cfg_path}")

    if args.dry_run:
        kwargs = model_kwargs_from_config(cfg)
        tcfg = train_config_from_config(cfg, lr=args.lr, out_dir=args.out_dir,
                                        resume_from=args.resume_from,
                                        ckpt_sha256=args.ckpt_sha256)
        print("dry-run: 闸门通过、阶段允许。未开始实际训练。")
        print(json.dumps({"model": kwargs,
                          "train": {k: getattr(tcfg, k) for k in (
                              "seq_len", "micro_batch_per_gpu", "grad_accum",
                              "global_batch_tokens", "lr", "total_tokens", "out_dir")}},
                         ensure_ascii=False, indent=2))
        return 0

    if not args.data_dir:
        print("缺少 --data-dir。先运行：\n"
              "  python -m hcl_tr.data.prefetch --cache-dir <dir> --dataset <hf-dataset>",
              file=sys.stderr)
        return 2
    if _cuda_count() == 0:
        print("训练需要 CUDA；当前环境没有可用 GPU。", file=sys.stderr)
        return 5

    from hcl_tr.models.backbone import HybridBackbone
    from hcl_tr.train.loop import train, train_ddp

    kwargs = model_kwargs_from_config(cfg)
    tcfg = train_config_from_config(cfg, lr=args.lr, out_dir=args.out_dir,
                                    resume_from=args.resume_from,
                                    ckpt_sha256=args.ckpt_sha256)
    n_gpu = args.n_gpu or _cuda_count()

    if n_gpu >= 2:
        consumed = train_ddp(kwargs, args.data_dir, tcfg, world_size=n_gpu)
    else:
        from hcl_tr.data.prefetch import ShardStream
        stream = ShardStream(args.data_dir, seq_len=tcfg.seq_len,
                             batch_tokens=tcfg.global_batch_tokens, ddp=False)
        consumed = train(HybridBackbone(**kwargs), stream, tcfg)
    print(f"训练结束：本 rank 消费 {consumed} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
