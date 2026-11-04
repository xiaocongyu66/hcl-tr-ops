"""训练循环：fp16 混合精度（T4 无 bf16 硬件支持）+ GradScaler + 预取数据流。

T4 注意事项（用户环境：双 T4 × 16GB）：
  - **T4 是 Turing (sm_75)，不支持 bf16**。必须用 fp16 + GradScaler（loss scale），
    否则会出现 NaN 或直接报错。
  - GDN-2 的递推在 fp32 里做（数值敏感），只在 matmul 上用 autocast(fp16)。
  - 双卡用 DDP；global_batch_tokens 固定 262144，分到两卡。
"""

from __future__ import annotations

import contextlib
import json
import random
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


@dataclass
class TrainConfig:
    seq_len: int = 8192
    micro_batch_per_gpu: int = 2       # T4 16GB 起步值；OOM 时降
    grad_accum: int = 4
    global_batch_tokens: int = 262144
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_tokens: int = 1_000_000_000
    total_tokens: int = 1_000_000_000
    use_fp16: bool = True              # T4：必须 True
    ce_chunk_tokens: int = 2048         # 分块 CE，控制 loss 显存峰值
    # SPAR（arXiv:2609.27925）超参：训练中固定，不调（计划书 §4.8）
    spar_lambda_inv: float = 1.0
    spar_suffix_len: int = 512
    spar_chunk_size: int = 64
    spar_nll_threshold: float = 1.0
    spar_gain_threshold: float = 0.1
    log_every: int = 10
    ckpt_every_tokens: int = 100_000_000
    out_dir: str = "out"
    resume_from: str | None = None   # 断点续训：指向 ckpt_*.pt
    ckpt_sha256: str | None = None   # resume_from 的完整性校验（必填，防静默加载错误权重）
    seed: int = 0
    fields: dict = field(default_factory=dict)


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor,
                         chunk_tokens: int = 2048, ignore_index: int = -100) -> torch.Tensor:
    """分块计算 CE，避免一次性把 [B*T, V] 转 fp32 撑爆显存。

    8K×2×32000 fp32 ≈ 2GB；分块后峰值降到 chunk_tokens×V。
    """
    V = logits.size(-1)
    flat_lg = logits.reshape(-1, V)
    flat_lb = labels.reshape(-1)
    n = flat_lb.numel()
    total = flat_lg.new_zeros((), dtype=torch.float32)
    count = 0
    for i in range(0, n, chunk_tokens):
        lg = flat_lg[i:i + chunk_tokens].float()
        lb = flat_lb[i:i + chunk_tokens]
        total = total + F.cross_entropy(lg, lb, ignore_index=ignore_index,
                                        reduction="sum")
        count += (lb != ignore_index).sum()
    return total / count.clamp_min(1)


def grad_accum_for_global_batch(micro_batch: int, seq_len: int, n_gpu: int,
                                target_global_tokens: int = 262144) -> int:
    """按目标 global batch 反解 grad_accum（单卡/双卡自适应）。

    例：mb=2, seq=8192, 单 T4 → accum=16；双 T4 → accum=8。
    """
    per_micro = micro_batch * seq_len * max(1, n_gpu)
    return max(1, target_global_tokens // per_micro)


def build_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    # fused=True：用单 kernel 完成 AdamW 更新（CUDA 上显著减少小 kernel 数）
    try:
        return AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=True,
        )
    except (TypeError, RuntimeError):
        return AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
        )


def _is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def train(model, stream, cfg: TrainConfig, device: str = "cuda",
          ddp: bool = False, local_rank: int = 0):
    """主训练循环。stream 是 data.prefetch.ShardStream。

    fp16 路径要点：
      - autocast(dtype=torch.float16) 包裹前向
      - GradScaler 处理梯度缩放（fp16 下小梯度会下溢）
      - scaler.unscale_ 后再 clip_grad_norm_，最后 scaler.step / scaler.update
    """
    # 计划书红线：不许静默忽略未实现的目标。SPAR 需要 clean/corrupt 配对数据算 KL，
    # IDAC 需要 RLVR 采样+verifier+BWS，两者当前都未接入训练循环。
    # 若配置打开了它们却仍按纯 NLL 训练，归因组会与对照组不可区分 → 直接报错。
    _m = model.module if isinstance(model, DDP) else model
    use_spar = bool(getattr(_m, "use_spar", False))
    if use_spar and cfg.seq_len > 4096:
        import warnings
        warnings.warn(
            f"use_spar=True 且 seq_len={cfg.seq_len}：SPAR 需要同时持有 clean/corrupt 两份 logits "
            f"（各 B×T×V），8K 下显存风险高。建议 SPAR 组用 seq_len<=4096 或减小 mb。",
            RuntimeWarning, stacklevel=2,
        )
    if getattr(_m, "use_idac", False):
        raise NotImplementedError(
            "use_idac=True 但训练循环尚未实现 IDAC 的 RLVR 路径"
            "（采样/verifier/Eq.5-8 奖励与 BWS）；实现前不可训练。"
        )

    model = model.to(device)
    if ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        # 关键：无论进程组是本次初始化还是调用方预先初始化，都必须包装成 DDP，
        # 否则梯度不会跨卡 all-reduce（会静默跑偏，两卡各训各的）。
        if not isinstance(model, DDP):
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DDP) else model

    opt = build_optimizer(model, cfg)
    tokens_per_step = cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len
    world = dist.get_world_size() if (ddp and dist.is_initialized()) else 1
    tokens_per_step *= world
    if cfg.total_tokens < tokens_per_step:
        raise ValueError("token 预算不足一个完整全局优化步")
    # 每 rank 的本地 token 预算 = 全局目标 / world（consumed 统计的是本 rank 本地量）
    local_budget = max(1, cfg.total_tokens // world)
    total_steps = max(1, cfg.total_tokens // tokens_per_step)
    warmup_steps = max(1, cfg.warmup_tokens // tokens_per_step)
    sched = LambdaLR(opt, lambda s: cosine_with_warmup(s, warmup_steps, total_steps))

    # T4 无 bf16：用 fp16 + GradScaler
    amp_dtype = torch.float16 if cfg.use_fp16 else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_fp16)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # corrupt 视图的随机源固定，保证可复现（计划书 §10：固定随机种子）
    spar_gen = torch.Generator(device=device).manual_seed(0)

    model.train()
    consumed = 0
    step = 0
    t0 = time.time()
    loss_hist = []
    last_ckpt_tokens = 0   # 距上次存档的基准（续训时在下面按 ckpt 的重置）
    resume_tokens = 0      # 续训起点，用于检测"零训练"
    data_manifest = None
    if hasattr(stream, "cache_dir") and (not stream.delete_consumed or stream.ddp):
        from ..checkpoint import sha256_file
        paths = sorted(Path(stream.cache_dir).glob("shard_*.npy"))
        if (paths and stream.planned_shards is not None
                and [p.name for p in paths] == [f'shard_{i:06d}.npy' for i in range(stream.planned_shards)]
                and all(p.with_suffix('.ready').exists() for p in paths)):
            data_manifest = {p.name: sha256_file(p) for p in paths}
    resume_runtime = None

    # 断点续训：10B 无法在单次 Kaggle 会话跑完，必须能跨会话接着训
    if cfg.resume_from:
        if not cfg.ckpt_sha256:
            raise ValueError(
                "resume_from 必须同时提供 ckpt_sha256（SOURCE_LOCK CI 闸门第 2 条）："
                "未校验 hash 就加载 checkpoint 可能静默加载错误权重。"
            )
        from ..checkpoint import safe_torch_load
        ck = safe_torch_load(cfg.resume_from, sha256=cfg.ckpt_sha256,
                             map_location=device)
        if ck.get("format_version") != 2 or ck.get("world_size") != world:
            raise ValueError("checkpoint token 口径或 world_size 不明确；拒绝含糊续训")
        if data_manifest is None or ck.get("data_manifest") != data_manifest:
            raise ValueError("续训需要保留完整且相同的数据分片，数据身份不匹配")
        previous_cfg = ck.get("train_config", {})
        # LambdaLR 的闭包不在 state_dict 内；拒绝预算/调度/目标迁移，确保恢复同一轨迹。
        for key in ("seq_len", "micro_batch_per_gpu", "grad_accum", "use_fp16",
                    "total_tokens", "warmup_tokens", "lr", "weight_decay", "betas", "eps",
                    "grad_clip", "spar_lambda_inv", "spar_suffix_len", "spar_chunk_size",
                    "spar_nll_threshold", "spar_gain_threshold", "seed"):
            if previous_cfg.get(key) != getattr(cfg, key):
                raise ValueError(f"续训不能改变计数或精度配置: {key}")
        if ck.get('model_config') != cfg.fields.get('model_config', {}):
            raise ValueError('续训模型配置不一致')
        consumed = int(ck["tokens_local"])
        step = int(ck["step"]) + 1
        if consumed * world != ck["tokens_global"] or step * tokens_per_step != ck["tokens_global"]:
            raise ValueError("checkpoint step/token 计数不一致")
        if "runtime_states" not in ck or len(ck["runtime_states"]) != world:
            raise ValueError("checkpoint 缺少各 rank 随机状态，不能精确续训")
        resume_runtime = ck["runtime_states"][local_rank]
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        last_ckpt_tokens = consumed
        resume_tokens = consumed
        if _is_main():
            print(f"[resume] from {cfg.resume_from} @ tokens={consumed} step={step}", flush=True)

    # 按「优化步」驱动，而不是按 micro-batch 驱动。
    # 旧实现按 micro-batch 循环 + `consumed >= local_budget` 中途 break，会在某次
    # grad-accum 周期中间退出：已做前/反向的梯度被丢弃（双卡 1e9 配置实测浪费
    # 6 个 micro-batch ≈196k token），且小预算下会超训。改为精确步数后：
    #   global_tokens_trained == steps_done * tokens_per_step（每步 consume 恰好 grad_accum 个 micro-batch）
    total_steps_local = max(1, local_budget // (cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len))
    start_step = step  # 续训时从 ckpt 的步数继续
    it = stream.batches(stop_after_tokens=(total_steps_local + 1) *
                        cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len)
    for _ in range(start_step * cfg.grad_accum):
        try:
            next(it)
        except (StopIteration, TimeoutError) as exc:
            raise ValueError("续训数据不足以恢复已消费的位置") from exc
    if resume_runtime is not None:
        torch.set_rng_state(resume_runtime["torch"].cpu())
        random.setstate(resume_runtime["python"])
        spar_gen.set_state(resume_runtime["spar"].cpu())
        if resume_runtime["cuda"] is not None:
            torch.cuda.set_rng_state(resume_runtime["cuda"].cpu(), device=device)

    def runtime_states():
        state = {"torch": torch.get_rng_state(), "python": random.getstate(),
                 "spar": spar_gen.get_state(),
                 "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}
        if ddp and dist.is_initialized():
            states = [None] * world
            dist.all_gather_object(states, state)
            return states
        return [state]

    def checkpoint_state(completed_step, states):
        return {"format_version": 2, "model": raw_model.state_dict(),
                "optim": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "tokens": consumed * world,
                "tokens_global": consumed * world, "tokens_local": consumed,
                "world_size": world, "step": completed_step,
                "steps_completed": completed_step + 1, "train_config": asdict(cfg),
                "model_config": cfg.fields.get("model_config", {}),
                "data_manifest": data_manifest, "runtime_states": states,
                "exact_resume_available": data_manifest is not None}

    for step in range(start_step, total_steps_local):
        loss_acc = None
        exhausted = False
        microbatch_stats = []
        for i in range(cfg.grad_accum):
            try:
                batch = next(it)
            except (StopIteration, TimeoutError) as e:
                # 数据耗尽 或 预取停摆：都走「全 rank 协调退出」，
                # 绝不单侧抛异常（那会让另一 rank 永久卡在 all-reduce）。
                exhausted = True
                if _is_main():
                    print(f"[warn] 数据流中断: {type(e).__name__}", flush=True)
            if ddp and dist.is_initialized():
                flag = torch.tensor([int(exhausted)], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                exhausted = bool(flag.item())
            if exhausted:
                break
            batch = batch.to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            use_no_sync = ddp and (i < cfg.grad_accum - 1)
            ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_fp16):
                    logits = model(input_ids)
                    views = [raw_model.chunk_stats()] if hasattr(raw_model, "chunk_stats") else []
                    if use_spar:
                        from ..models.spar import spargate_suffix_mask, spar_loss, corrupt_prefix_view
                        corrupt_ids = corrupt_prefix_view(
                            batch, suffix_len=cfg.spar_suffix_len,
                            chunk_size=cfg.spar_chunk_size, generator=spar_gen)
                        logits_k = model(corrupt_ids[:, :-1])
                        if hasattr(raw_model, "chunk_stats"):
                            views.append(raw_model.chunk_stats())
                        mask = spargate_suffix_mask(
                            logits, logits_k, labels,
                            suffix_start=max(0, labels.shape[1] - cfg.spar_suffix_len),
                            nll_threshold=cfg.spar_nll_threshold,
                            gain_threshold=cfg.spar_gain_threshold)
                        loss, _, _ = spar_loss(logits.float(), logits_k.float(), labels, mask,
                                               lbda_inv=cfg.spar_lambda_inv)
                    else:
                        loss = chunked_cross_entropy(logits, labels, chunk_tokens=cfg.ce_chunk_tokens)
                    loss = loss / cfg.grad_accum
                microbatch_stats.append({"microbatch": i, "views": views})
                scaler.scale(loss).backward()
            # 不在 micro-batch 内 .item()（会强制 CUDA 同步）；累积张量，步末读一次
            loss_acc = loss.detach() if loss_acc is None else loss_acc + loss.detach()

        if exhausted:
            opt.zero_grad(set_to_none=True)
            if _is_main():
                print(f"[stop] 数据耗尽，在所有 rank 同步停止于 step {step}", flush=True)
            break

        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        invalid_grad = (~torch.isfinite(grad_norm)).to(device=device, dtype=torch.int32)
        if ddp and dist.is_initialized():
            dist.all_reduce(invalid_grad, op=dist.ReduceOp.MAX)
        if bool(invalid_grad.item()):
            raise RuntimeError("梯度范数非有限值，拒绝更新或保存成功 checkpoint")
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()

        # 已 step 的量 = 已实际训练的量（不再有"喂了但没 step"的浪费）
        consumed = (step + 1) * cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len

        if ddp and dist.is_initialized():
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            loss_acc /= world
        loss_value = float(loss_acc) if loss_acc is not None else float("nan")
        if not math.isfinite(loss_value):
            raise RuntimeError("训练 loss 非有限值，停止运行")
        elapsed = time.time() - t0
        global_tok_s = (consumed - resume_tokens) * world / max(1e-9, elapsed)
        metric = {"step": step, "steps_completed": step + 1, "loss": loss_value,
                  "tokens_local": consumed, "tokens_global": consumed * world,
                  "world_size": world, "global_tok_s": global_tok_s,
                  "microbatches": microbatch_stats, "rank": local_rank}
        with (out / f"metrics_rank{local_rank}.jsonl").open("a") as f:
            f.write(json.dumps(metric, allow_nan=False) + "\n")
        if _is_main():
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric, allow_nan=False) + "\n")
        if _is_main() and step % cfg.log_every == 0:
            lr = sched.get_last_lr()[0]
            loss_hist.append(loss_value)
            spike = False
            if len(loss_hist) > 20:
                med = sorted(loss_hist[-20:])[len(loss_hist[-20:]) // 2]
                spike = loss_hist[-1] > 3 * med
            _cs = raw_model.chunk_stats() if hasattr(raw_model, "chunk_stats") else {}
            _ce = sorted({v[0] for v in _cs.values()}) if _cs else []
            _dm = max((v[1] for v in _cs.values()), default=0.0)
            print(f"step {step} loss {loss_value:.4f} "
                  f"lr {lr:.2e} tokens_local {consumed} tokens_global {consumed * world} "
                  f"world_size {world} global_tok_s {global_tok_s:.0f} "
                  f"dmax {_dm:.2f} chunk_eff {_ce}"
                  + ("  [LOSS SPIKE]" if spike else ""), flush=True)

        # 按「距上次存档的 token 数」触发，而不是数目录里的文件——
        # 后者在换输出目录/多 rank 时会漂移。
        if (consumed - last_ckpt_tokens) * world >= cfg.ckpt_every_tokens:
            states = runtime_states()
            if _is_main():
                p = out / f"ckpt_{consumed * world}.pt"
                torch.save(checkpoint_state(step, states), p)
                print(f"[ckpt] {p}", flush=True)
            last_ckpt_tokens = consumed

    # 零训练必须报错：否则上层会把"没数据"当成训练成功（曾出现 shards=0 仍报 OK）。
    if consumed <= resume_tokens:
        raise RuntimeError(
            f"没有训练任何数据：consumed={consumed} <= 续训起点 {resume_tokens}。"
            f"检查分片是否写好、planned_shards 是否正确、预取是否供得上。"
        )

    states = runtime_states()
    if _is_main():
        completed_step = consumed // (cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len) - 1
        torch.save(checkpoint_state(completed_step, states), out / "final.pt")
        print(f"done: tokens_global={consumed * world} tokens_local={consumed} world_size={world}", flush=True)
    # 每个 rank 都写自己的实际训练量，供 train_ddp 汇总
    (out / f"consumed_rank{local_rank}.txt").write_text(str(consumed))
    return consumed


def _wait_workers(procs):
    while True:
        codes = [p.exitcode for p in procs]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError(f"DDP worker 失败: {codes}")
        if all(code == 0 for code in codes):
            return
        for p in procs:
            if p.exitcode is None:
                p.join(timeout=0.1)


def _stop_workers(procs):
    for p in procs:
        if p.is_alive():
            p.terminate()
    deadline = time.monotonic() + 5.0
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
    for p in procs:
        p.join(timeout=5.0)
        if p.is_alive():
            raise RuntimeError(f"无法回收 worker pid={p.pid}")


def train_ddp(model_kwargs: dict | None, cache_dir: str, cfg: TrainConfig,
              world_size: int | None = None, ddp_kwargs: dict | None = None) -> int:
    """双卡（Kaggle T4 x2）DDP 训练入口。

    - 用 torch.multiprocessing.spawn 起 world_size 个进程，每进程一张卡。
    - 每 rank 的 ShardStream 按 index % world_size 取分片，样本不重复。
    - train() 内部用 DDP 包装，梯度经 all-reduce 同步。
    """
    import torch.multiprocessing as mp

    ddp_kwargs = ddp_kwargs or {}
    n_gpu = world_size or torch.cuda.device_count()
    if n_gpu < 2:
        raise RuntimeError(f"train_ddp 需要 >=2 张卡，检测到 {n_gpu}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("consumed_rank*.txt")) or (out / "metrics.jsonl").exists():
        raise ValueError("输出目录已有训练记录，请使用独立目录，避免混入旧实验结果")
    # 只传纯数据（model_kwargs 是 dict），避免 lambda/局部函数无法 pickle
    ctx = mp.get_context("spawn")
    procs = []
    try:
        for r in range(n_gpu):
            p = ctx.Process(target=_ddp_worker,
                            args=(r, n_gpu, model_kwargs or {}, cache_dir, cfg, ddp_kwargs))
            p.start()
            procs.append(p)
        _wait_workers(procs)
    finally:
        _stop_workers(procs)
    return _collect_consumed(out, n_gpu, cfg)


def _collect_consumed(out, world_size, cfg):
    paths = [Path(out) / f"consumed_rank{r}.txt" for r in range(world_size)]
    if not all(p.is_file() for p in paths):
        raise RuntimeError("worker 未报告全部 rank 的实际 token 数")
    consumed = [int(p.read_text().strip()) for p in paths]
    if len(set(consumed)) != 1 or min(consumed) <= 0:
        raise RuntimeError(f"rank token 数不一致或为零: {consumed}")
    per_step = cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len * world_size
    expected = max(1, cfg.total_tokens // per_step) * per_step
    total = sum(consumed)
    if total != expected:
        raise RuntimeError(f"未完成完整步预算: tokens_global={total}, expected={expected}")
    return total


def _ddp_worker(rank: int, world_size: int, model_kwargs: dict, cache_dir: str,
                cfg: TrainConfig, ddp_kwargs: dict) -> None:
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(ddp_kwargs.get("port", 29500))
    import torch.distributed as dist
    from ..data.prefetch import ShardStream
    from ..models.backbone import HybridBackbone
    torch.cuda.set_device(rank)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        stream = ShardStream(cache_dir, seq_len=cfg.seq_len,
                             batch_tokens=cfg.micro_batch_per_gpu * cfg.seq_len,
                             max_cache_bytes=ddp_kwargs.get("max_cache_bytes"),
                             delete_consumed=ddp_kwargs.get("delete_consumed", False),
                             rank=rank, world_size=world_size,
                             planned_shards=ddp_kwargs.get("planned_shards"), ddp=True)
        model = HybridBackbone(**model_kwargs)
        cfg2 = TrainConfig(**{**cfg.__dict__,
                              "fields": {**cfg.fields, "model_config": model_kwargs}})
        if rank == 0:
            Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
            initial_decay = {}
            for name, module in model.named_modules():
                if hasattr(module, "A_log") and hasattr(module, "dt_bias"):
                    a = module.A_log.detach().exp()
                    dt = F.softplus(module.dt_bias.detach())
                    initial_decay[name] = {"a_min": float(a.min()), "a_max": float(a.max()),
                                           "dt_min": float(dt.min()), "dt_max": float(dt.max())}
            (Path(cfg.out_dir) / "effective_config.json").write_text(
                json.dumps({"model": model_kwargs, "train": asdict(cfg2),
                            "world_size": world_size, "initial_decay": initial_decay}, indent=2))
        train(model, stream, cfg2, device=f"cuda:{rank}", ddp=True, local_rank=rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def estimate_memory(cfg: TrainConfig, d_model=768, n_layer=12, vocab=32000) -> dict:
    """粗略显存估算（用于选择合适的 micro_batch，避免盲试 OOM）。"""
    # 参数与梯度（fp32 master + fp16 copy）
    n_param = vocab * d_model + 12 * d_model * d_model * 12
    param_bytes = n_param * (4 + 4 + 2)          # master + grad + fp16
    # AdamW 两个 state
    optim_bytes = n_param * 8
    # 激活（近似：每层 10×B×T×d，fp16）
    act_bytes = n_layer * 10 * cfg.micro_batch_per_gpu * cfg.seq_len * d_model * 2
    return {
        "params_mb": n_param * 7 / 2**20,
        "optim_mb": optim_bytes / 2**20,
        "activations_mb": act_bytes / 2**20,
        "total_gb_est": (param_bytes + optim_bytes + act_bytes) / 2**30,
    }
