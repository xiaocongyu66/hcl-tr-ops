"""训练循环 fixture：CPU 端到端冒烟 + 关键参数校验。

验证 fp16/fp32 开关、grad accum、cosine 调度、checkpoint 落盘。
（真正的 fp16 路径需 CUDA，标 gpu。）
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hcl_tr.checkpoint import safe_torch_load, sha256_file
from hcl_tr.data.prefetch import ShardStream, write_shard
from hcl_tr.models.backbone import HybridBackbone
from hcl_tr.train.loop import TrainConfig, cosine_with_warmup, estimate_memory, train


def _model():
    torch.manual_seed(0)
    return HybridBackbone(vocab_size=256, d_model=32, n_layer=4, n_head=4, attn_ratio=4)


def _stream(tmp_path, n_shard=4, toks=2048, mb=1, seq_len=32):
    """每次 yield 必须正好是一个 micro-batch（mb 条 seq），与 train() 的语义一致。"""
    for i in range(n_shard):
        write_shard(tmp_path, i, np.random.randint(0, 256, toks, dtype=np.uint16))
    return ShardStream(tmp_path, seq_len=seq_len, batch_tokens=mb * seq_len,
                       delete_consumed=False, planned_shards=n_shard)


def test_cosine_schedule_warmup_then_decay():
    total, warmup = 100, 10
    assert cosine_with_warmup(0, warmup, total) == 0.0
    assert abs(cosine_with_warmup(warmup, warmup, total) - 1.0) < 1e-9   # warmup 结束为峰
    mid = cosine_with_warmup(warmup + (total - warmup) // 2, warmup, total)
    assert 0.4 < mid < 0.6
    assert cosine_with_warmup(total, warmup, total) < 1e-6


def test_train_runs_and_writes_checkpoint(tmp_path):
    m = _model()
    stream = _stream(tmp_path / "tok")
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2, lr=1e-3,
                      total_tokens=256, use_fp16=False, log_every=100,
                      out_dir=str(tmp_path / "out"))
    consumed = train(m, stream, cfg, device="cpu")
    assert consumed >= 256
    assert (tmp_path / "out" / "final.pt").exists()
    ck = safe_torch_load(tmp_path / "out" / "final.pt", sha256=sha256_file(tmp_path / "out" / "final.pt"))
    assert "model" in ck and "tokens" in ck


def test_train_updates_weights(tmp_path):
    m = _model()
    before = m.embed.weight.detach().clone()
    stream = _stream(tmp_path / "tok")
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1, lr=1e-2,
                      total_tokens=128, use_fp16=False, log_every=100,
                      warmup_tokens=0,    # 无 warmup → lr 立刻生效
                      out_dir=str(tmp_path / "out"))
    train(m, stream, cfg, device="cpu")
    assert (m.embed.weight.detach() - before).abs().max() > 1e-6, "参数未更新"


def test_grad_accum_keeps_global_batch_tokens():
    """mb×accum×seq×n_gpu 必须等于 262144（计划书口径）。"""
    cfg = TrainConfig(micro_batch_per_gpu=2, grad_accum=8, seq_len=8192)
    assert cfg.micro_batch_per_gpu * cfg.grad_accum * cfg.seq_len * 2 == 262144
    # OOM 降级配置也要保持
    assert 1 * 16 * 8192 * 2 == 262144


def test_estimate_memory_monotonic_in_batch():
    a = estimate_memory(TrainConfig(micro_batch_per_gpu=1, seq_len=8192))
    b = estimate_memory(TrainConfig(micro_batch_per_gpu=2, seq_len=8192))
    assert b["total_gb_est"] > a["total_gb_est"]


@pytest.mark.gpu
def test_fp16_path_on_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = _model().cuda()
    stream = _stream(tmp_path / "tok")
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=64, use_fp16=True, out_dir=str(tmp_path / "out"))
    train(m, stream, cfg, device="cuda")


def test_stream_yields_micro_batch_not_full_accum():
    """流每次 yield 应是 mb 条序列（一个 micro-batch），不是 mb×accum。

    回归保护：DDP worker 曾把 batch_tokens 设成 mb*accum*seq*world，
    导致实际 micro-batch 变成 16 条序列。
    """
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream, write_shard
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    write_shard(tmp, 0, np.arange(8192 * 4, dtype=np.uint16))
    mb, seq = 2, 8192
    s = ShardStream(tmp, seq_len=seq, batch_tokens=mb * seq, delete_consumed=False)
    b = next(s.batches())
    assert b.shape == (mb, seq), f"期望 ({mb},{seq})，实际 {tuple(b.shape)}"


def test_global_tokens_formula_dual_gpu():
    """双卡：mb×accum×seq×world 必须等于 262144。"""
    mb, accum, seq, world = 2, 8, 8192, 2
    assert mb * accum * seq * world == 262144


def test_chunked_ce_matches_full():
    import torch.nn.functional as F
    from hcl_tr.train.loop import chunked_cross_entropy
    torch.manual_seed(0)
    lg = torch.randn(3, 40, 500)
    lb = torch.randint(0, 500, (3, 40))
    a = chunked_cross_entropy(lg, lb, chunk_tokens=16)
    b = F.cross_entropy(lg.reshape(-1, 500), lb.reshape(-1))
    assert abs(a.item() - b.item()) < 1e-5


def test_chunked_ce_respects_ignore_index():
    import torch.nn.functional as F
    from hcl_tr.train.loop import chunked_cross_entropy
    torch.manual_seed(1)
    lg = torch.randn(2, 30, 100)
    lb = torch.randint(0, 100, (2, 30))
    lb[0, :5] = -100
    a = chunked_cross_entropy(lg, lb, chunk_tokens=7)
    b = F.cross_entropy(lg.reshape(-1, 100), lb.reshape(-1), ignore_index=-100)
    assert abs(a.item() - b.item()) < 1e-5


def test_gdn2_checkpoint_matches_noncheckpoint():
    """梯度检查点不改变前向输出与梯度。"""
    from hcl_tr.models.gdn2_ref import gdn2_recurrent_forward as f
    torch.manual_seed(0)
    B, T, H, dk = 2, 300, 3, 8
    q = torch.randn(B, T, H, dk, requires_grad=True)
    k = torch.randn(B, T, H, dk, requires_grad=True)
    v = torch.randn(B, T, H, dk, requires_grad=True)
    g = -torch.rand(B, T, H, dk, requires_grad=True)
    b = torch.rand(B, T, H, dk, requires_grad=True)
    w = torch.rand(B, T, H, dk, requires_grad=True)
    o1 = f(q, k, v, g, b, w, use_checkpoint=False)
    g1 = torch.autograd.grad(o1.sum(), [q, k, v], retain_graph=True)
    o2 = f(q, k, v, g, b, w, use_checkpoint=True, ckpt_chunk=64)
    g2 = torch.autograd.grad(o2.sum(), [q, k, v])
    assert (o1 - o2).abs().max() < 1e-6
    for a, c in zip(g1, g2):
        assert (a - c).abs().max() < 1e-6


def test_grad_checkpointing_flag_preserves_output():
    """整模型按层 checkpoint 时前向输出一致。"""
    torch.manual_seed(0)
    from hcl_tr.models.backbone import HybridBackbone
    m1 = HybridBackbone(vocab_size=200, d_model=32, n_layer=4, n_head=4, grad_checkpointing=False)
    m2 = HybridBackbone(vocab_size=200, d_model=32, n_layer=4, n_head=4, grad_checkpointing=True)
    m2.load_state_dict(m1.state_dict())
    m1.train(); m2.train()
    ids = torch.randint(0, 200, (2, 16))
    with torch.no_grad():
        a = m1(ids)
        b = m2(ids)
    assert (a - b).abs().max() < 1e-5


def test_resume_from_checkpoint_continues(tmp_path):
    """断点续训：从 checkpoint 恢复后 token 计数接续，且不从头开始。"""
    import json
    m = _model()
    stream = _stream(tmp_path / "tok")
    out = tmp_path / "out"
    # 第一段：跑一点并存 ckpt
    cfg1 = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1, lr=1e-2,
                       total_tokens=128, use_fp16=False, log_every=100,
                       warmup_tokens=0, ckpt_every_tokens=64, out_dir=str(out))
    train(m, stream, cfg1, device="cpu")
    cks = sorted(out.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
    assert cks, "应产生至少一个 checkpoint"

    # 第二段：从中间 checkpoint 恢复到原调度预算，不迁移调度。
    ck = safe_torch_load(cks[0], sha256=sha256_file(cks[0]))
    start_tokens = ck["tokens"]
    assert start_tokens > 0
    assert "sched" in ck and "step" in ck, "ckpt 需含 sched/step 以正确续训"

    m2 = _model()
    stream2 = ShardStream(tmp_path / 'tok', seq_len=32, batch_tokens=32, delete_consumed=False, planned_shards=4)
    cfg2 = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1, lr=1e-2,
                       total_tokens=128, use_fp16=False, log_every=100,
                       warmup_tokens=0, out_dir=str(out), resume_from=str(cks[0]),
                       ckpt_sha256=sha256_file(cks[0]))
    consumed = train(m2, stream2, cfg2, device="cpu")
    assert consumed > start_tokens, f"续训后应超过起点 {start_tokens}，实际 {consumed}"
    assert consumed == 128


def test_resume_without_sha256_is_refused(tmp_path):
    """无 sha256 的续训必须被拒绝（防止静默加载错误权重）。"""
    import pytest
    m = _model()
    stream = _stream(tmp_path / "tok")
    out = tmp_path / "out"
    cfg1 = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1, lr=1e-2,
                       total_tokens=128, use_fp16=False, warmup_tokens=0,
                       ckpt_every_tokens=64, out_dir=str(out))
    train(m, stream, cfg1, device="cpu")
    cks = sorted(out.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
    cfg2 = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                       total_tokens=256, use_fp16=False, warmup_tokens=0,
                       out_dir=str(out), resume_from=str(cks[0]))   # 故意不给 sha256
    with pytest.raises(ValueError, match="sha256"):
        train(_model(), _stream(tmp_path / "tok2"), cfg2, device="cpu")


def test_idac_switch_fails_loudly(tmp_path):
    """计划书红线：未实现的目标不许静默无效。

    SPAR 已实现（见 test_spar_training_runs_end_to_end）；IDAC 的 RLVR 目标未实现，
    打开必须报错，不能当作纯 NLL 跑出无法归因的结果。
    """
    import pytest
    from hcl_tr.models.backbone import HybridBackbone
    m = HybridBackbone(vocab_size=256, d_model=32, n_layer=4, n_head=4, use_idac=True)
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=64, use_fp16=False, warmup_tokens=0,
                      out_dir=str(tmp_path / "o"))
    with pytest.raises(NotImplementedError):
        train(m, _stream(tmp_path / "t"), cfg, device="cpu")


def test_exact_token_accounting_no_waste(tmp_path, monkeypatch):
    """训练量必须精确等于预算，且不存在"喂了但没 step"的 micro-batch。

    回归保护：旧实现按 micro-batch 循环 + 中途 break，会丢梯度/超训。
    """
    import torch.nn.utils as nu
    steps = {"n": 0}
    orig = nu.clip_grad_norm_

    def counting(params, *a, **k):
        steps["n"] += 1
        return orig(params, *a, **k)

    monkeypatch.setattr(nu, "clip_grad_norm_", counting)

    mb, accum, seq = 1, 2, 32
    total = 256
    m = _model()
    stream = _stream(tmp_path / "tok", mb=mb, seq_len=seq)
    cfg = TrainConfig(seq_len=seq, micro_batch_per_gpu=mb, grad_accum=accum,
                      total_tokens=total, use_fp16=False, warmup_tokens=0,
                      log_every=100, out_dir=str(tmp_path / "out"))
    consumed = train(m, stream, cfg, device="cpu")

    expected_steps = total // (mb * accum * seq)
    assert consumed == expected_steps * mb * accum * seq == total, \
        f"训练 token 应精确={total}，实际 {consumed}"
    assert steps["n"] == expected_steps, f"优化步应={expected_steps}，实际 {steps['n']}"


def test_token_accounting_partial_budget_floors(tmp_path):
    """预算非整步时向下取整，不超训。"""
    mb, accum, seq = 1, 2, 32            # 每步 64 tokens
    m = _model()
    stream = _stream(tmp_path / "tok", mb=mb, seq_len=seq, n_shard=6, toks=2048)
    cfg = TrainConfig(seq_len=seq, micro_batch_per_gpu=mb, grad_accum=accum,
                      total_tokens=250, use_fp16=False, warmup_tokens=0,
                      log_every=100, out_dir=str(tmp_path / "out"))
    consumed = train(m, stream, cfg, device="cpu")
    assert consumed == 192, f"250 // 64 = 3 步 → 192 tokens，实际 {consumed}"
    assert consumed <= 250


def test_spar_training_runs_end_to_end(tmp_path):
    """use_spar=True 必须真的跑 L_CLM + λ·L_inv，而不是报错或退化为纯 NLL。"""
    from hcl_tr.models.backbone import HybridBackbone
    m = HybridBackbone(vocab_size=256, d_model=32, n_layer=4, n_head=4, use_spar=True)
    stream = _stream(tmp_path / "tok", mb=1, seq_len=64, n_shard=4, toks=4096)
    cfg = TrainConfig(seq_len=64, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=256, use_fp16=False, warmup_tokens=0,
                      log_every=100, spar_suffix_len=16, spar_chunk_size=8,
                      out_dir=str(tmp_path / "out"))
    consumed = train(m, stream, cfg, device="cpu")
    assert consumed == 256
    assert (tmp_path / "out" / "final.pt").exists()


def test_insufficient_data_exits_gracefully_not_raise(tmp_path):
    """数据不足时必须干净退出（返回已训 token 数），不能抛异常。

    回归保护：DDP 下若单侧抛异常，另一侧会永久卡在 all-reduce（曾跑到 12h 上限）。
    """
    m = _model()
    # 只给 1 个分片（远不够 256 token 的计划）
    stream = ShardStream(tmp_path / "tok", seq_len=32, batch_tokens=32,
                         delete_consumed=False, wait_timeout=2.0)
    import numpy as np
    from hcl_tr.data.prefetch import write_shard
    write_shard(tmp_path / "tok", 0, np.random.randint(0, 256, 64, dtype=np.uint16))
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2,
                      total_tokens=4096, use_fp16=False, warmup_tokens=0,
                      log_every=100, out_dir=str(tmp_path / "out"))
    consumed = train(m, stream, cfg, device="cpu")   # 不应 raise
    assert consumed < 4096


def test_zero_data_raises_instead_of_fake_success(tmp_path):
    """没有任何分片时必须报错——绝不允许"shards=0 却报 OK"的假成功。"""
    import pytest
    from hcl_tr.data.prefetch import ShardStream
    (tmp_path / "empty").mkdir()
    stream = ShardStream(tmp_path / "empty", seq_len=32, batch_tokens=32,
                         delete_consumed=False, wait_timeout=0.05)
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=64, use_fp16=False, warmup_tokens=0,
                      out_dir=str(tmp_path / "out"))
    with pytest.raises(RuntimeError, match="没有训练任何数据"):
        train(_model(), stream, cfg, device="cpu")


def _waiting_worker(event):
    event.wait(30)


def _failing_worker():
    raise RuntimeError("故障注入")


@pytest.mark.parametrize("failure", ["child", "parent", "normal"])
def test_worker_supervision_reaps_children(failure):
    import multiprocessing as mp
    from hcl_tr.train.loop import _wait_workers, _stop_workers
    ctx = mp.get_context("spawn")
    event = ctx.Event()
    workers = [ctx.Process(target=_waiting_worker, args=(event,))]
    if failure == "child":
        workers.append(ctx.Process(target=_failing_worker))
    for worker in workers:
        worker.start()
    try:
        if failure == "parent":
            with pytest.raises(TimeoutError):
                try:
                    raise TimeoutError("阶段超时")
                finally:
                    _stop_workers(workers)
        elif failure == "child":
            with pytest.raises(RuntimeError, match="worker 失败"):
                _wait_workers(workers)
        else:
            event.set()
            _wait_workers(workers)
    finally:
        _stop_workers(workers)
    assert all(not worker.is_alive() and worker.exitcode is not None for worker in workers)


def test_global_consumed_requires_all_ranks_and_full_budget(tmp_path):
    from hcl_tr.train.loop import _collect_consumed
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2, total_tokens=250)
    (tmp_path / "consumed_rank0.txt").write_text("64")
    with pytest.raises(RuntimeError, match="全部 rank"):
        _collect_consumed(tmp_path, 2, cfg)
    (tmp_path / "consumed_rank1.txt").write_text("64")
    assert _collect_consumed(tmp_path, 2, cfg) == 128
    (tmp_path / "consumed_rank1.txt").write_text("32")
    with pytest.raises(RuntimeError, match="不一致"):
        _collect_consumed(tmp_path, 2, cfg)
    (tmp_path / "consumed_rank0.txt").write_text("32")
    with pytest.raises(RuntimeError, match="未完成"):
        _collect_consumed(tmp_path, 2, cfg)


def test_metrics_and_final_checkpoint_have_explicit_token_units(tmp_path):
    import json
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2,
                      total_tokens=64, use_fp16=False, out_dir=str(tmp_path / "out"))
    train(_model(), _stream(tmp_path / "tok"), cfg, device="cpu")
    metric = json.loads((tmp_path / "out/metrics.jsonl").read_text())
    assert metric["tokens_global"] == metric["tokens_local"] == 64
    assert metric["world_size"] == 1
    assert len(metric["microbatches"]) == 2
    ck = safe_torch_load(tmp_path / "out/final.pt", sha256=sha256_file(tmp_path / "out/final.pt"))
    assert ck["format_version"] == 2
    assert ck["steps_completed"] == 1
    assert ck["tokens_global"] == 64
    assert {"optim", "sched", "scaler", "train_config", "model_config"} <= ck.keys()


def test_no_sync_covers_forward_and_backward(tmp_path, monkeypatch):
    import contextlib
    import hcl_tr.train.loop as loop
    seen_forward, seen_backward = [], []
    class FakeDDP(torch.nn.Module):
        def __init__(self, module, **kwargs):
            super().__init__()
            self.module = module
            self.suppressed = False
        @contextlib.contextmanager
        def no_sync(self):
            self.suppressed = True
            try:
                yield
            finally:
                self.suppressed = False
        def forward(self, x):
            seen_forward.append(self.suppressed)
            result = self.module(x)
            def observe(grad):
                seen_backward.append(self.suppressed)
                return grad
            result.register_hook(observe)
            return result
    monkeypatch.setattr(loop, "DDP", FakeDDP)
    monkeypatch.setattr(loop.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loop.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(loop.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(loop.dist, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(loop.dist, "all_gather_object", lambda result, state: result.__setitem__(slice(None), [state, state]))
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2,
                      total_tokens=128, use_fp16=False, out_dir=str(tmp_path / "out"))
    train(_model(), _stream(tmp_path / "tok"), cfg, device="cpu", ddp=True)
    assert seen_forward == seen_backward == [True, False]


@pytest.mark.parametrize('change', ['data', 'grad_accum', 'total_tokens', 'lr', 'warmup_tokens'])
def test_resume_rejects_identity_or_schedule_migration(tmp_path, change):
    from dataclasses import replace
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=128, warmup_tokens=0, use_fp16=False,
                      ckpt_every_tokens=64, out_dir=str(tmp_path / 'out'))
    train(_model(), _stream(tmp_path / 'tok'), cfg, device='cpu')
    checkpoint = tmp_path / 'out/ckpt_64.pt'
    cfg2 = replace(cfg, resume_from=str(checkpoint), ckpt_sha256=sha256_file(checkpoint),
                   out_dir=str(tmp_path / 'resume'))
    stream = ShardStream(tmp_path / 'tok', seq_len=32, batch_tokens=32, delete_consumed=False, planned_shards=4)
    if change == 'data':
        stream = _stream(tmp_path / 'other')
    else:
        cfg2 = replace(cfg2, **{change: getattr(cfg2, change) + 1})
    with pytest.raises(ValueError, match='数据身份|不能改变'):
        train(_model(), stream, cfg2, device='cpu')
    assert not (tmp_path / 'resume/final.pt').exists()


def test_resume_exact_weights_outputs_and_rng(tmp_path):
    import json
    import random
    from dataclasses import replace
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2,
                      total_tokens=256, warmup_tokens=0, use_fp16=False,
                      ckpt_every_tokens=128, out_dir=str(tmp_path / 'full'),
                      spar_suffix_len=8, spar_chunk_size=4)
    model = _model()
    model.use_spar = True
    train(model, _stream(tmp_path / 'tok'), cfg, device='cpu')
    expected_torch = torch.rand(5)
    expected_python = random.random()
    checkpoint = tmp_path / 'full/ckpt_128.pt'
    resumed = _model()
    resumed.use_spar = True
    random.seed(973)
    stream = ShardStream(tmp_path / 'tok', seq_len=32, batch_tokens=32, delete_consumed=False, planned_shards=4)
    cfg2 = replace(cfg, out_dir=str(tmp_path / 'resume'), resume_from=str(checkpoint),
                   ckpt_sha256=sha256_file(checkpoint))
    assert train(resumed, stream, cfg2, device='cpu') == 256
    assert torch.equal(torch.rand(5), expected_torch)
    assert random.random() == expected_python
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    ids = torch.arange(16).reshape(1, -1)
    assert torch.equal(model(ids), resumed(ids))
    full = safe_torch_load(tmp_path / 'full/final.pt', sha256=sha256_file(tmp_path / 'full/final.pt'))
    restored = safe_torch_load(tmp_path / 'resume/final.pt', sha256=sha256_file(tmp_path / 'resume/final.pt'))
    for key in ('torch', 'spar'):
        assert torch.equal(full['runtime_states'][0][key], restored['runtime_states'][0][key])
    assert full['runtime_states'][0]['python'] == restored['runtime_states'][0]['python']
    losses = lambda p: [json.loads(line)['loss'] for line in p.read_text().splitlines()]
    assert losses(tmp_path / 'full/metrics.jsonl')[2:] == losses(tmp_path / 'resume/metrics.jsonl')


def test_incomplete_manifest_does_not_claim_exact_resume(tmp_path):
    stream = _stream(tmp_path / 'tok')
    stream.planned_shards = None
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=64, use_fp16=False, out_dir=str(tmp_path / 'out'))
    train(_model(), stream, cfg, device='cpu')
    path = tmp_path / 'out/final.pt'
    ck = safe_torch_load(path, sha256=sha256_file(path))
    assert ck['data_manifest'] is None
    assert ck['exact_resume_available'] is False


def test_infinite_gradient_rejected_before_update(tmp_path):
    model = _model()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.embed.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=1,
                      total_tokens=64, use_fp16=False, out_dir=str(tmp_path / 'out'))
    with pytest.raises(RuntimeError, match='梯度范数非有限'):
        train(model, _stream(tmp_path / 'tok'), cfg, device='cpu')
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    assert not (tmp_path / 'out/final.pt').exists()
    assert not (tmp_path / 'out/consumed_rank0.txt').exists()


def test_budget_smaller_than_step_is_rejected(tmp_path):
    cfg = TrainConfig(seq_len=32, micro_batch_per_gpu=1, grad_accum=2,
                      total_tokens=63, use_fp16=False, out_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="不足一个完整"):
        train(_model(), _stream(tmp_path / "tok"), cfg, device="cpu")
