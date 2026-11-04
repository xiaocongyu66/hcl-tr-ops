"""预取/消费 fixture：本地假分片验证 ShardStream 的原子性、顺序、清理。

不依赖网络——直接构造分片文件，验证训练侧消费逻辑。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from hcl_tr.data.prefetch import ShardStream, write_shard


def test_write_shard_is_atomic_with_ready_marker(tmp_path):
    toks = np.arange(100, dtype=np.uint16)
    p = write_shard(tmp_path, 0, toks)
    assert p.exists()
    assert (tmp_path / "shard_000000.ready").exists()
    # 没有残留 .tmp
    assert not list(tmp_path.glob("*.tmp"))


def test_stream_yields_in_order_and_correct_shape(tmp_path):
    for i in range(3):
        write_shard(tmp_path, i, np.full(64, i, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=32, delete_consumed=False)
    it = s.batches(stop_after_tokens=64)
    b0 = next(it)
    assert b0.shape == (4, 8)          # 32 tokens / 8 = 4 seq
    assert b0[0, 0].item() == 0        # 顺序：先消费 shard 0
    b1 = next(it)
    assert b1.shape == (4, 8)


def test_shards_without_ready_marker_are_ignored(tmp_path):
    write_shard(tmp_path, 0, np.arange(64, dtype=np.uint16))
    # 伪造一个无 .ready 的分片（模拟写到一半）
    np.save(tmp_path / "shard_000001.npy", np.arange(64, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=64, delete_consumed=False)
    ready = s._ready_shards()
    assert len(ready) == 1
    assert ready[0].name == "shard_000000.npy"


def test_consumed_shards_are_deleted(tmp_path):
    write_shard(tmp_path, 0, np.arange(32, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=32, delete_consumed=True)
    it = s.batches(stop_after_tokens=32)
    next(it)
    # 消费后应触发清理（下一次 _evict）；这里显式调用验证行为
    s._evict()
    assert not list(tmp_path.glob("shard_*.npy"))


def test_wait_for_shard_times_out(tmp_path):
    import pytest
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=8)
    with pytest.raises(TimeoutError):
        s.wait_for_shard(timeout=0.5)


def test_rank_aware_sharding_no_overlap(tmp_path):
    """双卡：rank0/rank1 分片互补且不重叠。"""
    for i in range(6):
        write_shard(tmp_path, i, np.arange(32, dtype=np.uint16))
    r0 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=0, world_size=2, planned_shards=6)
    r1 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=1, world_size=2, planned_shards=6)
    s0 = {p.name for p in r0._ready_shards()}
    s1 = {p.name for p in r1._ready_shards()}
    assert s0 & s1 == set(), f"分片重叠: {s0 & s1}"
    assert s0 | s1 == {f"shard_{i:06d}.npy" for i in range(6)}


def test_rank_aware_even_split(tmp_path):
    """等大分片时，两 rank 分到的 token 数必须相等且无重叠。"""
    for i in range(4):
        write_shard(tmp_path, i, np.arange(32, dtype=np.uint16))
    r0 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=0, world_size=2, planned_shards=4)
    r1 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=1, world_size=2, planned_shards=4)
    a = [p.name for p in r0._ready_shards()]
    b = [p.name for p in r1._ready_shards()]
    assert not (set(a) & set(b)), "两 rank 不得重叠"
    assert len(a) == len(b) == 2
    assert sorted(a + b) == [f"shard_00000{i}.npy" for i in range(4)]


def test_evict_never_deletes_unconsumed(tmp_path):
    """超预算时不得删除未消费分片（否则静默丢数据）。"""
    import warnings
    for i in range(6):
        write_shard(tmp_path, i, np.arange(1000, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=8,
                    max_cache_bytes=1, delete_consumed=True, rank=0, world_size=2,
                    planned_shards=6)
    before = {p.name for p in s._ready_shards()}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s._evict()
    after = {p.name for p in s._ready_shards()}
    assert after == before, f"未消费分片被删: {before - after}"
    assert any("超过上限" in str(x.message) for x in w), "应发出超预算警告"


def test_evict_deletes_only_consumed(tmp_path):
    for i in range(4):
        write_shard(tmp_path, i, np.arange(64, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=8, delete_consumed=True)
    first = s._ready_shards()[0]
    s._consumed.append(first)
    s._evict()
    assert not first.exists()
    assert len(s._ready_shards()) == 3


def test_shard_partition_balances_tokens_across_ranks(tmp_path):
    """根因修复：大小不均的分片也必须按 token 均衡分配给各 rank。"""
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    # 制造极不均：小、大、小、大……最后一个很小
    sizes = [100, 4000, 100, 4000, 50]
    for i, n in enumerate(sizes):
        write_shard(tmp_path, i, np.arange(n, dtype=np.uint16))
    toks = {0: 0, 1: 0}
    names = {0: [], 1: []}
    for r in (0, 1):
        s = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=r, world_size=2, planned_shards=5)
        got = s._ready_shards()
        names[r] = [p.name for p in got]
        toks[r] = sum(p.stat().st_size for p in got)
    total = sum(p.stat().st_size for p in (tmp_path).glob("shard_*.npy"))
    assert toks[0] + toks[1] == total, f"分片必须被完整覆盖: {toks} vs {total}"
    assert not (set(names[0]) & set(names[1])), "两 rank 不得重叠"
    assert abs(toks[0] - toks[1]) / total < 0.10, f"两 rank token 数相差过大: {toks}"


def test_shard_partition_covers_all_shards_exactly_once(tmp_path):
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    for i in range(7):
        write_shard(tmp_path, i, np.arange(10 + i * 7, dtype=np.uint16))
    seen = []
    for r in (0, 1, 2):
        s = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=r, world_size=3, planned_shards=7)
        seen += [p.name for p in s._ready_shards()]
    assert len(seen) == len(set(seen)) == 7, f"分片重复或遗漏: {seen}"


def test_multi_rank_requires_planned_shards(tmp_path):
    """多 rank 不给 planned_shards 必须报错，不能靠动态归属蒙混。"""
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    write_shard(tmp_path, 0, np.arange(64, dtype=np.uint16))
    s = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=0, world_size=2)
    try:
        s._ready_shards()
        assert False, "应当报错"
    except ValueError as e:
        assert "planned_shards" in str(e)


def test_partition_stable_when_new_shards_appear(tmp_path):
    """流式写出新分片时，已分配分片的归属不得改变（防重复/丢失）。"""
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    for i in range(2):
        write_shard(tmp_path, i, np.arange(64, dtype=np.uint16))
    r0 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=0, world_size=2, planned_shards=4)
    r1 = ShardStream(tmp_path, seq_len=8, batch_tokens=8, rank=1, world_size=2, planned_shards=4)
    before0 = set(p.name for p in r0._ready_shards())
    before1 = set(p.name for p in r1._ready_shards())
    for i in (2, 3):                       # 后到的分片
        write_shard(tmp_path, i, np.arange(64, dtype=np.uint16))
    after0 = set(p.name for p in r0._ready_shards())
    after1 = set(p.name for p in r1._ready_shards())
    assert before0 <= after0 and before1 <= after1, "归属发生漂移"
    assert not (after0 & after1), "两 rank 重叠"


def test_write_shard_accepts_str_path(tmp_path):
    """write_shard 必须接受字符串路径（曾传 str 导致 AttributeError: no mkdir）。"""
    import numpy as np
    from hcl_tr.data.prefetch import write_shard
    d = str(tmp_path / "sub" / "tokens")     # 故意用 str，且目录不存在
    p = write_shard(d, 0, np.arange(16, dtype=np.uint16))
    assert p.exists() and p.with_suffix(".ready").exists()


def _collect_batches(tmp_path, rank, world, stop, mb=1, seq=8, n_shard=3, toks=32,
                     wait_timeout=3.0):
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    s = ShardStream(tmp_path, seq_len=seq, batch_tokens=mb * seq, rank=rank,
                    world_size=world, delete_consumed=False, ddp=True,
                    wait_timeout=wait_timeout)
    out = []
    for b in s.batches(stop_after_tokens=stop):
        out.append(b.clone())
    return out


def test_ddp_ranks_get_same_number_of_batches(tmp_path):
    """根因修复：分片数不是 world 整数倍时，两个 rank 步数也必须相同。

    旧实现（3 片给 2 卡）会让 rank1 先耗尽并阻塞 → NCCL SIGABRT。
    """
    import numpy as np
    for i in range(3):                       # 3 片，不能均分给 2 卡
        write_shard(tmp_path, i, np.arange(32, dtype=np.uint16))
    # 3 片 × 32 = 96 token；world=2、每步本地 8 token → 全局每步 16
    a = _collect_batches(tmp_path, 0, 2, stop=96)
    b = _collect_batches(tmp_path, 1, 2, stop=96)
    assert len(a) == len(b), f"两 rank 步数不同: {len(a)} vs {len(b)}"
    assert len(a) > 0


def test_ddp_ranks_see_disjoint_slices_of_same_stream(tmp_path):
    """两 rank 的数据必须来自同一条全局流的不同切片，不重复。"""
    import numpy as np
    vals = np.arange(64, dtype=np.uint16)
    write_shard(tmp_path, 0, vals)
    a = _collect_batches(tmp_path, 0, 2, stop=32)
    b = _collect_batches(tmp_path, 1, 2, stop=32)
    # world=2、每步本地 8 → 每全局步 16；32 本地 token = 4 步 = 64 全局 token
    flat_a = np.concatenate([x.numpy().ravel() for x in a])
    flat_b = np.concatenate([x.numpy().ravel() for x in b])
    assert set(flat_a.tolist()) & set(flat_b.tolist()) == set(), "两 rank 数据重复"
    # 合起来正好覆盖源数据的前 2*mb*seq*world 个 token
    assert len(flat_a) + len(flat_b) == 64


def test_ddp_global_stream_consumes_all_tokens(tmp_path):
    """根因修复：world=2 时全局流必须能吃掉分片里的全部 token，不能少读。

    曾在 10M 跑里只训到 3.3M——因为 ddp 下 rank0 删分片，慢的 rank 提前没数据。
    """
    import numpy as np
    from hcl_tr.data.prefetch import ShardStream
    per_shard, n_shard = 1000, 5          # 共 5000 token
    for i in range(n_shard):
        write_shard(tmp_path, i, np.arange(per_shard, dtype=np.uint16) + i * per_shard)
    totals = {}
    for r in (0, 1):
        s = ShardStream(tmp_path, seq_len=8, batch_tokens=16, rank=r, world_size=2,
                        delete_consumed=True, ddp=True, wait_timeout=3.0)
        n = 0
        for b in s.batches(stop_after_tokens=10**9):
            n += b.numel()
        totals[r] = n
    # 每 rank 至少应读到接近一半（留一个全局步的余数）
    assert totals[0] >= 5000 // 2 - 16, totals
    assert totals[1] >= 5000 // 2 - 16, totals
