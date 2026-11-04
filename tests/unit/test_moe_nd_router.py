"""MoE-nD 路由 schema + 贪心校准小样本 + 原子应用/回滚。

SOURCE_LOCK: MoE-nD LOCKED-PAPER, arXiv:2604.17695（文档页码范围已核）
仅部署验收，不进训练期归因。
"""

from __future__ import annotations

import json

from hcl_tr.models.moe_nd import (
    RoutingEntry,
    apply_atomically,
    greedy_calibrate,
)


def test_routing_schema_valid_and_json():
    e = RoutingEntry(layer_id=0, eviction_ratio=0.1, k_bits=8, v_bits=8,
                     expected_quality_loss=0.02, memory_bytes=123456)
    e.validate()
    j = json.loads(e.to_json())
    assert set(j) == {"layer_id", "eviction_ratio", "k_bits", "v_bits",
                      "expected_quality_loss", "memory_bytes"}
    assert j["k_bits"] == 8


def test_routing_rejects_bad_bits():
    e = RoutingEntry(0, 0.1, 3, 8, 0.01, 1000)
    try:
        e.validate()
        assert False, "应拒绝非法 bits"
    except ValueError:
        pass


def test_routing_rejects_ratio_out_of_range():
    e = RoutingEntry(0, 1.5, 8, 8, 0.01, 1000)
    try:
        e.validate()
        assert False, "应拒绝越界 eviction_ratio"
    except ValueError:
        pass


def _sensitivity_table():
    return [
        {"layer_id": 0, "candidates": [
            {"eviction_ratio": 0.0, "k_bits": 16, "v_bits": 16, "expected_quality_loss": 0.0, "memory_bytes": 1000},
            {"eviction_ratio": 0.5, "k_bits": 8, "v_bits": 8, "expected_quality_loss": 0.02, "memory_bytes": 300},
        ]},
        {"layer_id": 1, "candidates": [
            {"eviction_ratio": 0.0, "k_bits": 16, "v_bits": 16, "expected_quality_loss": 0.0, "memory_bytes": 1000},
            {"eviction_ratio": 0.3, "k_bits": 8, "v_bits": 8, "expected_quality_loss": 0.01, "memory_bytes": 400},
        ]},
    ]


def test_greedy_respects_global_budget():
    table = _sensitivity_table()
    budget = 700   # 只能容纳两个压缩候选（300+400）
    entries = greedy_calibrate(table, budget)
    assert len(entries) == 2
    total = sum(e.memory_bytes for e in entries)
    assert total <= budget


def test_greedy_falls_back_to_min_memory_when_budget_tight():
    table = _sensitivity_table()
    entries = greedy_calibrate(table, global_budget_bytes=100)  # 极紧预算
    assert all(e.memory_bytes <= 400 for e in entries)


def _entries(n=3):
    return [RoutingEntry(i, 0.1, 8, 8, 0.01, 100) for i in range(n)]


class _LayerState:
    """模拟真实层配置：用来证明回滚把状态真正恢复，而不是只返回 False。"""

    def __init__(self):
        self.config = {i: "original" for i in range(3)}

    def apply(self, e):
        self.config[e.layer_id] = f"evicted@{e.eviction_ratio}"

    def rollback(self, e):
        self.config[e.layer_id] = "original"


def test_atomic_apply_success_commits_all_layers():
    state = _LayerState()
    ok = apply_atomically(_entries(3), state.apply, state.rollback)
    assert ok
    assert all(v.startswith("evicted@") for v in state.config.values())


def test_atomic_apply_rolls_back_on_failure():
    """第 2 层失败 → 已应用的第 0、1 层必须被真正回滚到 original。"""
    state = _LayerState()
    journal = []

    def apply_fn(e):
        if e.layer_id == 2:
            journal.append(("fail", 2))
            raise RuntimeError("layer 2 失败")
        state.apply(e)
        journal.append(("apply", e.layer_id))

    def rollback_fn(e):
        state.rollback(e)
        journal.append(("rollback", e.layer_id))

    ok = apply_atomically(_entries(3), apply_fn, rollback_fn)
    assert not ok
    # 关键断言：回滚确实发生，且顺序为逆序
    assert journal == [("apply", 0), ("apply", 1), ("fail", 2),
                       ("rollback", 1), ("rollback", 0)]
    # 关键断言：状态真的被恢复，不存在半层异构
    assert state.config == {0: "original", 1: "original", 2: "original"}


def test_atomic_apply_requires_rollback_fn():
    """缺失回滚函数必须显式失败，不得静默假装原子。"""
    entries = _entries(2)
    try:
        apply_atomically(entries, lambda e: None, None)
        assert False, "应拒绝缺失 rollback_fn"
    except ValueError:
        pass


def test_atomic_apply_surfaces_rollback_failure():
    """回滚本身失败不能吞掉——那正是半层异构场景。"""
    state = _LayerState()

    def apply_fn(e):
        if e.layer_id == 1:
            raise RuntimeError("apply 失败")
        state.apply(e)

    def bad_rollback(e):
        raise RuntimeError("回滚也失败")

    try:
        apply_atomically(_entries(2), apply_fn, bad_rollback)
        assert False, "回滚失败应抛出"
    except RuntimeError as e:
        assert "回滚失败" in str(e)
