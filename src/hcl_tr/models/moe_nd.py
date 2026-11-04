"""MoE-nD：逐层 KV 压缩路由表 + 贪心校准。

SOURCE_LOCK: MoE-nD LOCKED-PAPER, arXiv:2604.17695
  第 2 页 "Contribution"：每个压缩轴（eviction_ratio, K-precision, V-precision）作为 expert，
    离线校准 router 在单一全局 memory budget 下逐层选混合。
  第 2 页：per-layer **eviction** routing 是新杠杆；per-layer quantization routing 单独 ≈0。
  router = "greedy solver over an offline-measured per-layer sensitivity table"。

实现范围：路由表 schema 校验 + 贪心校准器（小样本）。**仅部署验收，不进训练期 S-NIAH-1 归因。**
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


REQUIRED_FIELDS = {
    "layer_id": int,
    "eviction_ratio": float,
    "k_bits": int,
    "v_bits": int,
    "expected_quality_loss": float,
    "memory_bytes": int,
}


@dataclass
class RoutingEntry:
    layer_id: int
    eviction_ratio: float
    k_bits: int
    v_bits: int
    expected_quality_loss: float
    memory_bytes: int

    def validate(self) -> None:
        d = asdict(self)
        for k, typ in REQUIRED_FIELDS.items():
            if k not in d:
                raise ValueError(f"routing entry 缺字段 {k}")
            if not isinstance(d[k], typ):
                raise TypeError(f"{k} 期望 {typ.__name__}，实际 {type(d[k]).__name__}")
        if not (0.0 <= self.eviction_ratio <= 1.0):
            raise ValueError(f"eviction_ratio 越界: {self.eviction_ratio}")
        if self.k_bits not in (4, 8, 16) or self.v_bits not in (4, 8, 16):
            raise ValueError(f"bits 只允许 4/8/16，got K={self.k_bits} V={self.v_bits}")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False)


def greedy_calibrate(
    sensitivity_table: list[dict],
    global_budget_bytes: int,
) -> list[RoutingEntry]:
    """贪心校准：逐层选择 (eviction_ratio, k_bits, v_bits)，满足全局 memory 预算。

    sensitivity_table 每项:
      {"layer_id": int,
       "candidates": [{"eviction_ratio", "k_bits", "v_bits", "expected_quality_loss",
                       "memory_bytes"}, ...]}
    贪心按 quality_loss / memory_saved 比排序选候选；预算耗尽则退到最省内存候选。
    """
    chosen: list[RoutingEntry] = []
    remaining = global_budget_bytes
    # 先把每层的候选按 "省内存性价比" 排序
    for row in sensitivity_table:
        cands = row["candidates"]
        # 性价比：每字节节省换取的 quality 损失越低越好
        cands = sorted(cands, key=lambda c: (c["expected_quality_loss"] / max(1, c["memory_bytes"])))
        pick = None
        for c in cands:
            if c["memory_bytes"] <= remaining:
                pick = c
                break
        if pick is None:
            pick = min(cands, key=lambda c: c["memory_bytes"])
        remaining -= pick["memory_bytes"]
        entry = RoutingEntry(layer_id=row["layer_id"], **pick)
        entry.validate()
        chosen.append(entry)
    return chosen


def apply_atomically(entries: list[RoutingEntry], apply_fn, rollback_fn) -> bool:
    """原子应用：全部成功才提交；任一失败则逆序回滚已应用的层。

    计划书 §4.6：推理 patch 原子应用；失败即回滚；不允许半层异构半层均匀。
    `rollback_fn` 必须能把一层恢复成应用前的配置——没有它就无法满足原子性，
    因此本函数拒绝在缺失时静默通过。
    """
    if rollback_fn is None:
        raise ValueError("apply_atomically 需要 rollback_fn：缺少回滚无法保证原子性")

    applied: list[RoutingEntry] = []
    try:
        for e in entries:
            apply_fn(e)
            applied.append(e)
        return True
    except Exception:
        rollback_errors = []
        for e in reversed(applied):
            try:
                rollback_fn(e)
            except Exception as rb_err:      # 回滚失败不能吞掉，否则就是半层异构
                rollback_errors.append((e.layer_id, rb_err))
        if rollback_errors:
            raise RuntimeError(f"回滚失败，层仍处于半应用状态: {rollback_errors}")
        return False
