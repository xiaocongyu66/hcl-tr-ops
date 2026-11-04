"""DASC：recurrent checkpoint 压缩（部署期）。

SOURCE_LOCK: DASC = **UNVERIFIED / 冻结**
  - 计划书来源 "SGLang 中的 DASC release tag / PR merge commit" 未定位（1917 个 head 分支无 DASC）。
  - 红线（计划书 §0）：**不得在 DASC 缺失稳定 API 时现场发明压缩格式；应冻结模块并 NoGo。**

本文件因此**不实现压缩算法**，只提供：
  1. DASC 部署验收的接口契约（compress/decompress 签名 + 误差界指标）。
  2. 一个显式的 `NotAvailableError`，保证任何调用方在 DASC 未解冻前无法静默降级。
  3. 接口契约 fixture（schema 级），压缩数值 fixture 标记 skip。

验证对象（计划书 §4.5）：GDN-2/CMM **recurrent checkpoint** 的 ragged 压缩，**不是 attention KV**。
"""

from __future__ import annotations

from dataclasses import dataclass


class DASCNotAvailableError(RuntimeError):
    """DASC 模块冻结时的显式失败。绝不返回经验压缩率或发明格式。"""


@dataclass(frozen=True)
class CompressionReport:
    """部署验收 fixture 输出的统一结构（计划书 §4.5）。"""
    max_abs_err: float
    max_rel_err: float
    per_head_max_abs_err: list[float]
    per_channel_max_abs_err: list[float]
    tp_imbalance_max: float   # TP rank 间 ragged balance 不均衡上限
    dtype: str
    kernel_version: str


class DASC:
    """DASC 接口壳。任何真实调用都会抛 DASCNotAvailableError（模块冻结）。"""

    def __init__(self):
        raise DASCNotAvailableError(
            "DASC 未解冻：SGLang 中未定位到稳定 API。"
            "按 SOURCE_LOCK 红线，禁止现场发明压缩格式。需提供精确 tag/PR merge commit。"
        )

    @staticmethod
    def compress(state: object) -> bytes:
        raise DASCNotAvailableError("DASC.compress 不可用（模块冻结）")

    @staticmethod
    def decompress(blob: bytes) -> object:
        raise DASCNotAvailableError("DASC.decompress 不可用（模块冻结）")
