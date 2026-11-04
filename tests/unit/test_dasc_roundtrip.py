"""DASC fixture：模块冻结下的契约测试 + 待解冻的数值测试（skip）。

SOURCE_LOCK: DASC = UNVERIFIED / 冻结。
计划书 §0：DASC 缺失稳定 API 时不得发明压缩格式，应冻结并 NoGo。
"""

from __future__ import annotations

import pytest

from hcl_tr.models.dasc import DASC, DASCNotAvailableError, CompressionReport


def test_dasc_construction_raises_not_available():
    with pytest.raises(DASCNotAvailableError):
        DASC()


def test_dasc_compress_raises_not_available():
    with pytest.raises(DASCNotAvailableError):
        DASC.compress(object())


def test_dasc_decompress_raises_not_available():
    with pytest.raises(DASCNotAvailableError):
        DASC.decompress(b"\x00")


def test_compression_report_schema():
    """部署验收输出的字段契约（计划书 §4.5）。"""
    r = CompressionReport(
        max_abs_err=1e-3, max_rel_err=2e-3,
        per_head_max_abs_err=[1e-3, 2e-3],
        per_channel_max_abs_err=[1e-3, 1e-3, 2e-3],
        tp_imbalance_max=0.05, dtype="bf16", kernel_version="dasc@unverified",
    )
    assert r.max_abs_err == 1e-3
    assert len(r.per_head_max_abs_err) == 2


@pytest.mark.skip(reason="DASC 模块冻结：SGLang 中未定位稳定 API。提供 tag/PR merge commit 后解冻。")
def test_dasc_compress_decompress_error_bound():
    """待解冻：compress→decompress 的逐头/逐通道误差界 + TP ragged balance。"""
    pytest.fail("尚未解冻")
