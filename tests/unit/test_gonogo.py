"""NoGo 三条件 + 可用长度 Wilson 规则 + 归因 JSON fixture。

计划书 §8.1 可用长度定义 / §11 自动 NoGo。
"""

from __future__ import annotations

import json
from pathlib import Path

from hcl_tr.eval.gonogo import (
    ALLOWED_ACTIONS,
    FORBIDDEN_ACTIONS,
    build_attribution_json,
    check_gonogo,
    diagnose,
    usable_context_length,
    _wilson_lower,
)


def _baseline():
    return {"baseline_length": 2048}


def test_go_when_all_conditions_pass():
    r = check_gonogo(
        {"absolute_length": 4096, "seed_results": [4096, 4100, 4090]},
        _baseline(), {},
    )
    assert r.verdict == "Go"


def test_nogo_relative_below_80pct():
    r = check_gonogo(
        {"absolute_length": 1500, "seed_results": [1500, 1500, 1500]},
        _baseline(), {},
    )
    # 1500/2048 = 0.73 < 0.8
    assert r.verdict == "NoGo"
    assert any("相对 baseline" in x for x in r.reasons)


def test_nogo_absolute_below_2000():
    r = check_gonogo(
        {"absolute_length": 1900, "relative_to_baseline": 0.95,
         "seed_results": [1900, 1900, 1900]},
        _baseline(), {},
    )
    assert r.verdict == "NoGo"
    assert any("绝对值" in x for x in r.reasons)


def test_nogo_seed_variance_above_20pct():
    r = check_gonogo(
        {"absolute_length": 4096, "relative_to_baseline": 1.5,
         "seed_results": [1000, 4000, 4000]},
        _baseline(), {},
    )
    assert r.verdict == "NoGo"
    assert any("种子方差" in x for x in r.reasons)


def test_missing_relative_and_baseline_is_nogo():
    r = check_gonogo({"absolute_length": 4096}, {"baseline_length": None}, {})
    assert r.verdict == "NoGo"


def test_wilson_lower_bounds():
    # 50/50 全对，下界仍 < 1
    assert _wilson_lower(50, 50) < 1.0
    assert _wilson_lower(50, 50) > 0.9
    # 0 次成功
    assert _wilson_lower(0, 100) == 0.0


def test_usable_length_prereg_rule():
    # 512/1024/2048 达标，4096 掉下 0.9 → 可用长度应为 2048
    acc = {
        512: (100, 100),
        1024: (98, 100),
        2048: (92, 100),
        4096: (80, 100),
        8192: (50, 100),
    }
    assert usable_context_length(acc) == 2048


def test_usable_length_requires_ci_lower_bound():
    # 900/1000 = 0.90 达标但 CI 下界可能 < 0.85（样本量充足时应 > 0.85）
    # 用极小样本来触发 CI 下界不足
    acc = {512: (10, 11)}   # 0.909, 但 n=11 → CI 下界远低于 0.85
    assert usable_context_length(acc) is None


def test_usable_length_all_fail():
    assert usable_context_length({512: (10, 100)}) is None


def test_attribution_json_contains_hashes(tmp_path):
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text("v: 1")
    lock = tmp_path / "SOURCE_LOCK.md"
    lock.write_text("# lock")

    r = check_gonogo(
        {"absolute_length": 1000, "seed_results": [1000, 1000]},
        _baseline(), {},
    )
    payload = build_attribution_json(r, prereg, lock, [{"name": "gdn2", "status": "pass"}])
    assert payload["verdict"] == "NoGo"
    assert len(payload["source_lock_sha256"]) == 64
    assert len(payload["prereg_yaml_sha256"]) == 64
    assert payload["allowed_actions"] == sorted(ALLOWED_ACTIONS)
    # 禁止动作必须被记录、且不与白名单重叠
    assert not (ALLOWED_ACTIONS & FORBIDDEN_ACTIONS)
    json.dumps(payload)  # 可序列化


def test_diagnose_does_not_tune():
    r = check_gonogo({"absolute_length": 1000, "seed_results": [1000, 1000]}, _baseline(), {})
    lines = diagnose(r, [{"name": "gdn2", "status": "pass"}])
    text = "\n".join(lines)
    assert "不执行任何调参" in text or "不自动调参" in text or "不执行" in text
