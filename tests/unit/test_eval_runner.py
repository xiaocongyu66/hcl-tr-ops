"""评测 runner 夹具：口径必须来自预注册 YAML，不能硬编码。"""

from __future__ import annotations

import pytest
import yaml

from hcl_tr.eval.gonogo import GoNoGoResult
from hcl_tr.eval.runner import (
    build_tasks, decide, evaluate, load_prereg, prereg_hash, score,
)

PREREG = "src/hcl_tr/configs/sniah1_ruler_prereg_v1.0.yaml"


def test_load_prereg_reads_yaml_values():
    p = load_prereg(PREREG)
    assert p["lengths"] == [512, 1024, 2048, 4096, 8192]
    assert p["primary_metric"] == "usable_context_length"
    assert p["training_length"] == 8192
    assert len(p["_prereg_hash"]) == 64


def test_load_prereg_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"lengths": [512]}), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需字段"):
        load_prereg(bad)


def test_build_tasks_follows_prereg_grid():
    p = load_prereg(PREREG)
    tasks = build_tasks(p, seed=0)
    n = len(p["lengths"]) * p["samples_per_length"]
    assert len(tasks) == n
    assert {t.length for t in tasks} == set(p["lengths"])


def test_build_tasks_embed_needle_and_answer_matches():
    p = load_prereg(PREREG)
    t = build_tasks(p, seed=1)[0]
    assert t.answer in t.prompt, "needle 必须出现在 prompt 里"
    assert t.answer == t.needle


def test_score_is_case_insensitive_substring():
    assert score("The answer is MAGIC-123456", "magic-123456")
    assert not score("no idea", "magic-123456")


def test_evaluate_all_correct_gives_full_accuracy():
    p = load_prereg(PREREG)
    # 假模型：总答对（从 prompt 里找 needle）
    def perfect(prompt: str) -> str:
        import re
        m = re.search(r"magic-\d{6}", prompt)
        return m.group(0) if m else ""
    r = evaluate(perfect, p)
    for L, (s, n) in r["acc_by_length"].items():
        assert s == n, (L, s, n)


def test_evaluate_all_wrong_gives_zero():
    p = load_prereg(PREREG)
    r = evaluate(lambda _: "wrong", p)
    assert all(s == 0 for s, _ in r["acc_by_length"].values())


def test_decide_uses_prereg_primary_metric_and_rule():
    p = load_prereg(PREREG)
    # 人为构造：<=2048 全对，4096/8192 全错 → 可用长度应为 2048
    acc = {512: (100, 100), 1024: (100, 100), 2048: (100, 100),
           4096: (0, 100), 8192: (0, 100)}
    out = decide({"acc_by_length": acc, "prereg_hash": p["_prereg_hash"]},
                 {"baseline_length": 2048}, p)
    assert out["usable_context_length"] == 2048
    assert out["primary_metric"] == "usable_context_length"
    assert out["prereg_hash"] == p["_prereg_hash"]


def test_decide_no_go_when_below_baseline():
    p = load_prereg(PREREG)
    acc = {512: (100, 100), 1024: (0, 100), 2048: (0, 100), 4096: (0, 100), 8192: (0, 100)}
    out = decide({"acc_by_length": acc, "prereg_hash": p["_prereg_hash"]},
                 {"baseline_length": 4096}, p)   # 512/4096 = 12.5% < 80%
    assert out["verdict"] == "NoGo"
