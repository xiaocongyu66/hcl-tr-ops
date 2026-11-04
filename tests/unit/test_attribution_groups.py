"""归因组配置 fixture（计划书 §9）。

保证：主 4 组进 Go/NoGo、诊断 2 组不进；依赖冻结模块的组被显式标记 blocked。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "docs" / "attribution_groups.json"


def _load():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["groups"]


def test_manifest_exists_and_has_six_groups():
    groups = _load()
    assert len(groups) == 6
    assert {g["id"] for g in groups} == {
        "base_t", "gdn_qed", "sla", "spar_when2think", "spar_only", "when2think_only"
    }


def test_main_groups_in_gonogo():
    groups = {g["id"]: g for g in _load()}
    for gid in ["base_t", "gdn_qed", "sla", "spar_when2think"]:
        assert groups[gid]["in_gonogo"] is True, gid


def test_diagnostic_groups_not_in_gonogo():
    groups = {g["id"]: g for g in _load()}
    for gid in ["spar_only", "when2think_only"]:
        assert groups[gid]["in_gonogo"] is False, gid


def test_only_idac_groups_blocked():
    """只有 IDAC 未实现的目标才允许 blocked；SLA/SPAR 已可用。"""
    groups = {g["id"]: g for g in _load()}
    assert groups["spar_when2think"]["blocked"] and groups["spar_when2think"]["blocked_by"] == "IDAC"
    assert groups["when2think_only"]["blocked"]
    for gid in ["base_t", "gdn_qed", "sla", "spar_only"]:
        assert not groups[gid]["blocked"], f"{gid} 不应 blocked"


def test_sla_group_stacks_on_gdn_qed():
    """计划书 §9：+SLA 是在 +GDN/QED 基础上叠加。"""
    import yaml
    groups = {g["id"]: g for g in _load()}
    sla = yaml.safe_load((REPO_ROOT / groups["sla"]["config_path"]).read_text(encoding="utf-8"))["model"]
    base = yaml.safe_load((REPO_ROOT / groups["gdn_qed"]["config_path"]).read_text(encoding="utf-8"))["model"]
    assert sla["use_sla"] is True and sla["use_qed"] is True
    assert base["use_sla"] is False and base["use_qed"] is True


def test_each_config_file_loads_and_matches_manifest():
    for g in _load():
        p = REPO_ROOT / g["config_path"]
        assert p.exists(), g["config_path"]
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert cfg["attribution_group"] == g["id"]
        assert cfg["in_gonogo"] == g["in_gonogo"]
        if g["blocked"]:
            assert cfg.get("blocked") is True


def test_group_flags_single_variable_diff():
    """每次只改一个变量（计划书 §12 第 7 步）。"""
    groups = {g["id"]: g for g in _load()}
    def flags(gid):
        cfg = yaml.safe_load((REPO_ROOT / groups[gid]["config_path"]).read_text(encoding="utf-8"))
        m = cfg["model"]
        return (m["use_qed"], m["use_sla"], m["use_spar"], m["use_idac"])

    assert flags("base_t") == (False, False, False, False)
    assert flags("gdn_qed") == (True, False, False, False)
    # 层叠：+SLA 在 +GDN/QED 之上；+SPAR/When2Think 在 +SLA 之上
    assert flags("sla") == (True, True, False, False)
    assert flags("spar_when2think") == (True, True, True, True)


def test_all_groups_share_identical_data_and_seed():
    """归因成立的必要条件：除被消融机制外，数据 mix 与种子必须完全一致。

    参数对齐只是必要条件之一；若 LR/数据 mix/种子不同，参数对齐也救不了归因。
    """
    import yaml
    groups = {g["id"]: g for g in _load()}
    sigs = {}
    for gid, g in groups.items():
        c = yaml.safe_load((REPO_ROOT / g["config_path"]).read_text(encoding="utf-8"))
        sigs[gid] = {
            "seed": c.get("seed"),
            "lr": c.get("lr"),
            "batch": c.get("batch"),
            "optimizer": c.get("optimizer"),
            "grad_clip": c.get("grad_clip"),
            "data": c.get("data"),
        }
    ref_id = "gdn_qed"
    for gid, sig in sigs.items():
        assert sig == sigs[ref_id], f"{gid} 的训练配置与 {ref_id} 不一致：{sig} vs {sigs[ref_id]}"


def test_data_mix_hash_is_either_null_everywhere_or_identical():
    """data_mix_hash 要么全为 null（未冻结），要么全部相同且非空（已冻结）。"""
    import yaml
    hashes = {}
    for g in _load():
        c = yaml.safe_load((REPO_ROOT / g["config_path"]).read_text(encoding="utf-8"))
        hashes[g["id"]] = (c.get("data") or {}).get("data_mix_hash")
    vals = set(hashes.values())
    assert len(vals) == 1, f"各组 data_mix_hash 不一致: {hashes}"
