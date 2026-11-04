"""gen_attribution_groups: 生成归因组配置（计划书 §9 / §12 第 8 步）。

主 4 组（进 Go/NoGo）:
  Base-T              100M Transformer（= 冻结 baseline，不重训，指向其 checkpoint）
  +GDN/QED            GDN-2 + QED + 3:1 全注意力
  +SLA                在 +GDN/QED 上加 SLA
  +SPAR/When2Think    在 +SLA 上加 SPAR + IDAC
诊断 2 组（不进 Go/NoGo）:
  +SPAR only
  +When2Think only

注意：+SPAR/When2Think 与 +When2Think only 都依赖 When2Think（UNVERIFIED 冻结），
因此生成的配置会显式标记 blocked=true，train.py 会拒绝启动这些组。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = REPO_ROOT / "src" / "hcl_tr" / "configs" / "attribution"
OUT = REPO_ROOT / "docs" / "attribution_groups.json"

GROUPS = [
    {"id": "base_t", "name": "Base-T", "in_gonogo": True,
     "config": {"use_qed": False, "use_sla": False, "use_spar": False, "use_idac": False,
                "arch": "transformer"},
     "note": "100M Transformer baseline，指向冻结 checkpoint，禁止重训"},
    {"id": "gdn_qed", "name": "+GDN/QED", "in_gonogo": True,
     "config": {"use_qed": True, "use_sla": False, "use_spar": False, "use_idac": False}},
    {"id": "sla", "name": "+SLA", "in_gonogo": True,
     # 计划书 §9：「在 +GDN/QED 上加 SLA」→ 叠在 GDN-2 上，参数与 +GDN/QED 对齐
     "config": {"use_qed": True, "use_sla": True, "use_spar": False, "use_idac": False}},
    {"id": "spar_when2think", "name": "+SPAR/When2Think", "in_gonogo": True,
     "config": {"use_qed": True, "use_sla": True, "use_spar": True, "use_idac": True},
     "blocked_by": "IDAC", "note": "IDAC 的 RLVR 训练目标未实现（§12 阶段2）；SPAR 已实现"},
    {"id": "spar_only", "name": "+SPAR only", "in_gonogo": False,
     "config": {"use_qed": True, "use_sla": False, "use_spar": True, "use_idac": False},
     "note": "诊断：相对 +GDN/QED 只加 SPAR"},
    {"id": "when2think_only", "name": "+When2Think only", "in_gonogo": False,
     "config": {"use_qed": True, "use_sla": False, "use_spar": False, "use_idac": True},
     "blocked_by": "IDAC", "note": "诊断：相对 +GDN/QED 只加 IDAC（未实现）"},
]

BASE_STAGE0 = yaml.safe_load((REPO_ROOT / "src/hcl_tr/configs/stage0_mvp1a.yaml").read_text())


def main() -> int:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for g in GROUPS:
        cfg = json.loads(json.dumps(BASE_STAGE0))  # deep copy
        cfg["model"].update({
            "use_qed": g["config"]["use_qed"],
            "use_sla": g["config"]["use_sla"],
            "use_spar": g["config"]["use_spar"],
            "use_idac": g["config"]["use_idac"],
        })
        # Base-T 必须是纯 Transformer（计划书 §9），不能沿用 3:1 混合骨干结构。
        # 16 层 attn_ratio=1 → 100.10M / non_embed 75.52M，与混合骨干参数对齐。
        if g["config"].get("arch") == "transformer":
            cfg["model"]["n_layer"] = 16
            cfg["model"]["attn_ratio"] = 1
        cfg["attribution_group"] = g["id"]
        cfg["in_gonogo"] = g["in_gonogo"]
        if "blocked_by" in g:
            cfg["blocked"] = True
            cfg["blocked_by"] = g["blocked_by"]
        path = CFG_DIR / f"{g['id']}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

        manifest.append({
            "id": g["id"], "name": g["name"],
            "in_gonogo": g["in_gonogo"],
            "config_path": str(path.relative_to(REPO_ROOT)),
            "blocked": "blocked_by" in g,
            "blocked_by": g.get("blocked_by"),
            "note": g.get("note", ""),
        })
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    OUT.write_text(json.dumps({"groups": manifest}, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"\nmanifest -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
