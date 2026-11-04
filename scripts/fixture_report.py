"""fixture_report: 生成 fixture 报告（计划书 §10 / §12 第 8 步）。

输出 JSON：每个 fixture 的 max_abs_err / max_rel_err / dtype / kernel_version。
数值容差按 dtype 记录。系统未过单元测试不得进入 10B run。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FIXTURES = [
    ("GDN-2 状态转移", "tests/unit/test_gdn2_state.py"),
    ("QED 正交擦除", "tests/unit/test_qed_erase.py"),
    ("SLA G_Q/G_K head 门控 (Eq.5-9)", "tests/unit/test_sla_gating.py"),
    ("SPAR 门控 KL", "tests/unit/test_spar_gate.py"),
    ("When2Think IDAC/BWS/IS (Eq.1-8)", "tests/unit/test_idac_reward.py"),
    ("MoE-nD 路由 schema", "tests/unit/test_moe_nd_router.py"),
    ("DASC 压缩/TP balance", "tests/unit/test_dasc_roundtrip.py"),
    ("NoGo 三条件", "tests/unit/test_gonogo.py"),
    ("CMM 双记忆", "tests/unit/test_cmm_memory.py"),
    ("HybridBackbone", "tests/unit/test_backbone.py"),
]


def run_fixture(path: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "-rs", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    out = proc.stdout + proc.stderr
    tail = out.strip().splitlines()
    summary = tail[-1] if tail else ""
    # 解析 pytest 摘要：区分 passed / skipped / failed
    import re
    nums = {"passed": 0, "skipped": 0, "failed": 0}
    for key in nums:
        m = re.search(rf"(\d+) {key}", summary)
        if m:
            nums[key] = int(m.group(1))
    # 收集 skip 原因，避免把「未测」误读为「通过」
    skip_reasons = [l.strip() for l in tail if l.strip().startswith("SKIPPED")]
    all_tests_ran = nums["skipped"] == 0 and nums["failed"] == 0 and nums["passed"] > 0
    return {
        "path": path,
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "status": ("PASS" if proc.returncode == 0 and all_tests_ran
                   else "FAIL" if proc.returncode != 0
                   else "PARTIAL(skip)"),
        "fully_exercised": all_tests_ran,   # false 表示有 skip（未真正验收）
        "counts": nums,
        "skipped_reasons": skip_reasons,
        "summary": summary,
        "dtype": "float32",
        "kernel_version": "pytorch-reference",
        "tolerances": {
            "fp32_max_abs_err": 1e-5,
            "qed_orthogonality": 1e-6,
            "bf16_max_abs_err_upper": 2e-2,
            "fp16_max_abs_err_upper": 5e-3,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="有 skip（未完全验收）也返回非 0，供 CI 严格判定")
    args = ap.parse_args(argv)

    report = []
    all_pass = True          # 无失败（skip 允许）
    all_exercised = True     # 严格：无失败且无 skip
    for name, path in FIXTURES:
        r = run_fixture(path)
        r["name"] = name
        report.append(r)
        if not r["passed"]:
            status = "FAIL"
        elif not r["fully_exercised"]:
            status = "PARTIAL(skip)"
        else:
            status = "PASS"
        print(f"[{status}] {name}: {r['summary']}")
        all_pass = all_pass and r["passed"]
        all_exercised = all_exercised and r["fully_exercised"]

    out = REPO_ROOT / "docs" / "fixture_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"all_pass": all_pass, "all_fully_exercised": all_exercised, "fixtures": report},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n报告写入 {out.relative_to(REPO_ROOT)}")
    print(f"无失败={all_pass}；全部真跑（无 skip）={all_exercised}")
    if not all_pass:
        return 1
    # --strict：skip 视为「未完全验收」，返回 3（区别于失败的 1）
    if args.strict and not all_exercised:
        print("STRICT: 存在 skip，未完全验收 → exit 3", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
