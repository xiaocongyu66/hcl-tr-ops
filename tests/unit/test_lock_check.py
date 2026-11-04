"""SOURCE_LOCK 闸门自测：确保 lock_check 能真实拦截违规，而非假绿。"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import lock_check  # noqa: E402


def _write_lock(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "SOURCE_LOCK.md"
    p.write_text(body, encoding="utf-8")
    return p


BASIC_LOCK = """# SOURCE_LOCK

## 模块：GDN-2
- Status: `LOCKED`
- Commit: `de8476de0fecae08fc16f0bf3154900244c61c73`

## 模块：DASC
- Status: `UNVERIFIED`
"""


def test_parse_canonical_name_strips_parenthetical(tmp_path):
    lock = _write_lock(tmp_path, """# x
## 模块：CMM (Continuous Memory Machines)
- Status: `LOCKED`
- Commit: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`
""")
    entries = lock_check.parse_source_lock(lock)
    assert "CMM" in entries
    assert entries["CMM"].status == "LOCKED"
    assert "4a6c9c3a" in entries["CMM"].fields["Commit"]


def test_frozen_status_parsed(tmp_path):
    lock = _write_lock(tmp_path, BASIC_LOCK)
    entries = lock_check.parse_source_lock(lock)
    assert entries["DASC"].status == "UNVERIFIED"


def test_commit_drift_detected(tmp_path, monkeypatch):
    lock = _write_lock(tmp_path, BASIC_LOCK)
    entries = lock_check.parse_source_lock(lock)

    # 伪造一个 commit 与锁文件不一致的 vendor 目录
    vdir = tmp_path / "vendor" / "gdn2"
    vdir.mkdir(parents=True)
    (vdir / "LOCKED_COMMIT").write_text("0" * 40)

    monkeypatch.setattr(lock_check, "REPO_ROOT", tmp_path)
    errors = lock_check.check_vendor_lock(entries)
    assert any("漂移" in e for e in errors), errors


def test_bad_lock_status_for_frozen_module_detected(tmp_path):
    lock = _write_lock(
        tmp_path,
        """# x
## 模块：DASC
- Status: `LOCKED`
## 模块：When2Think
- Status: `UNVERIFIED`
""",
    )
    entries = lock_check.parse_source_lock(lock)
    errors = lock_check.check_frozen(entries)
    assert any("DASC" in e for e in errors), errors


def test_real_repo_gate_passes():
    """真实仓库的 SOURCE_LOCK 闸门必须通过（Exit Criteria 第一项）。"""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "lock_check.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
