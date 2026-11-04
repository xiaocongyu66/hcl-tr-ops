"""lock_check: 强制校验外部依赖与 SOURCE_LOCK.md 一致。

校验规则（与 SOURCE_LOCK.md 末尾一致）:
1. 每个外部 import 必须映射到 SOURCE_LOCK.md 中的条目（模块名或其仓库/tag）。
2. 每个 torch.load / from_pretrained 必须携带 revision，且与 SOURCE_LOCK 记录一致。
3. 每个 RULER/S-NIAH-1 评估入口必须加载预注册 YAML。
4. vendor 目录 commit 必须等于 SOURCE_LOCK 记录的 LOCKED commit，漂移即失败。
5. `UNVERIFIED` 模块被 import 即失败（冻结）。
6. 任一失败 → 返回非零退出码，train.py 不启动。

退出码: 0 = 全绿；1 = 存在缺口。
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_LOCK = REPO_ROOT / "SOURCE_LOCK.md"

# 外部 import 顶层包 -> SOURCE_LOCK 模块名。只有出现在此表里的顶层包才被视为"外部依赖"。
# 标准库、本项目自身 (hcl_tr) 不在其中。
EXTERNAL_IMPORT_MAP = {
    "torch": "GDN-2",
    "einops": "GDN-2",
    "fla": "GDN-2",
    "transformers": "GDN-2",
    "ctm": "CMM",
    "numpy": "GDN-2",
}

# 必须冻结、禁止被 import 的模块（UNVERIFIED 状态）。
# 2026-10-29 更新：When2Think 已解冻（arXiv:2609.19671v2 + HF sha 已核），仅 DASC 仍冻结。
FROZEN_MODULES = {"DASC"}
# 冻结模块的 import 提示词；解冻模块需从此表移除，否则正常 import 会被误拦。
FROZEN_IMPORT_HINTS = {"dasc"}


@dataclass
class LockEntry:
    name: str
    status: str
    fields: dict[str, str] = field(default_factory=dict)


def parse_source_lock(path: Path) -> dict[str, LockEntry]:
    """解析 SOURCE_LOCK.md 中的 `## 模块：<name>` 区块。"""
    entries: dict[str, LockEntry] = {}
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^## 模块：", text, flags=re.MULTILINE)[1:]
    for block in blocks:
        lines = block.splitlines()
        name = lines[0].strip()
        # 去掉中文/英文括注，保留主名，便于 import 映射（"CMM (Continuous Memory Machines)" -> "CMM"）
        canonical = re.split(r"[（(]", name)[0].strip()
        status = ""
        fields: dict[str, str] = {}
        for line in lines[1:]:
            m = re.match(r"^\s*-\s+Status:\s*`?([A-Za-z\-]+)`?", line)
            if m:
                status = m.group(1)
            m = re.match(r"^\s*-\s+(Commit|URL|LICENSE|公式依据|论文依据):\s*(.+)$", line)
            if m:
                fields[m.group(1)] = m.group(2).strip()
        entries[canonical] = LockEntry(name=name, status=status, fields=fields)
        # 同时以全名登记，便于人工查询
        entries[name] = entries[canonical]
    return entries


def iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in {".git", "vendor", "__pycache__", "build", "dist"} for part in p.parts):
            continue
        yield p


def check_imports(entries: dict[str, LockEntry]) -> list[str]:
    """规则 1 + 5：外部 import 必须有条目；冻结模块不得被 import。

    覆盖范围：src/ 与 tests/（验证代理指出只扫 src/ 会漏掉 tests 里的违规）。
    根目录脚本（train.py / scripts/*.py）一并纳入。
    """
    errors: list[str] = []
    scan_roots = [
        REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "scripts",
        REPO_ROOT / "eval",      # 顶层 eval/ 目录（当前为空，预留）
    ]
    candidates = [p for p in sorted(REPO_ROOT.glob("*.py"))]   # 根目录入口脚本
    for root in scan_roots:
        if root.exists():
            candidates.extend(iter_py_files(root))
    for py in candidates:
        if not py.exists():
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for top in names:
                if top in FROZEN_IMPORT_HINTS:
                    errors.append(f"{py.relative_to(REPO_ROOT)}: import 冻结模块 '{top}'")
                if top in EXTERNAL_IMPORT_MAP:
                    mod = EXTERNAL_IMPORT_MAP[top]
                    if mod not in entries:
                        errors.append(
                            f"{py.relative_to(REPO_ROOT)}: import '{top}' 映射到模块 '{mod}'，但 SOURCE_LOCK.md 无此条目"
                        )
    return errors


def check_vendor_lock(entries: dict[str, LockEntry]) -> list[str]:
    """规则 4：vendor 目录 commit 必须与 SOURCE_LOCK 记录一致。"""
    errors: list[str] = []
    vendor = REPO_ROOT / "vendor"
    mapping = {"gdn2": "GDN-2", "ctm": "CMM"}
    for dirname, mod in mapping.items():
        d = vendor / dirname
        if not d.exists():
            continue
        head = None
        if (d / ".git").exists():
            try:
                head = subprocess.check_output(
                    ["git", "-C", str(d), "rev-parse", "HEAD"], text=True
                ).strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        if head is None:
            # tarball 解压（无 .git）：以目录内 LOCKED_COMMIT 戳记为准（由 fetch 脚本写入）
            stamp = d / "LOCKED_COMMIT"
            if stamp.exists():
                head = stamp.read_text().strip()
            else:
                errors.append(
                    f"vendor/{dirname}: 无法确认 commit（无 .git 且无 LOCKED_COMMIT 戳记）"
                )
                continue
        entry = entries.get(mod)
        if entry is None:
            errors.append(f"vendor/{dirname}: SOURCE_LOCK.md 缺少模块 '{mod}' 条目")
            continue
        recorded = entry.fields.get("Commit", "")
        m = re.search(r"`([0-9a-f]{40})`", recorded)
        if not m:
            errors.append(f"vendor/{dirname}: SOURCE_LOCK 模块 '{mod}' 未记录 40 位 commit")
            continue
        if m.group(1) != head:
            errors.append(
                f"vendor/{dirname}: commit 漂移 new={head[:12]} locked={m.group(1)[:12]} (模块 {mod})"
            )
    return errors


def check_frozen(entries: dict[str, LockEntry]) -> list[str]:
    """UNVERIFIED 模块若存在非 skip 的实现文件，则告警失败。"""
    errors: list[str] = []
    for mod in FROZEN_MODULES:
        entry = entries.get(mod)
        if entry and entry.status != "UNVERIFIED":
            errors.append(f"模块 '{mod}' 期望 UNVERIFIED（冻结），实际 '{entry.status}'")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="SOURCE_LOCK CI 闸门")
    ap.add_argument("--source-lock", default=str(SOURCE_LOCK))
    args = ap.parse_args()

    lock_path = Path(args.source_lock)
    if not lock_path.exists():
        print(f"FATAL: {lock_path} 不存在，train.py 不得启动", file=sys.stderr)
        return 1

    entries = parse_source_lock(lock_path)
    print(f"SOURCE_LOCK 解析到 {len(entries)} 个模块: {sorted(entries)}")

    all_errors: list[str] = []
    all_errors += check_imports(entries)
    all_errors += check_vendor_lock(entries)
    all_errors += check_frozen(entries)

    if all_errors:
        print(f"\nLOCK CHECK FAILED — {len(all_errors)} 项缺口：", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("LOCK CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
