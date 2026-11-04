"""check_load_revision: 强制所有 checkpoint 加载带 revision。

规则（SOURCE_LOCK.md CI 闸门第 2 条）:
- `torch.load(...)` 必须显式传入一个 `sha256=` 或 `path` 属于已冻结 baseline/fixture 目录。
- `from_pretrained(...)` 必须传入 `revision=`。

这是最简的 AST 级检查；不做数据流分析，因此宁可保守（要求显式关键字）。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def check_file(py: Path) -> list[str]:
    errors: list[str] = []
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = ""
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id

        if name == "from_pretrained":
            kwargs = {k.arg for k in node.keywords}
            if "revision" not in kwargs:
                errors.append(f"{py}: from_pretrained 缺少 revision=")

        # 只认 torch.load(...)；此前把 np.load(json).load(等任意 .load 都当 torch.load，
        # 产生误报（prefetch.py 的 np.load 被误判）。
        if name == "load" and isinstance(node.func, ast.Attribute):
            root = node.func.value
            is_torch = (
                (isinstance(root, ast.Name) and root.id == "torch")
                or (isinstance(root, ast.Attribute) and root.attr == "torch")
            )
            # checkpoint.py 是唯一集中做 hash 校验的地方，允许出现原始 torch.load
            if is_torch and py.name != "checkpoint.py":
                kwargs = {k.arg for k in node.keywords}
                if "sha256" not in kwargs and "expected_sha256" not in kwargs:
                    errors.append(
                        f"{py}:{node.lineno}: torch.load 缺少 sha256=/expected_sha256= 校验"
                    )
    return errors


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv[1:] if a.endswith(".py")]
    all_errors: list[str] = []
    for f in files:
        if not f.exists():
            continue
        all_errors += check_file(f)
    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
