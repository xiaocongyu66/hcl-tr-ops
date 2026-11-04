"""freeze_prereg: 冻结预注册评估口径与 baseline 哈希。

用法:
  python scripts/freeze_prereg.py                  # 冻结预注册 YAML 的 hash
  python scripts/freeze_prereg.py --checkpoint PATH --tokenizer PATH --data-list FILE
                                                    # 额外冻结 baseline 的三项哈希

冻结后 baseline JSON 中的 <FILL_ON_FREEZE> 被替换为真实 hash。若某哈希已存在，
再次运行会拒绝覆盖（防止静默换 baseline）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG = REPO_ROOT / "src" / "hcl_tr" / "configs" / "sniah1_ruler_prereg_v1.0.yaml"
BASELINE = REPO_ROOT / "src" / "hcl_tr" / "configs" / "baseline_100m_transformer.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_data_list(files: list[str]) -> str:
    """对 (相对路径, 内容 hash) 排序后取 sha256，得到数据集 mix 的稳定指纹。"""
    rows = []
    for fp in sorted(files):
        p = Path(fp)
        rows.append(f"{p.name}:{sha256_file(p)}")
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--tokenizer")
    ap.add_argument("--data-list", help="文件，每行一个数据分片路径")
    ap.add_argument("--force", action="store_true", help="覆盖已有冻结值（危险，需显式）")
    args = ap.parse_args()

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    def fill(key: str, value: str) -> None:
        cur = baseline.get(key)
        if cur and not cur.startswith("<FILL_ON_FREEZE>") and not args.force:
            print(f"REFUSE: {key} 已冻结为 {cur[:12]}…；换 baseline 需 --force", file=sys.stderr)
            raise SystemExit(2)
        baseline[key] = value
        print(f"froze {key} = {value}")

    if not PREREG.exists():
        print(f"FATAL: 预注册 YAML 不存在: {PREREG}", file=sys.stderr)
        return 1
    fill("eval_protocol_hash", sha256_file(PREREG))

    if args.checkpoint:
        fill("checkpoint_sha256", sha256_file(Path(args.checkpoint)))
    if args.tokenizer:
        fill("tokenizer_sha256", sha256_file(Path(args.tokenizer)))
    if args.data_list:
        files = [l.strip() for l in Path(args.data_list).read_text().splitlines() if l.strip()]
        if not files:
            print("FATAL: --data-list 为空", file=sys.stderr)
            return 1
        fill("data_mix_hash", sha256_data_list(files))

    BASELINE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwritten {BASELINE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
