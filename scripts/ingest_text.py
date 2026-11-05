"""把外部文本/指令数据集转成训练侧消费的 uint16 分片。

支持 jsonl / csv / txt：自动识别 text、prompt+response、question+answer 等字段。
流程：读取 → 提取文本 → 可选 MinHash 去重 → 可选教育分过滤 → 分词 → 原子写分片。

示例：
  # 训练词表（首次）
  python scripts/ingest_text.py --train-tokenizer data/tokenizer \
      --input data/sample/logic_qa.jsonl --vocab-size 64000

  # 转分片
  python scripts/ingest_text.py --tokenizer data/tokenizer/tokenizer.json \
      --input data/sample/logic_qa.jsonl --input data/sample/NEON_KNOWLEDGE_SFT_1.csv \
      --out data/shards --shard-tokens 2000000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from hcl_tr.data.pipeline import MinHashConfig, edu_filter, minhash_dedup_fast
from hcl_tr.data.prefetch import write_shard

TEXT_FIELDS = ("text", "content", "document", "completion", "answer", "response", "output")
PROMPT_FIELDS = ("prompt", "question", "instruction", "input", "query")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return [{"text": b} for b in blocks]


def _extract(row: dict, explicit: tuple[str, ...]) -> str:
    """把一行记录还原成一段训练文本；prompt+response 用换行拼接。"""
    if explicit:
        parts = [str(row[f]) for f in explicit if row.get(f) not in (None, "")]
        return "\n".join(parts)
    prompt = next((str(row[f]) for f in PROMPT_FIELDS if row.get(f) not in (None, "")), "")
    body = next((str(row[f]) for f in TEXT_FIELDS if row.get(f) not in (None, "")), "")
    if prompt and body:
        return f"{prompt}\n{body}"
    return prompt or body


def load_documents(paths: Iterable[str], explicit_fields: tuple[str, ...] = ()) -> list[str]:
    docs: list[str] = []
    for raw in paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = _read_jsonl(path)
        elif suffix == ".csv":
            rows = _read_csv(path)
        elif suffix in (".txt", ".md"):
            rows = _read_txt(path)
        else:
            raise ValueError(f"不支持的文件类型: {path}")
        before = len(docs)
        docs.extend(t for t in (_extract(r, explicit_fields) for r in rows) if t.strip())
        print(f"{path}: 读取 {len(rows)} 行，保留 {len(docs) - before} 条")
    return docs


def _sha256(paths: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def build_shards(docs: list[str], tokenizer_path: str, out_dir: str,
                 shard_tokens: int = 2_000_000, eos: str = "<|endoftext|>") -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id(eos)
    if eos_id is None:
        eos_id = tok.get_vocab_size() - 1
    vocab_size = tok.get_vocab_size()
    if vocab_size >= 65536:
        raise ValueError(f"词表 {vocab_size} 超出 uint16 上限 65535")

    out = Path(out_dir)
    shard_idx, buffer, written = 0, [], 0
    for enc in tok.encode_batch(docs, add_special_tokens=False):
        buffer.append(np.asarray(enc.ids, dtype=np.uint16))
        buffer.append(np.asarray([eos_id], dtype=np.uint16))
        if sum(len(b) for b in buffer) >= shard_tokens:
            tokens = np.concatenate(buffer)
            write_shard(out, shard_idx, tokens)
            written += len(tokens)
            shard_idx += 1
            buffer = []
    if buffer:
        tokens = np.concatenate(buffer)
        write_shard(out, shard_idx, tokens)
        written += len(tokens)
        shard_idx += 1
    return dict(documents=len(docs), tokens=int(written), shards=shard_idx,
                vocab_size=vocab_size, eos_id=int(eos_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--field", action="append", default=[])
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--tokenizer")
    ap.add_argument("--train-tokenizer", help="词表输出目录；提供则先训练再转分片")
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--shard-tokens", type=int, default=2_000_000)
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--edu-min", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    docs = load_documents(args.input, tuple(args.field))
    if args.limit:
        docs = docs[:args.limit]
    if args.dedup:
        keep = minhash_dedup_fast(docs, MinHashConfig())
        docs = [docs[i] for i in keep]
        print(f"MinHash 去重后保留 {len(docs)} 条")
    if args.edu_min > 0:
        docs = edu_filter(docs, min_score=args.edu_min)
        print(f"教育分过滤后保留 {len(docs)} 条")

    tokenizer_path = args.tokenizer
    if args.train_tokenizer:
        from hcl_tr.data.tokenizer import tokenizer_meta, train_bpe
        path = train_bpe(docs, args.train_tokenizer, vocab_size=args.vocab_size)
        tokenizer_path = str(path)
        print("词表:", tokenizer_meta(path, args.vocab_size))
    if not tokenizer_path:
        raise SystemExit("需要 --tokenizer 或 --train-tokenizer")

    result = build_shards(docs, tokenizer_path, args.out, args.shard_tokens)
    manifest = dict(inputs=list(args.input), input_sha256=_sha256(args.input),
                    tokenizer=tokenizer_path, **result)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
