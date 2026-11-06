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
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from hcl_tr.data.pipeline import (
    ContaminationError, MinHashConfig, decontaminate, edu_filter,
    interleave_token_streams, minhash_dedup_fast,
)
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


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def dedup_exact(docs: list[str]) -> list[str]:
    seen: set[bytes] = set()
    out = []
    for d in docs:
        h = hashlib.sha1(d.encode("utf-8")).digest()
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out


def filter_by_length(docs: list[str], min_chars: int = 50,
                     max_chars: int = 50_000) -> list[str]:
    return [d for d in docs if min_chars <= len(d) <= max_chars]


def shuffle_docs(docs: list[str], seed: int = 0) -> list[str]:
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    return [docs[i] for i in order]


def preprocess_docs(docs: list[str], *, min_chars: int = 50, max_chars: int = 50_000,
                    near_dedup: bool = True, threshold: float = 0.8, seed: int = 0,
                    log=print) -> list[str]:
    """规范化 → 精确去重 → 长度过滤 → 近重复去重 → 全局打散。

    打散是必须的：按文件顺序拼接会让训练依次只看到单一领域
    （先全部 code，再全部 dialogue…），已实测导致 loss 异常与衰减失控。
    """
    docs = [normalize_ws(d) for d in docs]
    docs = [d for d in docs if d]
    n0 = len(docs)
    docs = dedup_exact(docs)
    log(f"预处理: 精确去重 {n0} -> {len(docs)}")
    docs = filter_by_length(docs, min_chars, max_chars)
    log(f"预处理: 长度过滤 -> {len(docs)}")
    if near_dedup and docs:
        try:
            keep = minhash_dedup_fast(docs, MinHashConfig(threshold=threshold))
            docs = [docs[i] for i in keep]
            log(f"预处理: MinHash(0.8) -> {len(docs)}")
        except Exception as exc:  # noqa: BLE001
            log(f"预处理: MinHash 跳过（{type(exc).__name__}: {exc}）")
    docs = shuffle_docs(docs, seed)
    log(f"预处理: 打散完成（seed={seed}）")
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
    shard_idx, buffered, written = 0, 0, 0
    pending = []
    # 分块编码：整语料一次 encode_batch 在亿级 token 下会撑爆内存。
    for start in range(0, len(docs), 4096):
        for enc in tok.encode_batch(docs[start:start + 4096], add_special_tokens=False):
            pending.append(np.asarray(enc.ids, dtype=np.uint16))
            pending.append(np.asarray([eos_id], dtype=np.uint16))
            buffered += len(enc.ids) + 1
            if buffered >= shard_tokens:
                tokens = np.concatenate(pending)
                write_shard(out, shard_idx, tokens)
                written += len(tokens)
                shard_idx += 1
                buffered, pending = 0, []
    if pending:
        tokens = np.concatenate(pending)
        write_shard(out, shard_idx, tokens)
        written += len(tokens)
        shard_idx += 1
    return dict(documents=len(docs), tokens=int(written), shards=shard_idx,
                vocab_size=vocab_size, eos_id=int(eos_id))


def split_holdout(docs: list[str], frac: float = 0.02,
                  seed: int = 0) -> tuple[list[str], list[str]]:
    """按文档哈希做稳定划分；验证集绝不能进入训练流。"""
    if frac <= 0:
        return docs, []
    salt = f"holdout-{seed}".encode("utf-8")
    train, val = [], []
    for d in docs:
        h = hashlib.sha1(salt + d.encode("utf-8")).digest()
        if int.from_bytes(h[:8], "big") / 2**64 < frac:
            val.append(d)
        else:
            train.append(d)
    return train, val


def encode_docs(tok, docs: list[str], eos_id: int, batch: int = 4096) -> np.ndarray:
    parts = []
    for start in range(0, len(docs), batch):
        for enc in tok.encode_batch(docs[start:start + batch], add_special_tokens=False):
            parts.append(np.asarray(enc.ids, dtype=np.uint16))
            parts.append(np.asarray([eos_id], dtype=np.uint16))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)


def _write_arrays(tokens: np.ndarray, out: Path, shard_tokens: int) -> tuple[int, int]:
    if tokens.size == 0:
        return 0, 0
    idx = 0
    for start in range(0, len(tokens), shard_tokens):
        chunk = tokens[start:start + shard_tokens]
        write_shard(out, idx, chunk)
        idx += 1
    return len(tokens), idx


def build_mixed_shards(train_by_source: dict[str, list[str]],
                       val_by_source: dict[str, list[str]],
                       tokenizer_path: str, out_dir: str,
                       weights: dict[str, float], total_tokens: int,
                       shard_tokens: int = 32_000_000,
                       eval_texts: list[str] | None = None,
                       decontam_ngram: int = 0,
                       decontam_max_drop: float = 0.01,
                       seed: int = 0, overwrite: bool = False,
                       log=print) -> dict:
    """按 token 配额混合多源语料，产出训练分片与独立验证分片。

    - 去污染：decontam_ngram > 0 时必须提供 eval_texts，否则 loud fail。
    - 训练/验证分片物理隔离，验证集不参与训练流。
    - manifest 记录实际（realized）配比，便于审计。
    """
    from tokenizers import Tokenizer

    out = Path(out_dir)
    existing = list(out.glob("shard_*.npy"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{out} 已有 {len(existing)} 个分片；为避免旧分片混入新语料，"
            "请换目录或显式 overwrite=True")
    if overwrite:
        for p in existing + list(out.glob("shard_*.ready")):
            p.unlink()

    tok = Tokenizer.from_file(tokenizer_path)
    eos_id = tok.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = tok.get_vocab_size() - 1
    vocab_size = tok.get_vocab_size()
    if vocab_size >= 65536:
        raise ValueError(f"词表 {vocab_size} 超出 uint16 上限 65535")

    if decontam_ngram > 0:
        if not eval_texts:
            raise ContaminationError(
                "decontam_ngram>0 但未提供评测集；计划书 §6 要求 loud fail，拒绝产出")
        for name in list(train_by_source):
            keep = decontaminate(train_by_source[name], eval_texts,
                                 n=decontam_ngram, max_drop_ratio=decontam_max_drop)
            train_by_source[name] = [train_by_source[name][i] for i in keep]
            log(f"去污染 {name}: 保留 {len(train_by_source[name])} 条")

    streams = {name: encode_docs(tok, docs, eos_id)
               for name, docs in train_by_source.items()}
    available = sum(len(s) for s in streams.values())
    total = min(int(total_tokens), available)
    if total <= 0:
        raise ValueError("没有可用 token：检查输入与预处理过滤阈值")
    mixed, used = interleave_token_streams(streams, weights, total,
                                           return_counts=True)
    written, shards = _write_arrays(mixed, out, shard_tokens)

    val_tokens = 0
    val_dir = out / "val"
    for name, docs in val_by_source.items():
        if not docs:
            continue
        val_dir.mkdir(parents=True, exist_ok=True)
        arr = encode_docs(tok, docs, eos_id)
        n, _ = _write_arrays(arr, val_dir, shard_tokens)
        val_tokens += n
        log(f"验证集 {name}: {len(docs)} 文档 / {n} token")

    manifest = {
        "documents": {n: len(d) for n, d in train_by_source.items()},
        "tokens": int(written), "shards": int(shards), "vocab_size": vocab_size,
        "eos_id": int(eos_id), "seed": int(seed),
        "weights": {n: float(weights.get(n, 0.0)) for n in train_by_source},
        "realized_tokens": {n: int(v) for n, v in used.items()},
        "realized_ratio": {n: (v / written if written else 0.0)
                           for n, v in used.items()},
        "val_tokens": int(val_tokens),
        "val_documents": {n: len(d) for n, d in val_by_source.items()},
        "decontam_ngram": int(decontam_ngram),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--field", action="append", default=[])
    ap.add_argument("--out", default="data/shards")
    ap.add_argument("--tokenizer")
    ap.add_argument("--train-tokenizer", help="词表输出目录；提供则先训练再转分片")
    ap.add_argument("--vocab-size", type=int, default=64000)
    ap.add_argument("--shard-tokens", type=int, default=2_000_000)
    ap.add_argument("--dedup", action="store_true", help="已默认开启 MinHash；保留兼容")
    ap.add_argument("--no-near-dedup", action="store_true")
    ap.add_argument("--min-chars", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=50_000)
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-tokens", type=int, default=10**12)
    ap.add_argument("--eval-texts", help="去污染用评测文本（每行一条）")
    ap.add_argument("--decontam-ngram", type=int, default=0,
                    help=">0 时强制要求 --eval-texts，否则 loud fail")
    ap.add_argument("--edu-min", type=float, default=0.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    docs = load_documents(args.input, tuple(args.field))
    if args.limit:
        docs = docs[:args.limit]
    docs = preprocess_docs(docs, min_chars=args.min_chars, max_chars=args.max_chars,
                           near_dedup=not args.no_near_dedup, seed=args.seed)
    if args.edu_min > 0:
        docs = edu_filter(docs, min_score=args.edu_min)
        print(f"教育分过滤后保留 {len(docs)} 条")
    train_docs, val_docs = split_holdout(docs, args.holdout_frac, args.seed)
    print(f"训练/验证划分: {len(train_docs)} / {len(val_docs)}")

    tokenizer_path = args.tokenizer
    if args.train_tokenizer:
        from hcl_tr.data.tokenizer import tokenizer_meta, train_bpe
        path = train_bpe(train_docs, args.train_tokenizer, vocab_size=args.vocab_size)
        tokenizer_path = str(path)
        print("词表:", tokenizer_meta(path, args.vocab_size))
    if not tokenizer_path:
        raise SystemExit("需要 --tokenizer 或 --train-tokenizer")

    eval_texts = None
    if args.eval_texts:
        eval_texts = [ln.strip() for ln in
                      Path(args.eval_texts).read_text(encoding="utf-8").splitlines()
                      if ln.strip()]

    result = build_mixed_shards(
        {"main": train_docs}, {"main": val_docs}, tokenizer_path, args.out,
        weights={"main": 1.0}, total_tokens=args.total_tokens,
        shard_tokens=args.shard_tokens, eval_texts=eval_texts,
        decontam_ngram=args.decontam_ngram, seed=args.seed,
        overwrite=args.overwrite)
    result.update(inputs=list(args.input), input_sha256=_sha256(args.input),
                  tokenizer=tokenizer_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.manifest:
        Path(args.manifest).write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
