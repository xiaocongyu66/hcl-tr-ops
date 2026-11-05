"""外部数据集 → uint16 分片 的端到端校验（CPU，无需 GPU）。"""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")

SPEC = importlib.util.spec_from_file_location(
    'ingest_text', Path(__file__).resolve().parents[2] / 'scripts' / 'ingest_text.py')
ingest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ingest)


def _write_inputs(tmp: Path) -> tuple[Path, Path]:
    jsonl = tmp / "qa.jsonl"
    rows = [{"prompt": f"question {i}", "response": f"answer {i}"} for i in range(40)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    csv_path = tmp / "docs.csv"
    csv_path.write_text("text\nhello world\nsecond document\n", encoding="utf-8")
    return jsonl, csv_path


def test_extracts_prompt_and_response(tmp_path):
    jsonl, csv_path = _write_inputs(tmp_path)
    docs = ingest.load_documents([str(jsonl), str(csv_path)])
    assert len(docs) == 42
    assert docs[0].startswith("question 0")
    assert "answer 0" in docs[0]
    assert docs[-1] == "second document"


def test_builds_shards_with_ready_markers(tmp_path):
    from hcl_tr.data.prefetch import ShardStream
    jsonl, csv_path = _write_inputs(tmp_path)
    docs = ingest.load_documents([str(jsonl), str(csv_path)])
    tok_dir = tmp_path / "tok"
    from hcl_tr.data.tokenizer import train_bpe
    tok_path = train_bpe(docs, tok_dir, vocab_size=300)
    shard_dir = tmp_path / "shards"
    meta = ingest.build_shards(docs, str(tok_path), str(shard_dir), shard_tokens=200)
    assert meta["documents"] == len(docs)
    assert meta["tokens"] > 0
    shards = sorted(shard_dir.glob("shard_*.npy"))
    assert shards and len(shards) == meta["shards"]
    assert all(p.with_suffix(".ready").exists() for p in shards)
    stream = ShardStream(shard_dir, seq_len=16, batch_tokens=64, delete_consumed=False)
    batch = next(iter(stream.batches(stop_after_tokens=64)))
    assert batch.shape[1] == 16
    assert int(batch.max()) < meta["vocab_size"]
