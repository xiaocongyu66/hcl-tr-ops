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


def test_preprocess_dedups_filters_and_shuffles():
    base = "hello world " * 12
    messy = "   " + base.replace(" ", "   ") + "   "
    docs = [messy, base, "short", "x" * 60, "y" * 60, "z" * 60] + \
        [f"doc number {i} " * 20 for i in range(200)]
    out = ingest.preprocess_docs(docs, near_dedup=False, seed=3, log=lambda *_: None)
    expected = " ".join(base.split())
    assert out.count(expected) == 1                # 规范化后精确去重
    assert all(len(d) >= 50 for d in out)          # 长度过滤
    again = ingest.preprocess_docs(docs, near_dedup=False, seed=3, log=lambda *_: None)
    assert out == again                            # 固定 seed 可复现
    other = ingest.preprocess_docs(docs, near_dedup=False, seed=4, log=lambda *_: None)
    assert out != other                            # 不同 seed 顺序不同


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


def _toy_tokenizer(tmp_path, docs):
    from hcl_tr.data.tokenizer import train_bpe
    return train_bpe(docs, tmp_path / "tok", vocab_size=300)


def test_holdout_is_disjoint_and_nonempty():
    docs = [f"document {i} " * 30 for i in range(500)]
    train, val = ingest.split_holdout(docs, 0.2, seed=0)
    assert val and train
    assert set(train).isdisjoint(set(val))
    train2, val2 = ingest.split_holdout(docs, 0.2, seed=0)
    assert train == train2 and val == val2      # 同 seed 稳定


def test_build_mixed_shards_follows_weights_and_writes_val(tmp_path):
    docs_a = [f"alpha doc {i} " * 20 for i in range(300)]
    docs_b = [f"beta doc {i} " * 20 for i in range(300)]
    tok = _toy_tokenizer(tmp_path, docs_a + docs_b)
    out = tmp_path / "shards"
    meta = ingest.build_mixed_shards(
        {"a": docs_a, "b": docs_b}, {"a": docs_a[:10], "b": docs_b[:10]},
        str(tok), out, weights={"a": 0.75, "b": 0.25}, total_tokens=4000,
        shard_tokens=1500, seed=0)
    assert meta["tokens"] == 4000
    assert abs(meta["realized_ratio"]["a"] - 0.75) < 0.15
    assert meta["val_tokens"] > 0
    assert (out / "manifest.json").exists()
    assert (out / "val").is_dir()
    # 再次写入同一目录必须拒绝，避免旧分片混入
    with pytest.raises(FileExistsError):
        ingest.build_mixed_shards({"a": docs_a}, {"a": []}, str(tok), out,
                                  weights={"a": 1.0}, total_tokens=100,
                                  shard_tokens=1500)


def test_decontamination_requires_eval_texts(tmp_path):
    from hcl_tr.data.pipeline import ContaminationError
    docs = [f"doc {i} " * 30 for i in range(50)]
    tok = _toy_tokenizer(tmp_path, docs)
    with pytest.raises(ContaminationError):
        ingest.build_mixed_shards({"a": docs}, {"a": []}, str(tok),
                                  tmp_path / "s2", weights={"a": 1.0},
                                  total_tokens=100, shard_tokens=500,
                                  decontam_ngram=13)


def test_overwrite_clears_stale_val_shards(tmp_path):
    docs = [f"doc {i} " * 30 for i in range(200)]
    tok = _toy_tokenizer(tmp_path, docs)
    out = tmp_path / "s"
    ingest.build_mixed_shards({"a": docs}, {"a": docs[:40]}, str(tok), out,
                              weights={"a": 1.0}, total_tokens=2000,
                              shard_tokens=300)
    first = len(list((out / "val").glob("shard_*.npy")))
    assert first >= 2
    ingest.build_mixed_shards({"a": docs}, {"a": docs[:2]}, str(tok), out,
                              weights={"a": 1.0}, total_tokens=500,
                              shard_tokens=300, overwrite=True)
    second = len(list((out / "val").glob("shard_*.npy")))
    assert second < first, "overwrite 未清理旧验证分片"


def test_cli_multi_source_weights(tmp_path, monkeypatch):
    import sys
    src_a = tmp_path / "a.jsonl"
    src_b = tmp_path / "b.jsonl"
    rows_a = [{"text": f"alpha {i} " * 20} for i in range(100)]
    rows_b = [{"text": f"beta {i} " * 20} for i in range(100)]
    src_a.write_text("\n".join(json.dumps(r) for r in rows_a), encoding="utf-8")
    src_b.write_text("\n".join(json.dumps(r) for r in rows_b), encoding="utf-8")
    out = tmp_path / "shards"
    tok_dir = tmp_path / "tok"
    monkeypatch.setattr(sys, "argv", [
        "ingest_text.py", "--source", f"a={src_a}", "--source", f"b={src_b}",
        "--weight", "a=0.9", "--weight", "b=0.1",
        "--train-tokenizer", str(tok_dir), "--vocab-size", "300",
        "--out", str(out), "--total-tokens", "2000", "--shard-tokens", "1000",
        "--no-near-dedup",
    ])
    ingest.main()
    meta = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert meta["tokens"] == 2000
    assert meta["realized_ratio"]["a"] > 0.8
