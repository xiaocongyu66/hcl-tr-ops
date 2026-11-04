"""数据管线夹具：MinHash 去重 / 教育分 / 去污染 loud-fail / mix 可复现。"""

from __future__ import annotations

import pytest

from hcl_tr.data.pipeline import (
    ContaminationError,
    MinHashConfig,
    MixSource,
    build_frozen_manifest,
    decontaminate,
    edu_filter,
    heuristic_edu_score,
    minhash_dedup,
    mix_indices,
    ngram_hashes,
)

# 注意：必须用**不重复**的文本。若句子本身重复，shingle 集合很小，
# 追加一句就会让集合 Jaccard 掉到 0.67（我第一版测试就栽在这）。
DOC = " ".join(f"w{i}" for i in range(200))


def test_minhash_removes_exact_and_near_duplicates():
    # 近重复 = 原文 + 少量追加（shingle 重叠仍 > 0.8）
    near_dup = DOC + "one extra sentence appended at the very end"
    docs = [DOC, DOC, near_dup, "completely different text " * 20]
    kept = minhash_dedup(docs, MinHashConfig(num_perm=64, bands=16, threshold=0.8))
    assert kept == [0, 3], kept


def test_minhash_keeps_doc_below_threshold():
    """改动比例过大（集合 Jaccard < 阈值）的不应被当作重复删掉。"""
    heavy_edit = " ".join(f"z{i}" for i in range(200))   # 完全不同的词
    docs = [DOC, heavy_edit]
    kept = minhash_dedup(docs, MinHashConfig(num_perm=64, bands=16, threshold=0.8))
    assert kept == [0, 1], f"改动过大不该判重: {kept}"


def test_minhash_keeps_distinct_docs():
    docs = [f"document number {i} " + "word " * 60 for i in range(8)]
    kept = minhash_dedup(docs, MinHashConfig(num_perm=32, bands=8, threshold=0.9))
    assert len(kept) == 8


def test_minhash_deterministic():
    docs = [DOC, DOC, "another doc " * 40]
    a = minhash_dedup(docs, MinHashConfig(seed=3))
    b = minhash_dedup(docs, MinHashConfig(seed=3))
    assert a == b


def test_edu_score_orders_quality():
    good = "This is a well formed sentence with plenty of words. " * 12
    junk = "x " * 5
    assert heuristic_edu_score(good) > heuristic_edu_score(junk)
    assert heuristic_edu_score("") == 0.0


def test_edu_filter_drops_low_quality():
    good = "This is a well formed sentence with plenty of words. " * 12
    bad = "a b"
    kept = edu_filter([good, bad], min_score=0.5)
    assert kept == [0]


def test_decontamination_removes_overlap():
    evaldoc = "the capital of france is paris and it is a big city in europe " * 3
    docs = [evaldoc, "unrelated content about cooking pasta and tomato sauce " * 5]
    kept = decontaminate(docs, [evaldoc], max_drop_ratio=0.9)
    assert kept == [1]


def test_decontamination_loud_fail_on_massive_leak():
    """丢弃比例过高必须报错，不能静默放行（计划书红线）。"""
    evaldoc = "overlapping text about neural networks and attention mechanisms " * 3
    docs = [evaldoc] * 10
    with pytest.raises(ContaminationError):
        decontaminate(docs, [evaldoc], max_drop_ratio=0.01)


def test_mix_indices_respects_weights_and_is_deterministic():
    a = MixSource("en", [f"en{i}" for i in range(100)], weight=0.8)
    b = MixSource("zh", [f"zh{i}" for i in range(100)], weight=0.2)
    m1 = mix_indices([a, b], seed=0)
    m2 = mix_indices([a, b], seed=0)
    assert m1 == m2
    n_en = sum(1 for s, _ in m1 if s == "en")
    n_zh = sum(1 for s, _ in m1 if s == "zh")
    assert n_en == 80 and n_zh == 20, (n_en, n_zh)


def test_mix_hash_changes_with_weights():
    a = MixSource("en", ["x"] * 10, 0.9)
    b = MixSource("zh", ["y"] * 10, 0.1)
    h1 = build_frozen_manifest([a, b], 64000, "tokhash", 0, 0.5, 0.8)["data_mix_hash"]
    a2 = MixSource("en", ["x"] * 10, 0.8)
    b2 = MixSource("zh", ["y"] * 10, 0.2)
    h2 = build_frozen_manifest([a2, b2], 64000, "tokhash", 0, 0.5, 0.8)["data_mix_hash"]
    assert h1 != h2


def test_mix_hash_changes_with_tokenizer():
    a = MixSource("en", ["x"] * 10, 1.0)
    h1 = build_frozen_manifest([a], 64000, "tokA", 0, 0.5, 0.8)["data_mix_hash"]
    h2 = build_frozen_manifest([a], 64000, "tokB", 0, 0.5, 0.8)["data_mix_hash"]
    assert h1 != h2


def test_ngram_hashes_detects_shared_span():
    a = "the quick brown fox jumps over the lazy dog near the river bank"
    b = "totally different start " + a
    assert ngram_hashes(a, 5) & ngram_hashes(b, 5)


def test_cpp_minhash_matches_python_and_is_fast():
    """C++ 去重必须与 Python 语义一致（保留首次出现），且快得多。"""
    import random
    import time
    from hcl_tr.data.pipeline import minhash_dedup_fast, _build_cpp
    if _build_cpp() is None:
        pytest.skip("无 g++，跳过 C++ 路径")
    random.seed(0)
    base = [" ".join(f"w{random.randrange(5000)}" for _ in range(340)) for _ in range(200)]
    docs = (base + [base[i] for i in range(40)]
            + [base[i] + " extra tail" for i in range(40)]
            + [" ".join(f"w{random.randrange(5000)}" for _ in range(340)) for _ in range(60)])
    cfg = MinHashConfig()
    a = minhash_dedup(docs, cfg)
    b = minhash_dedup_fast(docs, cfg)
    assert a == b, f"C++ 与 Python 结果不同: {len(a)} vs {len(b)}"
    big = [" ".join(f"w{random.randrange(50000)}" for _ in range(340)) for _ in range(3000)]
    t = time.time(); minhash_dedup_fast(big, cfg); dt = time.time() - t
    assert dt < 20, f"C++ 去重过慢: {dt:.1f}s"


def test_cpp_minhash_handles_chinese():
    from hcl_tr.data.pipeline import minhash_dedup_fast, _build_cpp
    if _build_cpp() is None:
        pytest.skip("无 g++")
    zh = ["这是一个用于测试中文去重的句子，包含足够多的字符来形成 shingle。" * 5 for _ in range(20)]
    zh += ["完全不同的另一段中文内容，讲述的是天气与季节的变化。" * 5 for _ in range(20)]
    kept = minhash_dedup_fast(zh, MinHashConfig())
    assert len(kept) == 2, f"中文去重应保留 2 篇（各 20 个重复），实际 {len(kept)}"


def test_edu_score_handles_chinese():
    """中文没有空格，教育分不能因词数不足而误杀。"""
    zh = "人工智能的发展正在改变世界，语言模型在其中扮演着重要角色。" * 12
    score = heuristic_edu_score(zh)
    assert score >= 0.5, f"中文高质量文本被误杀: {score}"


def test_interleave_prevents_domain_blocking():
    """交错后，任意窗口内都应同时出现两路数据（防止末段全是同一领域）。"""
    import numpy as np
    from hcl_tr.data.pipeline import interleave_token_streams
    a = np.zeros(10000, dtype=np.int64)       # 源 A 全是 0
    b = np.ones(10000, dtype=np.int64)        # 源 B 全是 1
    out = interleave_token_streams({"a": a, "b": b}, {"a": 0.5, "b": 0.5}, 8000, chunk=100)
    assert len(out) == 8000
    # 前 1000 个 token 里两路都有
    head = out[:1000]
    assert (head == 0).any() and (head == 1).any(), "前段仍被单一源占据"
    # 整体比例接近 1:1
    ratio = (out == 0).mean()
    assert 0.4 < ratio < 0.6, ratio


def test_interleave_respects_weights():
    import numpy as np
    from hcl_tr.data.pipeline import interleave_token_streams
    a = np.zeros(100000, dtype=np.int64); b = np.ones(100000, dtype=np.int64)
    out = interleave_token_streams({"a": a, "b": b}, {"a": 0.8, "b": 0.2}, 10000, chunk=100)
    ratio = (out == 0).mean()
    assert 0.7 < ratio < 0.9, ratio
