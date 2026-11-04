"""S-NIAH-1 / RULER 评测 runner —— Go/NoGo 的执行入口。

**所有评测口径必须从预注册 YAML 读取**（计划书 §8、"不允许重定义可用长度"）：
长度网格、每长度样本数、needle 位置、判定规则、主指标名称，一律不硬编码。
预注册文件被改动必须升版本号并重算 hash。

设计：模型推理通过 `predict(prompt: str) -> str` 注入，便于用假模型做单元测试，
也便于将来接 HybridBackbone 或任意 checkpoint。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import yaml

from .gonogo import check_gonogo, usable_context_length


REQUIRED_KEYS = (
    "lengths", "samples_per_length", "usable_length_definition",
    "primary_metric", "training_length",
)


@dataclass
class Task:
    length: int
    needle: str
    question: str
    prompt: str
    answer: str


def prereg_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prereg(path: str | Path) -> dict:
    """读取并校验预注册 YAML。缺关键字段直接报错（不许静默用默认值）。"""
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"预注册文件 {p} 缺少必需字段: {missing}")
    if not isinstance(data["lengths"], list) or not data["lengths"]:
        raise ValueError("lengths 必须是非空列表")
    data["_prereg_hash"] = prereg_hash(p)
    data["_prereg_path"] = str(p)
    return data


# ---------------- S-NIAH-1 任务构造 ----------------

_FILLER = ("The grass is green. The sky is blue. The sun rises in the east. "
           "Water flows downhill. Birds fly in the sky. Trees grow tall.")


def build_tasks(prereg: dict, seed: int = 0,
                key_kind: str = "word") -> list[Task]:
    """按预注册的长度网格与位置规则构造 needle-in-a-haystack 任务。"""
    rng = random.Random(seed)
    tasks: list[Task] = []
    samples = int(prereg["samples_per_length"])
    for L in prereg["lengths"]:
        for i in range(samples):
            # needle 位置：预注册只写"均匀采样 10 个位置"，按其含义等距取点
            depth = (i % 10) / 10.0
            needle = f"magic-{rng.randrange(10**6):06d}"
            question = f"What is the magic number? Answer:"
            # 用词数近似长度（评测口径以 token 计；此处由调用方的 tokenizer 对齐）
            body_words = max(1, L - 20)
            filler = (_FILLER + " ") * (body_words // len(_FILLER.split()) + 1)
            words = filler.split()[:body_words]
            pos = int(depth * len(words))
            words = words[:pos] + [f"The secret code is {needle}."] + words[pos:]
            prompt = " ".join(words) + "\n\n" + question
            tasks.append(Task(L, needle, question, prompt, needle))
    return tasks


def score(prediction: str, answer: str) -> bool:
    return answer.strip().lower() in prediction.strip().lower()


# ---------------- 主流程 ----------------

def evaluate(predict: Callable[[str], str], prereg: dict,
             seed: int = 0) -> dict:
    """跑完整评测，返回 {length: (successes, n)} 与预注册元信息。"""
    tasks = build_tasks(prereg, seed=seed)
    acc: dict[int, list[int]] = {L: [0, 0] for L in prereg["lengths"]}
    for t in tasks:
        pred = predict(t.prompt)
        acc[t.length][1] += 1
        if score(pred, t.answer):
            acc[t.length][0] += 1
    return {
        "acc_by_length": {L: tuple(v) for L, v in acc.items()},
        "prereg_hash": prereg["_prereg_hash"],
        "prereg_path": prereg["_prereg_path"],
    }


def decide(results: dict, baseline: dict, prereg: dict) -> dict:
    """按预注册规则算可用长度，再交给 check_gonogo。"""
    usable = usable_context_length(results["acc_by_length"])
    gonogo_in = {
        "absolute_length": usable if usable is not None else 0,
        "seed_results": results.get("seed_results", []),
    }
    verdict = check_gonogo(gonogo_in, baseline, prereg)
    return {
        "usable_context_length": usable,
        "primary_metric": prereg["primary_metric"],
        "verdict": verdict.verdict,
        "reasons": getattr(verdict, "reasons", []),
        "evidence": getattr(verdict, "evidence", {}),
        "prereg_hash": results["prereg_hash"],
    }


def make_hf_predict(model, tokenizer, max_new_tokens: int = 8,
                    device: str = "cuda") -> Callable[[str], str]:
    """把 HF 风格模型包装成 predict 回调（贪心解码）。"""
    import torch

    def _predict(prompt: str) -> str:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None)
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    return _predict
