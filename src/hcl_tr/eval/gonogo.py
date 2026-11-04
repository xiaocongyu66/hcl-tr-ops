"""自动 NoGo（计划书 §11 / §12 第 6 步）。

三条硬条件（全部写进 eval.py，且阈值不可运行时调整）：
  1. relative_to_baseline < 0.80  → NoGo
  2. absolute_length     < 2000  → NoGo
  3. seed_std_over_mean  > 0.20  → NoGo

NoGo 动作白名单（只允许）：
  - 停止训练
  - 标记 NoGo
  - 保存归因 JSON（含 SOURCE_LOCK hash、fixture 结果、预注册 YAML hash）
  - 按预注册策略扩容到 300M–1B 或增加长文本数据比例

禁止自动动作：调 LR / 改门控阈值 / 换 baseline / 重定义可用长度 / 引入替代实现 / 悄悄 clipping。
diagnose() 只输出证据与建议阅读的 fixture/日志，不自动调参。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# 阈值为常量：禁止运行时改写（见 ALLOWED_ACTIONS / FORBIDDEN_ACTIONS）
THRESH_RELATIVE = 0.80
THRESH_ABSOLUTE = 2000
THRESH_SEED_STD = 0.20

ALLOWED_ACTIONS = {
    "stop_training",
    "mark_nogo",
    "save_attribution_json",
    "escalate_model_scale_300m_1b",
    "increase_long_text_data_ratio",
}

FORBIDDEN_ACTIONS = {
    "tune_lr",
    "change_gate_threshold",
    "swap_baseline",
    "redefine_usable_length",
    "substitute_implementation",
    "silent_is_clip",
}


@dataclass
class GoNoGoResult:
    verdict: str                       # "Go" | "NoGo"
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


def _wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% 置信区间下界（预注册口径 accuracy_ci: Wilson_95）。"""
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half)


def usable_context_length(
    acc_by_length: dict[int, tuple[int, int]],
) -> int | None:
    """按预注册规则求可用长度：最大 L 使 acc(L)>=0.90 且 Wilson 95% 下界>=0.85。

    acc_by_length: {L: (successes, n)}
    只允许在预注册长度点上判定，不外推。
    """
    usable = None
    for L in sorted(acc_by_length):
        s, n = acc_by_length[L]
        acc = s / n if n else 0.0
        lo = _wilson_lower(s, n)
        if acc >= 0.90 and lo >= 0.85:
            usable = L
        else:
            # 一旦某长度不达标，更长长度也不计入（单调性假设）
            break
    return usable


def check_gonogo(
    results: dict,
    baseline: dict,
    protocol: dict,
) -> GoNoGoResult:
    """核心判定。

    results 需含:
      - absolute_length: int   候选模型的可用长度
      - seed_results: list[float]  多 seed 的可用长度
      - relative_to_baseline: float 可选；缺省时用 absolute_length / baseline_length 计算
    baseline 需含: baseline_length（已冻结）
    """
    reasons = []
    ev = {"results": results, "baseline": baseline}

    baseline_len = baseline.get("baseline_length")
    abs_len = results.get("absolute_length")

    rel = results.get("relative_to_baseline")
    if rel is None and baseline_len and abs_len is not None:
        rel = abs_len / baseline_len
    ev["relative_to_baseline"] = rel

    if rel is None:
        return GoNoGoResult("NoGo", ["缺少 relative_to_baseline 或 baseline_length"], ev)

    if rel < THRESH_RELATIVE:
        reasons.append(f"相对 baseline <{THRESH_RELATIVE:.0%}")
    if abs_len is not None and abs_len < THRESH_ABSOLUTE:
        reasons.append(f"绝对值 <{THRESH_ABSOLUTE}")

    seeds = results.get("seed_results", [])
    if len(seeds) >= 2:
        import statistics
        mean = statistics.fmean(seeds)
        std = statistics.pstdev(seeds)
        ssm = std / mean if mean else float("inf")
        ev["seed_std_over_mean"] = ssm
        if ssm > THRESH_SEED_STD:
            reasons.append(f"种子方差 >{THRESH_SEED_STD:.0%}")

    return GoNoGoResult("NoGo" if reasons else "Go", reasons, ev)


def hash_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_attribution_json(
    gonogo: GoNoGoResult,
    prereg_path: Path,
    source_lock_path: Path,
    fixture_report: list[dict],
    extra: dict | None = None,
) -> dict:
    """NoGo 归因 JSON（白名单动作 save_attribution_json 的唯一输出）。"""
    payload = {
        "verdict": gonogo.verdict,
        "reasons": gonogo.reasons,
        "evidence": gonogo.evidence,
        "source_lock_sha256": hash_file(source_lock_path),
        "prereg_yaml_sha256": hash_file(prereg_path),
        "fixture_report": fixture_report,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "forbidden_actions": sorted(FORBIDDEN_ACTIONS),
    }
    if extra:
        payload.update(extra)
    return payload


def diagnose(gonogo: GoNoGoResult, fixture_report: list[dict]) -> list[str]:
    """只输出证据与建议阅读的 fixture/日志，不自动调参。"""
    lines = [f"verdict={gonogo.verdict}"]
    for r in gonogo.reasons:
        lines.append(f"  reason: {r}")
    lines.append("建议阅读：")
    for f in fixture_report:
        lines.append(f"  - fixture {f.get('name')}: {f.get('status')} {f.get('detail', '')}")
    lines.append("注意：本函数不执行任何调参；调 LR / 改阈值 / 换 baseline 均在禁止清单内。")
    return lines
