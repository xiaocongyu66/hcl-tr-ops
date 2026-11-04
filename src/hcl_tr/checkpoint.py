"""checkpoint 的安全加载：强制 sha256 校验后再反序列化。

存在理由（SOURCE_LOCK CI 闸门第 2 条）：直接 `torch.load` 一个未校验的权重文件，
可能静默加载错误/被篡改的 checkpoint，训练结果不可复现。所有 checkpoint 加载
必须经由本模块，原始 `torch.load` 只允许出现在这里。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_torch_load(path: str | Path, sha256: str | None = None,
                    map_location="cpu", weights_only: bool = False):
    """校验 sha256 后加载。

    sha256 为 None 时抛错——不允许"先加载再说"，必须显式给出期望值。
    """
    if not sha256:
        raise ValueError(
            f"拒绝加载 {path}：必须提供 sha256（禁止无校验加载 checkpoint）"
        )
    actual = sha256_file(path)
    if actual != sha256:
        raise ValueError(f"sha256 不匹配: 期望 {sha256}, 实际 {actual}（文件: {path}）")
    return torch.load(path, map_location=map_location, weights_only=weights_only)
