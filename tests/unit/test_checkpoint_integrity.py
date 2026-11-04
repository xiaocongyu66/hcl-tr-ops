"""checkpoint 完整性夹具：禁止无校验/错校验加载。"""

from __future__ import annotations

import pytest
import torch

from hcl_tr.checkpoint import safe_torch_load, sha256_file


def test_safe_torch_load_roundtrip(tmp_path):
    p = tmp_path / "ck.pt"
    torch.save({"model": {"w": torch.zeros(2)}, "tokens": 10}, p)
    h = sha256_file(p)
    ck = safe_torch_load(p, sha256=h)
    assert ck["tokens"] == 10


def test_safe_torch_load_requires_sha(tmp_path):
    p = tmp_path / "ck.pt"
    torch.save({"a": 1}, p)
    with pytest.raises(ValueError, match="sha256"):
        safe_torch_load(p, sha256=None)


def test_safe_torch_load_detects_tamper(tmp_path):
    p = tmp_path / "ck.pt"
    torch.save({"a": 1}, p)
    h = sha256_file(p)
    # 篡改内容
    torch.save({"a": 2}, p)
    with pytest.raises(ValueError, match="不匹配"):
        safe_torch_load(p, sha256=h)


def test_ci_gate_flags_unverified_torch_load(tmp_path):
    """check_load_revision 必须能抓到裸 torch.load。"""
    import subprocess, sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\nck = torch.load('x.pt')\n")
    r = subprocess.run([sys.executable, str(repo / "scripts/check_load_revision.py"), str(bad)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "torch.load" in r.stderr


def test_ci_gate_ignores_numpy_load(tmp_path):
    """np.load 不是 torch.load，不得误报。"""
    import subprocess, sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    ok = tmp_path / "ok.py"
    ok.write_text("import numpy as np\nx = np.load('a.npy')\n")
    r = subprocess.run([sys.executable, str(repo / "scripts/check_load_revision.py"), str(ok)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
