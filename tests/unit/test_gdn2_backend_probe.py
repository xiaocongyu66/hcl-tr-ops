"""GDN-2 TileLang 可行性探测的 CPU 可验证部分。

GPU 编译、真实计时必须在 Kaggle 上产生，这里不伪造任何 GPU 结论。
"""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from hcl_tr.models.gdn2_ref import gdn2_recurrent_forward

SPEC = importlib.util.spec_from_file_location(
    'gdn2_backend_probe', Path(__file__).resolve().parents[2] / 'scripts' / 'gdn2_backend_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _tensors(**kw):
    return probe.make_tensors(**kw)


def _batched(**kw):
    chunk = kw.pop('chunk')
    return probe.reference_recurrence(kw['q'], kw['k'], kw['v'], kw['g'], kw['b'], kw['w'])


def test_reference_matches_independent_pytorch_recurrent():
    tensors = _tensors(batch=2, tokens=12, heads=3, dk=6, dv=5)
    expected = gdn2_recurrent_forward(
        tensors['q'], tensors['k'], tensors['v'], tensors['g'], tensors['b'], tensors['w'],
        divide_by_sqrt_t=True, l2norm_qk=True)
    actual = probe.reference_recurrence(
        tensors['q'], tensors['k'], tensors['v'], tensors['g'], tensors['b'], tensors['w'])
    torch.testing.assert_close(actual.float(), expected, atol=1e-5, rtol=1e-5)


def test_compare_backend_accepts_consistent_and_rejects_drifted():
    tensors = _tensors(batch=2, tokens=8, heads=2, dk=4, dv=4)
    good = probe.compare_backend(_batched, tensors, chunk=4)
    assert good['ok'] and good['finite'] and good['max_abs'] <= 1e-9

    def drifted(**kw):
        return _batched(**kw) * 1.01

    bad = probe.compare_backend(drifted, tensors, chunk=4)
    assert not bad['ok'] and bad['max_abs'] > 1e-3


def test_compare_backend_rejects_shape_mismatch():
    tensors = _tensors(batch=1, tokens=4, heads=1, dk=4, dv=4)
    result = probe.compare_backend(lambda **kw: _batched(**kw)[:, :2], tensors, chunk=4)
    assert not result['ok'] and 'shape' in result['reason']


def test_environment_blockers_report_missing_cuda():
    info = probe.environment()
    assert isinstance(info['torch'], str)
    if info['cuda_available']:
        pytest.skip('本机有 CUDA；无 GPU 分支由 Kaggle CPU 环境覆盖')
    assert info['device'] is None
    blockers = probe.environment_blockers(info, require_capability=(7, 5))
    assert any('无 CUDA' in b for b in blockers)


def test_run_probe_returns_report_without_gpu(tmp_path):
    out = tmp_path / 'feasibility.json'
    report = probe.run_probe(output=out)
    assert report['feasible'] is False
    assert report['blockers']
    assert json.loads(out.read_text())['feasible'] is False
    # 未安装 TileLang 时必须记录原因，而不是静默跳过。
    assert report['steps']['tilelang']['ok'] is False
    assert report['steps']['tilelang']['stage'] in {'import', 'device', 'compile'}


def test_profile_breakdown_reports_all_stages_on_cpu():
    result = probe.profile_pytorch_breakdown(device='cpu', tokens=32, chunk=8,
                                             heads=2, dk=4, dv=4, reps=1)
    assert result['ok']
    assert set(result['stages']) == {'prep', 'masks', 'trisolve', 'recurrence', 'total'}
    assert all(v['median_ms'] >= 0 for v in result['stages'].values())


def test_time_callable_uses_wall_clock_without_cuda():
    timing = probe.time_callable(lambda: torch.ones(8).sum(), warmup=1, reps=3)
    assert timing['reps'] == 3 and len(timing['samples']) == 3
    if not torch.cuda.is_available():
        assert timing['source'] == 'wall_ms'
