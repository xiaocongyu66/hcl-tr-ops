"""基准控制面的 CPU 测试；不启动 CUDA、不执行 GPU 子进程。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import contextlib
import gc
import os
import selectors
import signal
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    'benchmark_gdn2', Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_gdn2.py')
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


@pytest.mark.parametrize('warmup', [0, 1, 2])
@pytest.mark.parametrize('amp', [False, True])
def test_profile_releases_previous_output(monkeypatch, tmp_path, warmup, amp):
    import hcl_tr.models.backbone as backbone
    import hcl_tr.models.rope_fused as rope
    if amp:
        try:
            torch.ones(1, dtype=torch.float16).float()
        except RuntimeError as exc:
            if 'cpuinfo' in str(exc).lower():
                pytest.skip('PyTorch CPU cpuinfo 初始化失败，无法执行 half 转换桩')
            raise
    refs, alive, order, layers, rope_inputs = [], [], [], [], []
    phase = [None]

    class Layer(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.rope = SimpleNamespace(inv_freq=torch.zeros(32))
            layers.append(self)

        def forward(self, x):
            gc.collect()
            order.append(('forward', phase[0]))
            alive.append(sum(ref() is not None for ref in refs))
            output = x * self.weight
            refs.append(weakref.ref(output))
            return output

    class Profile:
        def __init__(self, **kwargs):
            self.phase = 'measured' if 'profile_memory' in kwargs else 'warmup'

        def __enter__(self):
            phase[0] = self.phase
            order.append(('enter', self.phase))
            return self

        def __exit__(self, *args):
            order.append(('exit', self.phase))
            phase[0] = None

        def step(self):
            order.append(('step', self.phase))

        def export_chrome_trace(self, path):
            order.append(('export', self.phase))

        def key_averages(self, **kwargs):
            return []

    randn = torch.randn
    monkeypatch.setattr(backbone, 'FullAttention', Layer)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch, 'randn', lambda *a, **kw: randn(*a, **dict(kw, device='cpu')))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext())
    monkeypatch.setattr(torch.profiler, 'profile', lambda **kw: Profile(**kw))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)

    def reset():
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert layers[0].weight.grad is None
        order.append(('reset', None))

    def operator(q, k, cos, sin):
        rope_inputs.append((q, k))
        order.append(('rope', None))
        assert q.dtype == k.dtype == (torch.float16 if amp else torch.float32)
        # CPU 桩只验证控制流；GPU 精度由 RoPE 专项测试覆盖。
        return q.float() * cos[None, :, None, :].repeat(1, 1, 1, 2), k.float() * 0.3

    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    monkeypatch.setattr(rope, 'fused_rope', operator)
    monkeypatch.setattr(rope, 'reference_rope', operator)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    args = bench.parser().parse_args(['--output-dir', str(tmp_path), '--warmup', str(warmup),
                                      '--profile-steps', '3'])
    case = next(c for c in bench.profile_cases() if c['backend'] == 'full_attention_triton'
                and c['dtype'] == ('amp_fp32' if amp else 'fp32'))
    case['shape'][1] = 3
    bench.profile_case(case, args)
    # 首次编译预热 + 可选额外预热 + 三次测量，测量后不得再 forward。
    assert alive == [0] * (max(1, warmup) + 3)
    assert order[:3] == [('enter', 'warmup'), ('forward', 'warmup'), ('exit', 'warmup')]
    reset_index = order.index(('reset', None))
    assert order[reset_index + 1:] == [
        ('enter', 'measured'),
        ('forward', 'measured'), ('step', 'measured'),
        ('forward', 'measured'), ('step', 'measured'),
        ('forward', 'measured'), ('step', 'measured'),
        ('exit', 'measured'), ('export', 'measured')]
    assert len(rope_inputs) == 2
    assert all(a is b for a, b in zip(rope_inputs[0], rope_inputs[1]))
    assert order.index(('rope', None)) < reset_index
    assert case['rope_check']['input_dtype'] == str(torch.float16 if amp else torch.float32)
    assert case['rope_check']['output_dtypes'] == [case['rope_check']['input_dtype']] * 2
    assert case['rope_check']['grad_dtypes'] == [case['rope_check']['input_dtype']] * 2
    assert case['status'] == 'passed'


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU 测试禁止调用 CUDA')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_device_name', forbidden)


def test_cli_defaults_and_controls():
    args = bench.parser().parse_args([])
    assert args.mode == 'benchmark'
    args = bench.parser().parse_args(['--mode', 'calibrate', '--warmup', '0', '--reps', '2',
                                     '--confirm-reps', '7', '--seed', '19',
                                     '--case-timeout', '1.5', '--budget-seconds', '9'])
    assert (args.warmup, args.reps, args.confirm_reps, args.seed) == (0, 2, 7, 19)
    assert args.case_timeout == 1.5
    assert bench.parser().parse_args(['--mode', 'profile']).profile_steps == 1


@pytest.mark.parametrize('flag,value', [('--warmup', '-1'), ('--reps', '0'),
    ('--confirm-reps', '0'), ('--case-timeout', 'nan'), ('--budget-seconds', 'inf'),
    ('--case-timeout', '-2'), ('--profile-steps', '0'), ('--shortlist', '10'), ('--seed', '-1')])
def test_invalid_options(flag, value):
    with pytest.raises(SystemExit):
        bench.parser().parse_args([flag, value])


def test_safe_case_grid_and_real_effective_chunk():
    from hcl_tr.models.gdn2_ref import _effective_chunk
    cases = bench.calibration_cases()
    assert len(cases) == 2 * 2 * 9 * 4
    assert len({c['id'] for c in cases}) == len(cases)
    assert bench.summarize(cases, 0)['status'] == 'running'
    assert all(c['chunk_requested'] == 512 and c['decay'] == 2.0
               for c in cases if c['target_chunk'] == 20)
    assert {c['shape'][1] for c in cases} == {2048, 8192}
    assert {c['window'] for c in cases} == {1, 4, 8, 16}
    for case in cases:
        assert case['shape'][0::2] == [1, 10, 64]
        assert case['shape'][3] == 64
        assert 'chunk_override' not in case and 'chunk_eff_in' not in case
        # 用衰减上界验证最坏情况，而非只依赖随机样本碰巧安全。
        base = [None, None, None, torch.full((1, 1, 1, 1), -case['decay'])]
        safety = bench.validate_chunk(base, case)
        effective = _effective_chunk(base[3], case['chunk_requested'], torch.float32, None, None)
        assert effective == safety['predicted_chunk'] == case['target_chunk']
        assert effective * safety['dmax'] <= 40
    bad = dict(cases[0], decay=20)
    with pytest.raises(AssertionError):
        bench.validate_chunk([None, None, None, torch.tensor([-20.])], bad)


def test_seed_and_reference_output_six_gradients():
    base, cotangent = bench.inputs((1, 5, 2, 3, 4), 0.1, seed=4)
    again, _ = bench.inputs((1, 5, 2, 3, 4), 0.1, seed=4)
    other, _ = bench.inputs((1, 5, 2, 3, 4), 0.1, seed=5)
    assert all(torch.equal(a, b) for a, b in zip(base, again))
    assert not torch.equal(base[0], other[0])
    expected = bench.cpu_reference(base, cotangent)
    checks = {'reference': bench.compare(expected, expected, 3e-5, 3e-4)}
    assert bench.checks_pass(checks)
    del checks['reference']['g']
    assert not bench.checks_pass(checks)
    assert not bench.checks_pass({})
    assert not bench.metrics(torch.tensor([float('nan')]), torch.ones(1), 1e-5, 1e-5)['passed']


def test_result_status_and_shortlist(tmp_path):
    cases = bench.calibration_cases()[:4]
    for i, case in enumerate(cases):
        case.update(status='passed', speedup_wall=i + 1, stage='screen_2k')
    assert bench.shortlist_chunks(cases, 2) == {('fp32', 4)}
    cases[0]['status'] = 'failed'
    cases[1].update(status='skipped', reason='budget_exhausted')
    summary = bench.summarize(cases, 1.0)
    assert summary['status'] == 'failed'
    assert summary['passed'] == 2
    path = tmp_path / 'summary.json'
    bench.save_json(path, summary)
    assert json.loads(path.read_text()) == summary
    assert not path.with_suffix('.tmp').exists()


def test_case_timeout_and_stale_result(monkeypatch, tmp_path):
    args = bench.parser().parse_args(['--output-dir', str(tmp_path), '--case-timeout', '3'])
    case = bench.calibration_cases()[0]
    result = tmp_path / (case['id'] + '.json')
    bench.save_json(result, dict(case, status='passed'))

    def timeout(command, **kw):
        assert kw['timeout'] == 2
        assert json.loads(result.read_text())['status'] == 'running'
        assert '--worker-case' in command
        raise subprocess.TimeoutExpired(command, kw['timeout'])

    monkeypatch.setattr(bench, 'managed_process', timeout)
    record = bench.run_case(case, args, remaining=2)
    assert record['status'] == 'timeout'
    assert json.loads(result.read_text())['status'] == 'timeout'


@pytest.mark.parametrize('returncode,reported,expected', [(1, 'passed', 'error'),
    (0, 'running', 'error'), (1, 'failed', 'failed'), (0, 'passed', 'passed')])
def test_worker_exit_not_false_pass(monkeypatch, tmp_path, returncode, reported, expected):
    args = bench.parser().parse_args(['--output-dir', str(tmp_path)])
    case = bench.calibration_cases()[0]

    def finish(command, **kw):
        bench.save_json(tmp_path / (case['id'] + '.json'), dict(case, status=reported))
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(bench, 'managed_process', finish)
    assert bench.run_case(case, args, remaining=2)['status'] == expected


def test_staged_budget_and_all_skips_recorded(monkeypatch, tmp_path):
    args = bench.parser().parse_args(['--mode', 'calibrate', '--output-dir', str(tmp_path),
                                     '--shortlist', '1'])
    calls = []

    def fake(case, args, remaining):
        calls.append((case['shape'][1], case['stage'], case['reps']))
        case.update(status='passed', speedup_wall=case['target_chunk'])
        return case

    monkeypatch.setattr(bench, 'run_case', fake)
    assert bench.orchestrate(args) == 0
    assert len(calls) == 18 + 6 + 8
    assert all(length == 2048 for length, _, _ in calls[:18])
    assert all(length == 8192 for length, _, _ in calls[18:20])
    assert all(reps == args.confirm_reps for length, _, reps in calls if length == 8192)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['status'] == 'partial'
    assert len(summary['cases']) == 144
    assert all(c['status'] != 'pending' for c in summary['cases'])
    for case in summary['cases']:
        if case['status'] == 'skipped':
            assert case['reason']
            assert (tmp_path / (case['id'] + '.json')).exists()


def test_exhausted_budget_never_launches(monkeypatch, tmp_path):
    args = bench.parser().parse_args(['--mode', 'calibrate', '--output-dir', str(tmp_path),
                                     '--budget-seconds', '1'])
    ticks = iter([0] + [2] * 1000)
    monkeypatch.setattr(bench.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(bench, 'run_case', lambda *a: pytest.fail('预算耗尽不得启动子进程'))
    assert bench.orchestrate(args) == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['passed'] == 0
    assert len(summary['skipped']) == 144
    assert all(c['reason'] == 'budget_exhausted' for c in summary['cases'])


@pytest.mark.parametrize('fields,source,value', [
    ({'self_device_time_total': 12, 'device_time_total': 34,
      'self_cuda_time_total': 99, 'cuda_time_total': 99}, 'self_device_time_total', 12),
    ({'self_cuda_time_total': 23, 'cuda_time_total': 45}, 'self_cuda_time_total', 23),
    ({}, 'unavailable', None),
])
def test_profiler_fields(fields, source, value):
    record = bench.profiler_cuda_times(SimpleNamespace(**fields))
    assert record['self_cuda_us'] == value
    assert record['self_cuda_us_source'] == source
    assert record['cuda_timing_supported'] == bool(fields)
    if not fields:
        assert record['cuda_us'] is None
        assert json.loads(json.dumps(record))['self_cuda_us'] is None


def test_run_path_releases_previous_gradients(monkeypatch):
    from hcl_tr.models.gdn2_ref import STATS
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, 'to', lambda self, *a, **kw:
                        self if a == ('cuda',) else original_to(self, *a, **kw))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext())
    for name in ('synchronize', 'reset_peak_memory_stats', 'empty_cache'):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ('max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'Event', lambda **kw: SimpleNamespace(
        record=lambda: None, elapsed_time=lambda other: 1.0))
    refs, calls = [], []
    original_finite = torch.isfinite

    def finite(tensor):
        refs.append(weakref.ref(tensor))
        return original_finite(tensor)

    monkeypatch.setattr(torch, 'isfinite', finite)

    def forward(*leaves, **kw):
        gc.collect()
        # 上一轮 output 与六个 grad 的弱引用必须全部消失。
        assert all(ref() is None for ref in refs)
        refs.clear()
        calls.append(1)
        STATS['chunk_eff'] = 4
        return sum(leaves)

    base = [torch.randn(1, 2, 1, 2) for _ in range(6)]
    bench.run_path(forward, base, torch.ones_like(base[0]), torch.float32, False, 4,
                   warmup=1, reps=2, expected_chunk=4)
    assert len(calls) == 3


@pytest.mark.parametrize('duration', [30, 300])
def test_budget_clock_reserves_both_dtypes(monkeypatch, tmp_path, duration):
    args = bench.parser().parse_args(['--mode', 'calibrate', '--output-dir', str(tmp_path)])
    clock, calls = [0.0], []
    monkeypatch.setattr(bench.time, 'monotonic', lambda: clock[0])

    def fake(case, args, remaining):
        allowance = min(remaining, args.case_timeout)
        clock[0] += min(duration, allowance)
        status = 'passed' if duration <= allowance else 'timeout'
        case.update(status=status, speedup_wall=case['target_chunk'])
        calls.append((case['shape'][1], case['dtype'], case['window'], status))
        return case

    monkeypatch.setattr(bench, 'run_case', fake)
    bench.orchestrate(args)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert clock[0] <= 1100
    assert all(c['status'] != 'pending' for c in result['cases'])
    if duration == 30:
        assert len([c for c in calls if c[0] == 2048 and c[2] == 4]) == 18
        assert {dt for length, dt, _, status in calls if length == 8192 and status == 'passed'} == set(bench.DTYPES)
        assert calls[18][:3] == (8192, 'fp32', 4)
        assert calls[19][:3] == (8192, 'amp_fp32', 4)
    else:
        assert result['status'] == 'failed'
        assert all(c['status'] == 'skipped' and c['reason'] for c in result['cases'] if c['shape'][1] == 8192)


def test_slow_first_confirmation_preserves_amp_slot(monkeypatch, tmp_path):
    args = bench.parser().parse_args(['--mode', 'calibrate', '--output-dir', str(tmp_path)])
    clock, confirmations = [0.0], []
    monkeypatch.setattr(bench.time, 'monotonic', lambda: clock[0])

    def fake(case, args, remaining):
        allowance = min(args.case_timeout, remaining)
        if case['shape'][1] == 2048:
            clock[0] += min(30, allowance)
            case.update(status='passed' if allowance >= 30 else 'timeout', speedup_wall=1)
        else:
            confirmations.append((case['dtype'], allowance))
            clock[0] += allowance
            case['status'] = 'timeout'
        return case

    monkeypatch.setattr(bench, 'run_case', fake)
    assert bench.orchestrate(args) == 1
    assert confirmations[:2] == [('fp32', 180), ('amp_fp32', 180)]
    assert clock[0] <= args.budget_seconds


def test_shortlist_only_common_window():
    cases = bench.calibration_cases()
    for case in cases:
        case.update(status='passed', stage='screen_2k' if case['window'] == 4 else 'shortlist_windows_2k',
                    speedup_wall=(2 if case['target_chunk'] == 20 else 1) if case['window'] == 4 else 999)
    assert bench.shortlist_chunks(cases, 1) == {(dt, 20) for dt in bench.DTYPES}


@pytest.mark.parametrize('failure', ['timeout', 'sigterm'])
def test_real_cpu_worker_cleanup(tmp_path, failure):
    # 保留可独立复跑的真实 CPU 故障注入；管道就绪握手，不依赖固定 sleep。
    # 父进程只加载控制函数 AST，避免测试重复导入 torch 的开销。
    parent_code = '''
import ast, os, signal, subprocess, sys
source = open(sys.argv[1]).read()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'managed_process')
exec(compile(ast.Module(body=[node], type_ignores=[]), sys.argv[1], 'exec'))
child = "import os,signal; print(os.getpid(), flush=True); signal.pause()"
try:
    managed_process([sys.executable, '-u', '-c', child], stdout=None,
                    timeout=1 if sys.argv[2] == 'timeout' else 60)
except subprocess.TimeoutExpired:
    sys.exit(42)
'''
    proc = subprocess.Popen([sys.executable, '-u', '-c', parent_code,
                             str(SPEC.origin), failure], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    child_pid = None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), '子进程未就绪'
        child_pid = int(proc.stdout.readline().strip())
        if failure == 'sigterm':
            os.kill(proc.pid, signal.SIGTERM)
        proc.communicate(timeout=15)
        assert proc.returncode == (42 if failure == 'timeout' else 143)
        # kill(0) 对僵尸也成功，因此这里同时验证已经 wait 回收。
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('detailed', [False, True])
def test_profile_recording_flags_forwarded(monkeypatch, tmp_path, detailed):
    flags = ['--profile-shapes', '--profile-memory'] if detailed else []
    args = bench.parser().parse_args(['--mode', 'profile', '--output-dir', str(tmp_path),
                                     '--profile-backends', 'full_attention_triton', *flags])
    assert args.profile_shapes == args.profile_memory == detailed
    assert args.profile_steps == 1
    case = next(c for c in bench.profile_cases() if c['backend'] == 'full_attention_triton')

    def finish(command, **kwargs):
        worker_args = bench.parser().parse_args(command[2:])
        assert worker_args.profile_shapes == worker_args.profile_memory == detailed
        assert worker_args.profile_backends == ['full_attention_triton']
        bench.save_json(tmp_path / (case['id'] + '.json'), dict(case, status='passed'))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bench, 'managed_process', finish)
    assert bench.run_case(case, args, remaining=10)['status'] == 'passed'


@pytest.mark.parametrize('failed', [False, True])
def test_profile_selected_scope_preserves_failure(monkeypatch, tmp_path, failed):
    args = bench.parser().parse_args(['--mode', 'profile', '--output-dir', str(tmp_path),
                                     '--profile-backends', 'full_attention', 'full_attention_triton'])
    seen = []

    def fake(case, args, remaining):
        seen.append(case)
        case['status'] = 'timeout' if failed else 'passed'
        return case

    monkeypatch.setattr(bench, 'run_case', fake)
    assert bench.orchestrate(args) == int(failed)
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert len(seen) == 8
    assert {c['backend'] for c in seen} == set(args.profile_backends)
    assert all(c['selected_backends'] == args.profile_backends for c in result['cases'])
    assert result['status'] == ('failed' if failed else 'passed')
    assert len(result['failures']) == (8 if failed else 0)


def test_profile_covers_full_layers():
    cases = bench.profile_cases()
    assert len(cases) == 16
    assert {c['backend'] for c in cases} == {'serial', 'batched', 'full_attention',
                                             'full_attention_triton'}
    assert all(c['window'] == 16 for c in cases if c['backend'] == 'batched')
    assert {c['shape'][1] for c in cases} == {2048, 8192}
