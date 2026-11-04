"""Standalone CUDA comparison; no training, downloads, or third-party deps beyond torch."""
import argparse
import gc
import statistics
import subprocess
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import torch

NAMES = ('q', 'k', 'v', 'g', 'b', 'w')
SEED = 20261102


def reference(q, k, v, g, b, w):
    """Independent differentiable, token-by-token fp64 recurrence."""
    q, k, v, g, b, w = [x.double() for x in (q, k, v, g, b, w)]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True).clamp_min(1e-12)
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = []
    for t in range(q.shape[1]):
        decayed = state * torch.exp(g[:, t])[..., None]
        erased = ((b[:, t] * k[:, t])[..., None] * decayed).sum(dim=-2)
        state = decayed + k[:, t, :, :, None] * (w[:, t] * v[:, t] - erased)[..., None, :]
        outputs.append((q[:, t, :, :, None] * state).sum(dim=-2) / math.sqrt(t + 1))
    return torch.stack(outputs, dim=1)


def metrics(actual, expected, atol, rtol):
    a, e = actual.double(), expected.double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    if not finite:
        return dict(finite=False, passed=False, atol=atol, rtol=rtol)
    error = (a - e).abs()
    return dict(finite=True, passed=bool((error <= atol + rtol * e.abs()).all()),
                max_abs=error.max().item(), rms=error.square().mean().sqrt().item(),
                relative_l2=(error.norm() / e.norm().clamp_min(1e-30)).item(),
                atol=atol, rtol=rtol)


def inputs(shape, decay, seed=SEED):
    B, T, H, dk, dv = shape
    rng = torch.Generator(device='cpu').manual_seed(seed)
    key, val = (B, T, H, dk), (B, T, H, dv)
    q, k = [torch.randn(key, generator=rng) for _ in range(2)]
    v = torch.randn(val, generator=rng)
    g = -(0.5 + 0.5 * torch.rand(key, generator=rng)) * decay
    b, w = torch.rand(key, generator=rng), torch.rand(val, generator=rng)
    cotangent = torch.randn(val, generator=rng) / math.sqrt(T)
    return (q, k, v, g, b, w), cotangent


def compare(a, e, atol, rtol):
    return {name: metrics(x, y, atol, rtol) for name, x, y in zip(('output',) + NAMES, a, e)}


def cpu_reference(base, cotangent):
    leaves = [x.clone().double().requires_grad_() for x in base]
    output = reference(*leaves)
    (output * cotangent.double()).sum().backward()
    return [output.detach()] + [x.grad.detach() for x in leaves]


def run_path(fn, base, cotangent, md, autocast, chunk, *, warmup=1, reps=3,
             window=None, expected_chunk=None):
    from hcl_tr.models.gdn2_ref import STATS
    leaves = [x.to('cuda').clone().requires_grad_() for x in base]
    weight = cotangent.to('cuda')
    samples, snapshot = [], None
    actual_chunk = None
    kwargs = {} if window is None else {'precompute_chunks': window}
    for iteration in range(warmup + reps):
        for leaf in leaves:
            leaf.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_before_sync = time.perf_counter()
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        with torch.autocast('cuda', dtype=torch.float16, enabled=autocast):
            output = fn(*leaves, chunk=chunk, matmul_dtype=md, solve_mode='solver', **kwargs)
            loss = (output * weight).sum()
        middle.record()
        # 分阶段墙钟包含同步；总时间也包含这个同步，不冒充无同步训练吞吐。
        torch.cuda.synchronize()
        wall_middle = time.perf_counter()
        loss.backward()
        end.record()
        wall_before_end_sync = time.perf_counter()
        torch.cuda.synchronize()
        wall_end = time.perf_counter()
        current_chunk = int(STATS['chunk_eff'])
        if actual_chunk is not None and actual_chunk != current_chunk:
            raise AssertionError('auto chunk changed between repetitions')
        actual_chunk = current_chunk
        if expected_chunk is not None and current_chunk != expected_chunk:
            raise AssertionError(f'实际 C={current_chunk}，预期 C={expected_chunk}')
        sample = dict(gpu_ms=start.elapsed_time(end), wall_ms=(wall_end-wall_start)*1000,
                      total_gpu_ms=start.elapsed_time(end), total_wall_ms=(wall_end-wall_start)*1000,
                      forward_gpu_ms=start.elapsed_time(middle), backward_gpu_ms=middle.elapsed_time(end),
                      forward_wall_ms=(wall_middle-wall_start)*1000,
                      backward_wall_ms=(wall_end-wall_middle)*1000,
                      wall_including_initial_sync_ms=(wall_end-wall_before_sync)*1000,
                      host_enqueue_ms=(wall_before_end_sync-wall_start)*1000,
                      final_sync_ms=(wall_end-wall_before_end_sync)*1000,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        # All checks and CPU copies are outside the measured region.
        for name, tensor in zip(('output',) + NAMES, [output] + [x.grad for x in leaves]):
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise AssertionError('nonfinite/missing ' + name)
        if iteration >= warmup:
            samples.append(sample)
        if iteration == warmup + reps - 1:
            snapshot = [output.detach().cpu()] + [x.grad.detach().cpu() for x in leaves]
        # for 循环变量会持有最后一个梯度；必须在下一轮清空 grad 前释放。
        del tensor, output, loss, start, middle, end
    del leaf, leaves, weight
    gc.collect()
    torch.cuda.empty_cache()
    return snapshot, dict(samples=samples, actual_chunk=actual_chunk,
                          timing_policy='阶段同步，精度检查与复制不计时',
                          median_gpu_ms=statistics.median(x['gpu_ms'] for x in samples),
                          median_wall_ms=statistics.median(x['wall_ms'] for x in samples))


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward
    output_dir = Path(os.environ.get('GDN2_RESULTS', '/kaggle/working/gdn2_results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        dest = output_dir / (name + '.json')
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
        temp.replace(dest)
        print(json.dumps(dict(result=str(dest), status=data.get('status'))), flush=True)
    # External entrypoint adds a hard subprocess timeout, including CUDA hangs.
    def expired(*_):
        raise TimeoutError('benchmark runtime budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(1150)
    failures = []
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required: CPU is not a GPU benchmark')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        save('environment', dict(status='started', torch=torch.__version__, cuda=torch.version.cuda,
             gpu=torch.cuda.get_device_name(0), seed=SEED, warmup=1, measured=3,
             note='Single GPU; no throughput guarantee; no optimizer; fp16 has separate wider tolerance.'))
        configs = [('small_weak', (1, 19, 2, 5, 7), 0.002, 8),
                   ('small_strong', (1, 23, 2, 7, 5), 2.0, 8),
                   ('production', (1, 8192, 10, 64, 64), 0.01, 512),
                   ('production_strong', (1, 8192, 10, 64, 64), 2.0, 512)]
        modes = [('fp32', torch.float32, False, 3e-5, 3e-4),
                 ('autocast_fp32', torch.float32, True, 3e-5, 3e-4),
                 ('explicit_fp16_wide', torch.float16, False, 2e-2, 5e-2)]
        for name, shape, decay, chunk in configs:
            for mode, md, amp, atol, rtol in modes:
                group = name + '_' + mode
                record = dict(status='running', shape=shape, decay=decay, chunk_requested=chunk,
                              chunk_policy='auto', mode=mode, input_dtype='fp32', matmul_dtype=str(md),
                              autocast=amp, atol=atol, rtol=rtol)
                save(group, record)
                base = cotangent = expected = serial = batched = None
                try:
                    base, cotangent = inputs(shape, decay)
                    if name.startswith('small'):
                        expected = cpu_reference(base, cotangent)
                    serial, record['serial'] = run_path(gdn2_chunkwise_forward, base, cotangent, md, amp, chunk)
                    save(group, record)
                    batched, record['batched'] = run_path(gdn2_chunkwise_batched_forward, base, cotangent, md, amp, chunk)
                    if record['serial']['actual_chunk'] != record['batched']['actual_chunk']:
                        raise AssertionError('different actual chunks')
                    record['serial_vs_batched'] = compare(batched, serial, atol, rtol)
                    if expected is not None:
                        record['serial_vs_fp64'] = compare(serial, expected, atol, rtol)
                        record['batched_vs_fp64'] = compare(batched, expected, atol, rtol)
                    checks = [v for k, v in record.items() if '_vs_' in k]
                    record['status'] = 'passed' if all(m['passed'] for c in checks for m in c.values()) else 'failed'
                    if record['status'] == 'failed':
                        failures.append(group)
                except Exception:
                    record.update(status='error', traceback=traceback.format_exc())
                    traceback.print_exc()
                    failures.append(group)
                finally:
                    save(group, record)
                    base = cotangent = expected = serial = batched = None
                    gc.collect()
                    torch.cuda.empty_cache()
                if time.perf_counter() - started > 1100:
                    raise TimeoutError('insufficient remaining runtime budget')
    except Exception:
        failures.append('fatal')
        save('fatal', dict(status='error', traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        signal.alarm(0)
        save('summary', dict(status='failed' if failures else 'passed', failures=failures,
                             elapsed_seconds=time.perf_counter() - started))
    if failures:
        raise SystemExit(1)


# 校准只生成安全输入，不向算子传 chunk_override / chunk_eff_in。
CHUNKS = (4, 8, 16, 20, 32, 64, 128, 256, 512)
WINDOWS = (1, 4, 8, 16)
DTYPES = ('fp32', 'amp_fp32')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('必须为正整数')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('不能为负数')
    return value


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('必须为有限正秒数')
    return value


def parser():
    p = argparse.ArgumentParser(
        description='GDN2 旧基准、安全校准与完整层 CPU/CUDA profile',
        epilog='warmup/reps/seed/预算等新控制参数用于 calibrate/profile；benchmark 保留旧协议。'
               '外层硬超时请由调用者管理整个进程组，例如 timeout -k 10s 1200s python 此脚本。')
    p.add_argument('--mode', choices=('benchmark', 'calibrate', 'profile'), default='benchmark')
    p.add_argument('--output-dir', type=Path, default=Path(os.environ.get('GDN2_RESULTS', '/kaggle/working/gdn2_results')))
    p.add_argument('--warmup', type=nonnegative_int, default=1)
    p.add_argument('--reps', type=positive_int, default=3)
    p.add_argument('--confirm-reps', type=positive_int, default=5)
    p.add_argument('--seed', type=nonnegative_int, default=SEED)
    p.add_argument('--case-timeout', type=positive_seconds, default=180)
    p.add_argument('--budget-seconds', type=positive_seconds, default=1100)
    p.add_argument('--shortlist', type=positive_int, choices=range(1, 10), default=2)
    p.add_argument('--profile-steps', type=positive_int, default=1)
    p.add_argument('--profile-shapes', action='store_true')
    p.add_argument('--profile-memory', action='store_true')
    p.add_argument('--profile-backends', nargs='+',
                   choices=('serial', 'batched', 'full_attention', 'full_attention_triton'),
                   default=['serial', 'batched', 'full_attention', 'full_attention_triton'])
    p.add_argument('--worker-case', type=Path, help=argparse.SUPPRESS)
    return p


def calibration_cases():
    cases = []
    for length in (2048, 8192):
        for dtype in DTYPES:
            for chunk in CHUNKS:
                for window in WINDOWS:
                    # 安全指数预算留余量；不同 C 的衰减不同，不能当作模型等价候选。
                    decay = 2.0 if chunk == 20 else min(2.0, 16.0 / chunk)
                    requested = 512 if chunk == 20 else chunk
                    cases.append(dict(id=f't{length}_{dtype}_c{chunk}_w{window}',
                                      shape=[1, length, 10, 64, 64], dtype=dtype,
                                      input_dtype='torch.float32', matmul_dtype='torch.float32',
                                      target_chunk=chunk, chunk_requested=requested, decay=decay,
                                      window=window, backend='batched', baseline_backend='serial',
                                      status='pending', kind='calibrate'))
    return cases


def profile_cases():
    return [dict(id=f'profile_t{length}_{dtype}_{backend}', kind='profile',
                 shape=[1, length, 10, 64, 64], dtype=dtype, input_dtype='torch.float32',
                 matmul_dtype=('torch.float16' if backend.startswith('full_attention') and dtype == 'amp_fp32'
                               else 'torch.float32'),
                 backend=backend, chunk_requested=512 if not backend.startswith('full_attention') else None,
                 window=16 if backend == 'batched' else None, status='pending')
            for length in (2048, 8192) for dtype in DTYPES
            for backend in ('serial', 'batched', 'full_attention', 'full_attention_triton')]


def environment():
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(0),
                arch=list(torch.cuda.get_device_capability(0)),
                device_total_memory=torch.cuda.get_device_properties(0).total_memory,
                tf32=False)


def save_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def checks_pass(checks):
    # 缺失任何 output 或六组梯度都不能通过。
    return bool(checks) and all(set(group) == set(('output',) + NAMES)
                               and all(m.get('passed') is True for m in group.values())
                               for group in checks.values())


def validate_chunk(base, case):
    dmax = float((-base[3]).amax())
    actual = min(case['chunk_requested'], max(1, int(40.0 / dmax)))
    if not math.isfinite(dmax) or actual != case['target_chunk'] or actual * dmax > 40:
        raise AssertionError('输入不能安全实现目标 C')
    return dict(dmax=dmax, predicted_chunk=actual, safe_exp=40.0,
                policy='auto，无 override；实际 C 必须与目标一致')


def calibrate_case(case, args):
    from hcl_tr.models.gdn2_ref import gdn2_chunkwise_forward, gdn2_chunkwise_batched_forward
    base, cotangent = inputs(case['shape'], case['decay'], args.seed)
    case['safety'] = validate_chunk(base, case)
    atol, rtol = 3e-5, 3e-4
    case.update(atol=atol, rtol=rtol)
    amp = case['dtype'] == 'amp_fp32'
    # 独立 fp64 递推只检查短前缀，完整 2K/8K 检查 serial 对 batched 的 output+6grad。
    # 尽量跨完整批量窗口及尾块；CPU fp64 图昂贵，最多 65 token。
    # 大 C/窗口的跨窗口正确性由完整长度 serial/batched 比较覆盖，不夸大 fp64 范围。
    prefix = min(case['shape'][1], case['target_chunk'] * case['window'] + 1)
    prefix = min(prefix, 65)
    small = tuple(x[:, :prefix].contiguous() for x in base)
    small_cotangent = cotangent[:, :prefix].contiguous()
    expected = cpu_reference(small, small_cotangent)
    case['fp64_reference_tokens'] = prefix
    case['fp64_scope'] = '仅短前缀；完整长度门禁为 serial 对 batched，不声称完整 fp64 验证'
    common = dict(md=torch.float32, autocast=amp, chunk=case['chunk_requested'],
                  warmup=0, reps=1, expected_chunk=case['target_chunk'])
    short_serial, _ = run_path(gdn2_chunkwise_forward, small, small_cotangent, **common)
    short_batched, _ = run_path(gdn2_chunkwise_batched_forward, small, small_cotangent,
                                window=case['window'], **common)
    checks = dict(serial_vs_fp64=compare(short_serial, expected, atol, rtol),
                  batched_vs_fp64=compare(short_batched, expected, atol, rtol))
    case['checks'] = checks
    if not checks_pass(checks):
        case['status'] = 'failed'
        return
    common.update(warmup=args.warmup, reps=case['reps'])
    serial, case['serial'] = run_path(gdn2_chunkwise_forward, base, cotangent, **common)
    batched, case['batched'] = run_path(gdn2_chunkwise_batched_forward, base, cotangent,
                                      window=case['window'], **common)
    case['actual_chunk'] = case['batched']['actual_chunk']
    checks['batched_vs_serial_full'] = compare(batched, serial, atol, rtol)
    case['status'] = 'passed' if checks_pass(checks) else 'failed'
    case['speedup_wall'] = case['serial']['median_wall_ms'] / case['batched']['median_wall_ms']


def profiler_cuda_times(event):
    """兼容新旧 PyTorch；缺失字段不是测得的零耗时。"""
    result = {}
    for key, modern, legacy in (
            ('self_cuda_us', 'self_device_time_total', 'self_cuda_time_total'),
            ('cuda_us', 'device_time_total', 'cuda_time_total')):
        source = next((name for name in (modern, legacy)
                       if getattr(event, name, None) is not None), None)
        result[key] = getattr(event, source) if source else None
        result[key + '_source'] = source or 'unavailable'
    result['cuda_timing_supported'] = all(result[k] is not None for k in ('self_cuda_us', 'cuda_us'))
    return result


def profile_case(case, args):
    from hcl_tr.models.backbone import FullAttention, GDN2Layer
    from torch.profiler import ProfilerActivity, profile, record_function
    backend = case['backend']
    rope_backend = 'triton' if backend == 'full_attention_triton' else 'torch'
    layer = (FullAttention(640, 10, qk_norm=True, rope_backend=rope_backend)
             if backend.startswith('full_attention') else
             GDN2Layer(640, 10, chunk=512, matmul_dtype='fp32', chunk_backend=backend)).cuda()
    x = torch.randn(1, case['shape'][1], 640, device='cuda', requires_grad=True)
    amp = case['dtype'] == 'amp_fp32'
    actual_chunks, dmax_values = [], []

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with record_function(backend + '_完整层_forward'):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                result = layer(x)
                y = result if backend.startswith('full_attention') else result['out']
                loss = y.float().square().mean()
        with record_function(backend + '_完整层_backward'):
            loss.backward()
        if not backend.startswith('full_attention'):
            actual = int(layer._chunk_eff)
            dmax = float(layer._last_dmax)
            predicted = min(layer.chunk, max(1, int(40.0 / dmax))) if dmax > 0 else layer.chunk
            if actual != predicted or actual * dmax > 40.0:
                raise AssertionError('完整层实际 C 未满足安全约束')
            actual_chunks.append(actual)
            dmax_values.append(dmax)
        return y

    def compare_attention():
        # 直接比较 RoPE 算子：两条路径拿到相同 q/k、相同 fp16 autocast，参考在 GPU 上用 PyTorch 算，
        # 因此只隔离 RoPE（含 kernel vs 参考），而不是整层 fp16 对 fp32。
        if backend != 'full_attention_triton':
            return
        from hcl_tr.models.rope_fused import fused_rope, reference_rope
        dtype = torch.float16 if amp else torch.float32
        q = torch.randn(1, 97, 10, 64, device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn(1, 97, 10, 64, device='cuda', dtype=dtype, requires_grad=True)
        cos, sin = layer.rope.inv_freq.new_zeros(97, 32) + 0.3, layer.rope.inv_freq.new_zeros(97, 32) + 0.1
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            got = fused_rope(q, k, cos, sin)
            want = reference_rope(q, k, cos, sin)
        gq, gk = torch.randn_like(got[0]), torch.randn_like(got[1])
        got_grads = torch.autograd.grad(got, (q, k), (gq, gk), retain_graph=True)
        want_grads = torch.autograd.grad(want, (q, k), (gq, gk))
        tolerance = 1e-5 if not amp else 5e-3
        pairs = list(zip(got, want)) + list(zip(got_grads, want_grads))
        for actual, expected in pairs:
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise AssertionError('fused RoPE 输出或梯度非有限')
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        case['rope_check'] = dict(tokens=97, amp=amp, tolerance=tolerance,
                                  input_dtype=str(dtype), cache_dtype=str(cos.dtype),
                                  output_dtypes=[str(t.dtype) for t in got],
                                  grad_dtypes=[str(t.dtype) for t in got_grads],
                                  checked='output_q, output_k, grad_q, grad_k')

    # 首次调用（含编译）必须在正式 profile 前捕获；warmup=0 也不能省略。
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as warm_events:
        y = step()
    warmup_trace = args.output_dir / (case['id'] + '.warmup.trace.json')
    warm_events.export_chrome_trace(str(warmup_trace))
    warm_tensors = [y, x.grad] + [p.grad for p in layer.parameters() if p.requires_grad]
    if any(t is None or not bool(torch.isfinite(t).all()) for t in warm_tensors):
        raise AssertionError('warmup 输出或梯度缺失/非有限')
    y = warm_tensors = None
    for _ in range(max(1, args.warmup) - 1):
        step()
    compare_attention()
    # 清除预热输出和所有梯度引用后再重置峰值，精度检查也不污染测量。
    layer.zero_grad(set_to_none=True)
    x.grad = None
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=args.profile_shapes, profile_memory=args.profile_memory) as prof:
        y = None
        for _ in range(args.profile_steps):
            y = None
            y = step()
            prof.step()
    torch.cuda.synchronize()
    tensors = [y, x.grad] + [p.grad for p in layer.parameters() if p.requires_grad]
    if any(t is None or not bool(torch.isfinite(t).all()) for t in tensors):
        raise AssertionError('完整层输出或梯度缺失/非有限')
    y = tensors = None
    trace = args.output_dir / (case['id'] + '.trace.json')
    prof.export_chrome_trace(str(trace))
    events = []
    for event in prof.key_averages(group_by_input_shape=args.profile_shapes):
        events.append(dict(name=event.key, count=event.count, input_shapes=event.input_shapes,
                           self_cpu_us=event.self_cpu_time_total, cpu_us=event.cpu_time_total,
                           **profiler_cuda_times(event)))
    case.update(status='passed', validation='仅完整层有限值检查；profile 不是精度或吞吐验收',
                actual_chunks=actual_chunks, actual_chunk=actual_chunks[-1] if actual_chunks else None,
                dmax_values=dmax_values, safe_exp=40.0 if actual_chunks else None,
                rope_backend=rope_backend,
                cuda_timing_supported=all(e['cuda_timing_supported'] for e in events),
                trace=str(trace), warmup_trace=str(warmup_trace),
                events=sorted(events, key=lambda e: (
                    e['self_cuda_us'] is None, -(e['self_cuda_us'] or 0))),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                note='含完整投影/门控及 FullAttention QK Norm/RoPE；warmup trace 单独保存，首次编译不计入 profile；不能仅凭 kernel 次数判定 CPU-bound。GDN2 层窗口沿用现有默认 16。')


def worker(args):
    case = json.loads(args.worker_case.read_text())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('需要 CUDA，不能用 CPU 冒充 GPU 基准')
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        case['environment'] = environment()
        case.update(seed=args.seed, warmup=args.warmup, profile_steps=args.profile_steps,
                    profile_shapes=args.profile_shapes, profile_memory=args.profile_memory)
        if case['kind'] == 'calibrate':
            calibrate_case(case, args)
        else:
            profile_case(case, args)
    except Exception:
        case.update(status='error', traceback=traceback.format_exc())
    save_json(args.output_dir / (case['id'] + '.json'), case)
    return 0 if case['status'] == 'passed' else 1


def managed_process(command, *, stdout, timeout):
    """独立进程组；超时、父 SIGTERM/中断均杀掉并回收当前 worker。"""
    proc = None
    previous = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminated)
    try:
        proc = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                                start_new_session=True)
        proc.wait(timeout=timeout)
        return proc
    finally:
        # 清理期间忽略重复 SIGTERM，避免 kill/wait 被再次打断。
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)


def run_case(case, args, remaining):
    # 每例超时涵盖 CUDA 卡死；父进程退出也回收 worker。
    started = time.monotonic()
    case.update(status='running', seed=args.seed, warmup=args.warmup,
                timeout_seconds=min(args.case_timeout, remaining))
    request = args.output_dir / (case['id'] + '.request.json')
    result = args.output_dir / (case['id'] + '.json')
    save_json(request, case)
    save_json(result, case)  # 覆盖旧结果，不能误认上次的通过记录。
    command = [sys.executable, str(Path(__file__).resolve()), '--worker-case', str(request),
               '--output-dir', str(args.output_dir), '--seed', str(args.seed),
               '--warmup', str(args.warmup), '--profile-steps', str(args.profile_steps)]
    if args.profile_shapes:
        command.append('--profile-shapes')
    if args.profile_memory:
        command.append('--profile-memory')
    command.extend(['--profile-backends', *args.profile_backends])
    try:
        with (args.output_dir / (case['id'] + '.log')).open('w') as log:
            proc = managed_process(command, stdout=log,
                                   timeout=min(args.case_timeout, remaining))
        record = json.loads(result.read_text())
        if proc.returncode != 0 or record.get('status') != 'passed':
            if record.get('status') not in ('failed', 'error'):
                record.update(status='error', reason=f'子进程退出码 {proc.returncode}')
        case.update(record)
        case['returncode'] = proc.returncode
    except subprocess.TimeoutExpired:
        case.update(status='timeout', reason='逐例或剩余总预算硬超时')
    except (SystemExit, KeyboardInterrupt):
        case.update(status='error', reason='父进程终止，worker 已清理')
        raise
    except Exception:
        case.update(status='error', traceback=traceback.format_exc())
    finally:
        case['elapsed_seconds'] = time.monotonic() - started
        save_json(result, case)
    return case


def shortlist_chunks(cases, count):
    selected = set()
    for dtype in DTYPES:
        ranked = sorted((c for c in cases if c['shape'][1] == 2048 and c['dtype'] == dtype
                         and c['window'] == 4 and c.get('stage') == 'screen_2k'
                         and c['status'] == 'passed'), key=lambda c: -c['speedup_wall'])
        chunks = []
        for c in ranked:
            if c['target_chunk'] not in chunks:
                chunks.append(c['target_chunk'])
            if len(chunks) == count:
                break
        selected.update((dtype, chunk) for chunk in chunks)
    return selected


def summarize(cases, elapsed):
    failures = [c['id'] for c in cases if c['status'] in ('failed', 'error', 'timeout')]
    skipped = [c['id'] for c in cases if c['status'] == 'skipped']
    passed = sum(c['status'] == 'passed' for c in cases)
    pending = any(c['status'] in ('pending', 'running') for c in cases)
    status = 'failed' if failures else ('running' if pending else ('partial' if skipped else 'passed'))
    return dict(status=status,
                failures=failures, skipped=skipped, passed=passed, elapsed_seconds=elapsed,
                cases=cases, note='跳过项不算通过；跨 C 使用不同衰减，只按同一输入的 serial/batched 加速比入围，不生成 auto 阈值。')


def orchestrate(args):
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = calibration_cases() if args.mode == 'calibrate' else [
        c for c in profile_cases() if c['backend'] in args.profile_backends]
    for case in cases:
        if args.mode == 'profile':
            case.update(selected_backends=list(args.profile_backends),
                        profile_shapes=args.profile_shapes, profile_memory=args.profile_memory)
    start = time.monotonic()
    summary = args.output_dir / 'summary.json'

    def execute(case, stage, reps, deadline=None):
        case.update(stage=stage, reps=reps)
        deadline = args.budget_seconds if deadline is None else deadline
        case['stage_deadline_seconds'] = deadline
        case['total_budget_seconds'] = args.budget_seconds
        remaining = min(args.budget_seconds, deadline) - (time.monotonic() - start)
        if remaining <= 0:
            case.update(status='skipped', reason=('budget_exhausted' if
                        time.monotonic() - start >= args.budget_seconds else 'stage_budget_exhausted'))
            save_json(args.output_dir / (case['id'] + '.json'), case)
        else:
            run_case(case, args, remaining)
        save_json(summary, summarize(cases, time.monotonic() - start))
        print(json.dumps(dict(case=case['id'], stage=stage, status=case['status']),
                         ensure_ascii=False), flush=True)

    save_json(summary, summarize(cases, 0))
    if args.mode == 'profile':
        for case in cases:
            execute(case, 'profile', args.profile_steps)
    else:
        # 筛选最多占总预算 55%；共同窗口、相同重复次数，按 C 交错 dtype。
        # 先为两种 dtype 各保留一个 8K 窗口4确认槽位，再花剩余预算扫入围窗口。
        screen_deadline = args.budget_seconds * 0.55
        index = {(c['shape'][1], c['dtype'], c['target_chunk'], c['window']): c for c in cases}
        for chunk in CHUNKS:
            for dtype in DTYPES:
                execute(index[2048, dtype, chunk, 4], 'screen_2k', args.reps, screen_deadline)
        selected = shortlist_chunks(cases, args.shortlist)
        ranked = {dtype: sorted((chunk for dt, chunk in selected if dt == dtype),
                               key=lambda chunk: -index[2048, dtype, chunk, 4]['speedup_wall'])
                  for dtype in DTYPES}
        first = [index[8192, dtype, ranked[dtype][0], 4] for dtype in DTYPES if ranked[dtype]]
        remaining = max(0, args.budget_seconds - (time.monotonic() - start))
        slot = min(args.case_timeout, remaining / len(first)) if first else 0
        for case in first:
            # 每种 dtype 独享一个槽位；第一例变慢不能吃掉第二例的预算。
            deadline = min(args.budget_seconds, time.monotonic() - start + slot)
            execute(case, 'confirm_8k', args.confirm_reps, deadline)
        # 只扫入围 C；额外 2K 窗口最多用掉此时剩余预算的一半。
        elapsed = time.monotonic() - start
        window_deadline = elapsed + max(0, args.budget_seconds - elapsed) * 0.5
        for window in (1, 8, 16):
            for rank in range(args.shortlist):
                for dtype in DTYPES:
                    if rank < len(ranked[dtype]):
                        execute(index[2048, dtype, ranked[dtype][rank], window],
                                'shortlist_windows_2k', args.reps, window_deadline)
        for window in (4, 1, 8, 16):
            for rank in range(args.shortlist):
                for dtype in DTYPES:
                    if rank < len(ranked[dtype]):
                        case = index[8192, dtype, ranked[dtype][rank], window]
                        if case['status'] == 'pending':
                            execute(case, 'confirm_8k', args.confirm_reps)
        for case in cases:
            if case['status'] == 'pending':
                case.update(status='skipped', reason=('budget_exhausted' if
                            time.monotonic() - start >= args.budget_seconds else
                            ('无通过共同窗口筛选的候选，无法保证该 dtype 的 8K 确认'
                             if case['shape'][1] == 8192 and not ranked[case['dtype']] else
                             '预算策略：仅扫描共同窗口入围 C 的其他窗口')))
                save_json(args.output_dir / (case['id'] + '.json'), case)
    result = summarize(cases, time.monotonic() - start)
    save_json(summary, result)
    print(json.dumps(dict(summary=str(summary), status=result['status']), ensure_ascii=False))
    return 1 if result['failures'] or not result['passed'] else 0


def cli(argv=None):
    args = parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    if args.worker_case:
        return worker(args)
    if args.mode == 'benchmark':
        # 无参数仍运行历史基准；新预算、采样参数用于 calibrate/profile。
        os.environ['GDN2_RESULTS'] = str(args.output_dir)
        main()
        return 0
    return orchestrate(args)


if __name__ == '__main__':
    raise SystemExit(cli())
