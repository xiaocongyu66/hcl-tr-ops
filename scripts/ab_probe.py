"""本地 A/B 诊断探针；导入模块不会启动训练。"""
import hashlib
import json
import math
import random
from pathlib import Path


def model_kwargs(a_max):
    return dict(vocab_size=64000, d_model=640, n_layer=12, n_head=10,
                grad_checkpointing=True, gdn2_chunk=512,
                gdn2_dt_min=.001, gdn2_dt_max=.1, gdn2_a_log_max=a_max)


def token_budget(world_size, target=5_000_000):
    if world_size < 2:
        raise ValueError('A/B DDP requires at least two GPUs')
    per_step = 8192 * 16 * world_size
    return per_step, target // per_step * per_step


def needle_cases(tok):
    """直接拼接精确长度的 token ID，仅裁剪填充内容，不裁剪 needle。"""
    rng = random.Random(0)
    filler = tok.encode('The grass is green. The sky is blue. ', add_special_tokens=False).ids
    if not filler:
        raise ValueError('Empty filler encoding')
    cases = []
    for length in (2048, 4096):
        for i in range(4):
            answer = f'magic-{rng.randrange(10**6):06d}'
            needle = tok.encode(f'\nThe secret code is {answer}.\n', add_special_tokens=False).ids
            question = tok.encode('\nWhat is the secret code? Answer:', add_special_tokens=False).ids
            remaining = length - len(needle) - len(question)
            if remaining < 0:
                raise ValueError('Needle and question exceed context')
            padding = (filler * (remaining // len(filler) + 1))[:remaining]
            pos = remaining * (i + 1) // 5
            ids = padding[:pos] + needle + padding[pos:] + question
            assert len(ids) == length
            cases.append(dict(length=length, sample=i, answer=answer, ids=ids,
                              needle_start=pos, needle_ids=needle,
                              max_new_tokens=max(32, len(tok.encode(answer, add_special_tokens=False).ids) + 16),
                              input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    return cases


def read_curve(path, curve):
    """严格读取训练端 tokens_global 契约，后续行损坏时保留有效前缀。"""
    curve.clear()
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            tokens = row['tokens_global']
            loss = float(row['loss'])
            if not isinstance(tokens, int) or tokens <= 0 or not math.isfinite(loss):
                raise ValueError('Invalid global token/loss metric')
            curve.append(dict(step=row['step'], tokens_global=tokens, loss=loss))


def validate_curve(curve, actual, per_step, expected):
    if actual != expected or len(curve) != expected // per_step:
        raise ValueError(f'Incomplete training: actual={actual}, expected={expected}, metrics={len(curve)}')
    for i, row in enumerate(curve):
        if row['tokens_global'] != (i + 1) * per_step or row['step'] != i:
            raise ValueError('Metrics are not consecutive global steps')
    if not curve:
        raise ValueError('No training metrics')


def run_probe(work_dir='/kaggle/working', planned_shards=0, log=print):
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from hcl_tr.models.backbone import HybridBackbone
    from hcl_tr.train.loop import TrainConfig, train_ddp
    from hcl_tr.eval.runner import score

    root = Path(work_dir)
    result_path = root / 'ab_probe.json'
    results = dict(status='running', diagnostic_only=True,
                   limitation='Four samples per length; not equivalence validation or Go/NoGo evidence.',
                   groups={f'alog{a}': dict(status='pending', curve=[], needle=[]) for a in (16.0, 2.4)})

    def save():
        tmp = result_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        tmp.replace(result_path)

    save()
    try:
        tokenizer_path = root / 'tokenizer/tokenizer.json'
        results['tokenizer_sha256'] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        tok = Tokenizer.from_file(str(tokenizer_path))
        provenance = {'tokenizer_sha256': results['tokenizer_sha256']}
        for key, path in (
            ('data_manifest_sha256', root / 'manifest.json'),
            ('payload_manifest_sha256', root / 'model/payload-manifest.json'),
        ):
            if path.exists():
                provenance[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif root == Path('/kaggle/working'):
                raise FileNotFoundError(f'部署追溯文件缺失: {path}')
        results['provenance'] = provenance
        cases = needle_cases(tok)
        n = torch.cuda.device_count()
        per_step, expected = token_budget(n)
        results.update(target_tokens=5_000_000, expected_tokens=expected, tokens_per_step=per_step, seed=0)
        for a_max in (16.0, 2.4):
            name = f'alog{a_max}'
            group = results['groups'][name]
            out = root / f'out_{name}'
            mdl = None
            try:
                group.update(status='training', model_kwargs=model_kwargs(a_max), seed=0)
                save()
                # 拒绝复用旧指标或旧检查点，避免误报成功。
                if out.exists() and any(out.iterdir()):
                    raise RuntimeError(f'Output must be empty: {out}')
                # 两组只读同一份分片，分别记录哈希并检查输入一致。
                cache = root / 'tokens'
                group['tokenizer_sha256'] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
                if group['tokenizer_sha256'] != results['tokenizer_sha256']:
                    raise ValueError('Tokenizer changed during A/B probe')
                group['input_sha256'] = {
                    p.relative_to(cache).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(cache.rglob('*')) if p.is_file()}
                previous = results['groups']['alog16.0'].get('input_sha256')
                if previous is not None and previous != group['input_sha256']:
                    raise ValueError('A/B input token files differ')
                save()
                random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                cfg = TrainConfig(seq_len=8192, micro_batch_per_gpu=1, grad_accum=16,
                                  global_batch_tokens=per_step, lr=2e-4,
                                  warmup_tokens=1_000_000, total_tokens=expected,
                                  use_fp16=True, log_every=1, ckpt_every_tokens=expected,
                                  out_dir=str(out), seed=0, fields=dict(provenance))
                actual = train_ddp(model_kwargs(a_max), str(cache), cfg, world_size=n,
                                   ddp_kwargs={'max_cache_bytes': int(6e9), 'planned_shards': planned_shards,
                                               'delete_consumed': False})
                group['actual_tokens'] = actual
                read_curve(out / 'metrics.jsonl', group['curve'])
                save()
                validate_curve(group['curve'], actual, per_step, expected)
                from hcl_tr.checkpoint import safe_torch_load, sha256_file
                from dataclasses import asdict
                group['checkpoint_sha256'] = sha256_file(out / 'final.pt')
                save()
                ck = safe_torch_load(out / 'final.pt', sha256=group['checkpoint_sha256'])
                if ck.get('format_version') != 2 or ck.get('model_config') != model_kwargs(a_max):
                    raise ValueError('Checkpoint model configuration mismatch')
                saved_cfg = ck.get('train_config', {})
                if any(saved_cfg.get(k) != v for k, v in asdict(cfg).items() if k != 'fields'):
                    raise ValueError('Checkpoint training configuration mismatch')
                if ck.get('world_size') != n or ck.get('tokens_global') != expected:
                    raise ValueError('Checkpoint global token contract mismatch')
                if ck['tokens'] != expected:
                    raise ValueError('Checkpoint global tokens disagree with training')
                group.update(status='evaluating', final_loss=group['curve'][-1]['loss'])
                save()
                kw = model_kwargs(a_max)
                kw['grad_checkpointing'] = False
                mdl = HybridBackbone(**kw).cuda().eval()
                mdl.load_state_dict(ck['model'])
                del ck
                for case in cases:
                    ids = case['ids']
                    x = torch.tensor([ids], device='cuda')
                    with torch.inference_mode():
                        for _ in range(case['max_new_tokens']):
                            nxt = mdl(x)[0, -1].argmax().reshape(1, 1)
                            x = torch.cat((x, nxt), dim=1)
                    pred = tok.decode(x[0, len(ids):].tolist())
                    group['needle'].append({k: v for k, v in case.items() if k not in ('ids', 'needle_ids')}
                                           | dict(prediction=pred, correct=bool(score(pred, case['answer']))))
                    save()
                group['status'] = 'complete'
            except Exception as exc:
                group.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                if (out / 'metrics.jsonl').exists():
                    try:
                        read_curve(out / 'metrics.jsonl', group['curve'])
                    except Exception as metric_exc:
                        group['metrics_error'] = str(metric_exc)
            finally:
                del mdl
                torch.cuda.empty_cache()
                save()
            log(f'{name}: {group["status"]}')
        if any(g['status'] != 'complete' for g in results['groups'].values()):
            raise RuntimeError('A/B probe incomplete; see ab_probe.json')
        results['status'] = 'complete'
    except BaseException as exc:
        results.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()
    return results
