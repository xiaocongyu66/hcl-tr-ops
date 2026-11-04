import ast
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ab_probe import model_kwargs, needle_cases, read_curve, token_budget, validate_curve
from scripts.package_probe import build_payload, render_kernel, verify_payload


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': list(text.encode())})()


def test_candidates_only_differ_in_a_log():
    a, b = model_kwargs(16.), model_kwargs(2.4)
    assert [key for key in a if a[key] != b[key]] == ['gdn2_a_log_max']
    assert a['gdn2_dt_min'] == .001 and a['gdn2_dt_max'] == .1
    assert token_budget(2) == (262144, 4980736)
    with pytest.raises(ValueError):
        token_budget(0)


def test_needle_exact_tokens_and_identical_cases():
    cases = needle_cases(CharacterTokenizer())
    assert cases == needle_cases(CharacterTokenizer())
    assert len(cases) == 8
    for case in cases:
        assert len(case['ids']) == case['length']
        pos = case['needle_start']
        assert case['ids'][pos:pos + len(case['needle_ids'])] == case['needle_ids']
        assert case['answer'] in bytes(case['ids']).decode()
        assert case['max_new_tokens'] >= len(case['answer']) + 16


def test_metrics_reject_worker_counts_and_preserve_partial(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text(json.dumps(dict(step=0, tokens_global=262144, loss=3.)) + '\n' +
                    json.dumps(dict(step=1, tokens_global=524288, loss=2.)) + '\n')
    curve = []
    read_curve(path, curve)
    validate_curve(curve, 524288, 262144, 524288)
    with pytest.raises(ValueError):
        validate_curve(curve, 262144, 262144, 524288)
    with path.open('a') as f:
        f.write('{broken')
    with pytest.raises(ValueError):
        read_curve(path, curve)
    assert len(curve) == 2


def test_payload_reproducible_and_rendered(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/example.py').write_text('VALUE = 1\n')
    payload, manifest = build_payload(tmp_path)
    assert build_payload(tmp_path) == (payload, manifest)
    files, verified = verify_payload(payload)
    assert verified == manifest
    assert files == {'src/example.py': b'VALUE = 1\n'}
    (tmp_path / 'SOURCE_LOCK.md').write_text('lock')
    (tmp_path / 'pytest.ini').write_text('[pytest]')
    (tmp_path / 'vendor').mkdir()
    (tmp_path / 'vendor/unused.py').write_text('unused')
    _, full_manifest = build_payload(tmp_path)
    assert 'SOURCE_LOCK.md' in full_manifest and 'pytest.ini' in full_manifest
    assert not any(name.startswith('vendor/') for name in full_manifest)
    template = 'PAYLOAD = "old"\ndef do_extract():\n    pass\ndef do_ab_probe():\n    pass\n'
    rendered = render_kernel(template, payload)
    ast.parse(rendered)
    assert 'run_probe(planned_shards=N_SHARDS' in rendered
    assert 'checksum mismatch' in rendered
    with pytest.raises(Exception):
        verify_payload(payload[:-8] + 'AAAAAAAA')


def test_failed_probe_persists_both_groups(tmp_path, monkeypatch):
    import torch
    import types
    from hcl_tr.train import loop
    from scripts.ab_probe import run_probe
    tokenizers = types.ModuleType('tokenizers')
    tokenizers.Tokenizer = type('FakeTokenizer', (), {
        'from_file': staticmethod(lambda _: CharacterTokenizer())})
    monkeypatch.setitem(sys.modules, 'tokenizers', tokenizers)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(loop, 'TrainConfig', lambda **kwargs: type('Config', (), kwargs)())

    def fail_train(*args, **kwargs):
        cfg = args[2]
        assert args[1] == str(tmp_path / 'tokens')
        assert kwargs['ddp_kwargs']['delete_consumed'] is False
        assert cfg.fields['tokenizer_sha256']
        assert cfg.fields['data_manifest_sha256']
        assert cfg.fields['payload_manifest_sha256']
        out = Path(cfg.out_dir)
        out.mkdir(exist_ok=True)
        (out / 'metrics.jsonl').write_text(json.dumps(dict(step=0, tokens_global=262144, loss=2.)) + '\n')
        raise RuntimeError('simulated worker failure')

    (tmp_path / 'tokenizer').mkdir()
    (tmp_path / 'tokenizer/tokenizer.json').write_text('{}')
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/payload-manifest.json').write_text('{}')
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'tokens').mkdir()
    (tmp_path / 'tokens/shard.bin').write_bytes(b'fixed input')
    monkeypatch.setattr(loop, 'train_ddp', fail_train)
    with pytest.raises(RuntimeError, match='incomplete'):
        run_probe(tmp_path)
    results = json.loads((tmp_path / 'ab_probe.json').read_text())
    assert results['status'] == 'failed'
    for group in results['groups'].values():
        assert group['status'] == 'failed'
        assert len(group['curve']) == 1
        assert 'simulated worker failure' in group['error']
        assert group['input_sha256'] == results['groups']['alog16.0']['input_sha256']
        assert group['tokenizer_sha256'] == results['tokenizer_sha256']
    assert (tmp_path / 'tokens/shard.bin').read_bytes() == b'fixed input'


@pytest.mark.parametrize('wrong_key', ['tokens', 'global_tokens', 'tokens_local'])
def test_metrics_contract_rejects_ambiguous_names(tmp_path, wrong_key):
    path = tmp_path / 'metrics.jsonl'
    path.write_text(json.dumps({'step': 0, wrong_key: 262144, 'loss': 2.}))
    with pytest.raises(KeyError, match='tokens_global'):
        read_curve(path, [])


def test_actual_training_metrics_contract(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    rows = [dict(step=i, steps_completed=i + 1, loss=2.,
                 tokens_local=(i + 1) * 131072, tokens_global=(i + 1) * 262144,
                 world_size=2, global_tok_s=100., microbatches={}, rank=0)
            for i in range(19)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    curve = []
    read_curve(path, curve)
    validate_curve(curve, 4980736, 262144, 4980736)
    assert curve[-1]['tokens_global'] == 4980736
    curve[-1]['step'] = 99
    with pytest.raises(ValueError, match='consecutive'):
        validate_curve(curve, 4980736, 262144, 4980736)


def test_render_preserves_sharding():
    template = ('PAYLOAD = "old"\ndef do_extract():\n    pass\n'
                'def do_ab_probe():\n    pass\n'
                'def do_shards():\n    buf=mixed.tolist()\n    while buf:\n'
                '        arr=buf[:2_000_000]\n        buf=buf[len(arr):]\n'
                '    N_SHARDS = idx\n')
    assert render_kernel(template, 'test').split('def do_shards():')[1] == template.split('def do_shards():')[1]


@pytest.mark.parametrize('size, expected', [(0, 0), (7, 1), (2_000_001, 2)])
def test_repository_template_actual_shard_count(size, expected):
    import numpy as np
    template = (Path(__file__).resolve().parents[1] / 'scripts/kaggle_probe_template.py').read_text()
    for source in (template, render_kernel(template, 'test')):
        tree = ast.parse(source)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'do_shards')
        start = max(i for i, node in enumerate(fn.body)
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'buf' for t in node.targets))
        end = next(i for i, node in enumerate(fn.body)
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'N_SHARDS' for t in node.targets))
        fragment = ast.Module(body=fn.body[start:end + 1], type_ignores=[])
        writes = []
        namespace = dict(np=np, mixed=np.zeros(size, dtype=np.uint16), idx=0, written=0,
                         log=lambda message: None,
                         write_shard=lambda path, index, arr: writes.append((index, arr.size)))
        exec(compile(fragment, '<shard-count>', 'exec'), namespace)
        assert namespace['N_SHARDS'] == len(writes) == expected
        assert [index for index, _ in writes] == list(range(expected))
        assert namespace['written'] == sum(count for _, count in writes) == size


def test_packager_preserves_metadata(tmp_path, monkeypatch):
    from scripts.package_probe import main
    kernel = tmp_path / 'kernel.py'
    metadata = tmp_path / 'kernel-metadata.json'
    kernel.write_text('PAYLOAD = "old"\ndef do_extract():\n    pass\ndef do_ab_probe():\n    pass\n')
    original = dict(enable_internet=True, id='unchanged', code_file='old.py')
    metadata.write_text(json.dumps(original))
    monkeypatch.setattr(sys, 'argv', ['package_probe', '--root', str(tmp_path),
                                    '--kernel', str(kernel), '--metadata', str(metadata)])
    main()
    assert json.loads(metadata.read_text()) == original | {'code_file': 'kernel.py'}
