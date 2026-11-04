"""离线、可复现的源码打包器；不调用 Kaggle，也不启动训练。"""
import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def build_payload(root):
    root = Path(root)
    files = {}
    for directory in ('src', 'scripts'):
        for p in sorted((root / directory).rglob('*')):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.yaml', '.yml'):
                files[p.relative_to(root).as_posix()] = p.read_bytes()
    for name in ('train.py', 'requirements.txt', 'SOURCE_LOCK.md', 'pytest.ini'):
        if (root / name).is_file():
            files[name] = (root / name).read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    files['payload-manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w', format=tarfile.USTAR_FORMAT) as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return base64.b64encode(gzip.compress(raw.getvalue(), mtime=0)).decode(), manifest


def verify_payload(payload):
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(payload, validate=True))), mode='r:') as archive:
        files = {}
        for member in archive.getmembers():
            p = Path(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in files:
                raise ValueError('Unsafe or duplicate archive member')
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files.pop('payload-manifest.json'))
    actual = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if actual != manifest:
        raise ValueError('Payload manifest mismatch')
    return files, manifest


EXTRACT = '''def do_extract():
    import hashlib, io, tarfile
    from pathlib import Path
    destination = Path('/kaggle/working/model')
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD.strip())), mode='r:gz') as archive:
        files = {}
        for member in archive.getmembers():
            p = Path(member.name)
            if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in files:
                raise ValueError('Unsafe payload member')
            files[member.name] = archive.extractfile(member).read()
    manifest = json.loads(files.pop('payload-manifest.json'))
    if {n: hashlib.sha256(v).hexdigest() for n, v in files.items()} != manifest:
        raise ValueError('Payload checksum mismatch')
    for name, data in files.items():
        p = destination / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest[name]
    (destination / 'payload-manifest.json').write_text(json.dumps(manifest, indent=2))
    os.chdir(destination)
    sys.path[:0] = [str(destination), str(destination / 'src')]
    log('Current payload extracted and SHA256 verified')
'''
PROBE = '''def do_ab_probe():
    from scripts.ab_probe import run_probe
    return run_probe(planned_shards=N_SHARDS, log=log)
'''


def render_kernel(template, payload):
    tree = ast.parse(template)
    lines = template.splitlines(keepends=True)
    replacements = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in node.targets):
            replacements.append((node.lineno, node.end_lineno, 'PAYLOAD = ' + repr(payload) + '\n'))
            found.add('PAYLOAD')
        if isinstance(node, ast.FunctionDef) and node.name in ('do_extract', 'do_ab_probe'):
            replacements.append((node.lineno, node.end_lineno, EXTRACT if node.name == 'do_extract' else PROBE))
            found.add(node.name)
    if found != {'PAYLOAD', 'do_extract', 'do_ab_probe'}:
        raise ValueError('Unsupported kernel template')
    for start, end, text in sorted(replacements, reverse=True):
        lines[start - 1:end] = [text]
    result = ''.join(lines)
    start = result.find('    # 实际 token 占比（用于记录真实 mix，而非名义权重）')
    if start >= 0:
        end = result.index('    buf=mixed.tolist()', start)
        result = result[:start] + '''    # 各来源已按配额截断；总目标不少于所有来源的实际 token 总数。
    source_tokens = {name: len(ids) for name, ids in streams.items()}
    assert mixed.size == sum(source_tokens.values()), 'Source accounting mismatch'
    man['source_actual_tokens'] = source_tokens
    log(f"source actual tokens: {source_tokens}; interleaved={mixed.size}")
''' + result[end:]
    compile(result, '<kernel>', 'exec')
    return result


def cpu_smoke(files):
    """从解码后的当前源码构造两组小尺寸 CPU 模型，验证参数接口。"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, data in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        code = '''import sys
sys.path[:0] = [sys.argv[1], sys.argv[1] + '/src']
from scripts.ab_probe import model_kwargs
from hcl_tr.models.backbone import HybridBackbone
from hcl_tr.train.loop import TrainConfig
import torch
for a in (16., 2.4):
    torch.manual_seed(0)
    kw = model_kwargs(a)
    kw.update(vocab_size=64, d_model=32, n_layer=4, n_head=2)
    model = HybridBackbone(**kw)
    cfg = TrainConfig(seed=0)
    assert next(model.parameters()).device.type == 'cpu'
    print(f'CPU constructor OK: A_log_max={a}, dt=[{kw["gdn2_dt_min"]},{kw["gdn2_dt_max"]}], seed={cfg.seed}')
'''
        subprocess.run([sys.executable, '-c', code, str(root)], check=True, cwd=root)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--kernel', type=Path, required=True, help='Existing local kernel template to update')
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    payload, manifest = build_payload(args.root)
    files, decoded_manifest = verify_payload(payload)
    assert manifest == decoded_manifest
    kernel = render_kernel(args.kernel.read_text(), payload)
    # 从生成的 kernel 再次解析并校验实际 payload，不执行 kernel。
    assignment = next(n for n in ast.parse(kernel).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
    verify_payload(ast.literal_eval(assignment.value))
    if args.smoke:
        cpu_smoke(files)
    metadata = json.loads(args.metadata.read_text())
    metadata.update(code_file=args.kernel.name)
    args.kernel.write_text(kernel)
    args.metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    args.kernel.with_suffix('.manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(f'LOCAL ONLY: verified {len(manifest)} files; updated {args.kernel} and {args.metadata}')


if __name__ == '__main__':
    main()
