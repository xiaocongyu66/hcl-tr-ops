"""train.py 入口的配置映射测试（不需要 GPU、不启动训练）。"""
import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('train_entry', REPO / 'train.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_stage0_config_maps_to_accepted_model_kwargs():
    from hcl_tr.models.backbone import HybridBackbone
    import inspect

    cfg = yaml.safe_load((REPO / 'src/hcl_tr/configs/stage0_mvp1a.yaml').read_text())
    kwargs = entry.model_kwargs_from_config(cfg)
    accepted = set(inspect.signature(HybridBackbone.__init__).parameters) - {'self'}
    assert set(kwargs) <= accepted
    assert kwargs['gdn2_chunk'] == 512
    assert kwargs['gdn2_chunk_backend'] == 'fused'
    assert kwargs['gdn2_fused_safe_exp'] == 80.0


def test_unknown_model_key_is_rejected():
    with pytest.raises(SystemExit, match='未知参数'):
        entry.model_kwargs_from_config({'model': {'d_model': 64, 'not_a_param': 1}})


def test_train_config_maps_budget_and_lr():
    cfg = yaml.safe_load((REPO / 'src/hcl_tr/configs/stage0_mvp1a.yaml').read_text())
    tcfg = entry.train_config_from_config(cfg, lr=None, out_dir='out',
                                          resume_from=None, ckpt_sha256=None)
    assert tcfg.seq_len == 8192
    assert tcfg.micro_batch_per_gpu == 2
    assert tcfg.grad_accum == 8
    assert tcfg.global_batch_tokens == 262144
    # lr.sweep 的中位值
    assert tcfg.lr == pytest.approx(3e-4)
    assert tcfg.total_tokens == 10_000_000_000


def test_lr_override_wins():
    cfg = yaml.safe_load((REPO / 'src/hcl_tr/configs/stage0_mvp1a.yaml').read_text())
    tcfg = entry.train_config_from_config(cfg, lr=1e-4, out_dir='out',
                                          resume_from=None, ckpt_sha256=None)
    assert tcfg.lr == pytest.approx(1e-4)


def test_batch_product_mismatch_is_rejected():
    cfg = yaml.safe_load((REPO / 'src/hcl_tr/configs/stage0_mvp1a.yaml').read_text())
    tcfg = entry.train_config_from_config(cfg, lr=None, out_dir='out',
                                          resume_from=None, ckpt_sha256=None)
    entry.validate_batch_product(tcfg, n_gpu=2)          # 2×8×8192×2 = 262144
    with pytest.raises(SystemExit, match='batch 口径不一致'):
        entry.validate_batch_product(tcfg, n_gpu=1)


def test_only_stage0_is_implemented():
    assert entry.IMPLEMENTED_STAGES == {'0'}
