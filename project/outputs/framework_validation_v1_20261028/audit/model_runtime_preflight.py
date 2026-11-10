"""Actual train-only numeric/runtime preflight. No validation/test scoring.

Uses the first 128 cached training examples in stored order. Exactly three
AdamW steps per model and task, discarded afterwards. Does not save weights.
"""
from pathlib import Path
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from models import MODEL_NAMES, make_model
from train_study import fit_normalizer, normalize, get_mask, times_array, load_npz, seed_all, sha

OUTPUT = ROOT / 'audit/model_runtime_preflight.json'


def write_record(record):
    OUTPUT.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sync():
    torch.cuda.synchronize()


def finite_gradients(model):
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    return sum(float(g.detach().abs().sum()) for g in grads)


@torch.no_grad()
def cde_diagnostic(model, x, mask, times):
    was_training = model.training
    model.eval()
    assert model.rk4_substeps == 4
    four = model(x, mask, times)
    try:
        model.rk4_substeps = 8
        eight = model(x, mask, times)
    finally:
        model.rk4_substeps = 4
        model.train(was_training)
    assert all(torch.isfinite(v).all() for v in four.values())
    assert all(torch.isfinite(v).all() for v in eight.values())
    diff = (four['logits'] - eight['logits']).abs()
    pdiff = (four['logits'].sigmoid() - eight['logits'].sigmoid()).abs()
    embedding_relative = (four['embedding'] - eight['embedding']).norm(dim=1) / eight['embedding'].norm(dim=1).clamp_min(1e-8)
    return {'logit_absolute_difference_max': float(diff.max()), 'logit_absolute_difference_mean': float(diff.mean()),
            'logit_absolute_difference_p95': float(torch.quantile(diff, .95)),
            'sigmoid_absolute_difference_max': float(pdiff.max()), 'sigmoid_absolute_difference_mean': float(pdiff.mean()),
            'embedding_relative_l2_difference_max': float(embedding_relative.max()),
            'engineering_attention_flag': bool((diff.max() > .05) | (pdiff.max() > .01)),
            'attention_flag_definition': 'max absolute logit difference > 0.05 or sigmoid difference > 0.01; numeric diagnostic only',
            'frozen_substeps_after_diagnostic': model.rk4_substeps}


def run_one(name, task, x, mask, times, y, protocol):
    seed_all(10000)
    torch.cuda.empty_cache()
    model = make_model(name, x.shape[-1], max_len=20, hidden=64).cuda()
    decay = [p for p in model.parameters() if not getattr(p, '_no_weight_decay', False)]
    nodecay = [p for p in model.parameters() if getattr(p, '_no_weight_decay', False)]
    groups = [{'params': decay, 'weight_decay': protocol['training']['weight_decay']}]
    if nodecay:
        groups.append({'params': nodecay, 'weight_decay': 0.0})
    opt = torch.optim.AdamW(groups, lr=protocol['training']['learning_rate'])
    result = {'task': task, 'backbone': name, 'input_dim': x.shape[-1], 'batch_size': len(y),
              'parameters': sum(p.numel() for p in model.parameters()), 'steps': []}
    if name == 'Neural-CDE':
        result['cde_initial_4_vs_8'] = cde_diagnostic(model, x, mask, times)
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        model.train()
        opt.zero_grad(set_to_none=True)
        sync()
        start = time.perf_counter()
        out = model(x, mask, times)
        loss = F.binary_cross_entropy_with_logits(out['logits'], y)
        sync()
        forward_seconds = time.perf_counter() - start
        assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in out.values())
        start = time.perf_counter()
        loss.backward()
        sync()
        backward_seconds = time.perf_counter() - start
        gradient_l1 = finite_gradients(model)
        assert gradient_l1 > 0
        start = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sync()
        optimizer_seconds = time.perf_counter() - start
        assert all(torch.isfinite(p).all() for p in model.parameters())
        row = {'step': step + 1, 'loss': float(loss.detach()), 'logit_abs_max': float(out['logits'].detach().abs().max()),
               'embedding_abs_max': float(out['embedding'].detach().abs().max()), 'gradient_l1_before_clipping': gradient_l1,
               'gradient_l2_norm_before_clipping': float(grad_norm), 'forward_seconds': forward_seconds,
               'backward_seconds': backward_seconds, 'optimizer_clip_step_seconds': optimizer_seconds,
               'measured_compute_seconds': forward_seconds + backward_seconds + optimizer_seconds,
               'finite': True}
        result['steps'].append(row)
    result['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    result['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
    result['steady_compute_seconds_mean_steps_2_3'] = float(np.mean([r['measured_compute_seconds'] for r in result['steps'][1:]]))
    result['steady_compute_seconds_min_steps_2_3'] = min(r['measured_compute_seconds'] for r in result['steps'][1:])
    result['steady_compute_seconds_max_steps_2_3'] = max(r['measured_compute_seconds'] for r in result['steps'][1:])
    if name == 'Neural-CDE':
        result['cde_after_three_steps_4_vs_8'] = cde_diagnostic(model, x, mask, times)
    result['passed'] = True
    print(json.dumps({'task': task, 'model': name, 'passed': True,
                      'steady_compute_s': result['steady_compute_seconds_mean_steps_2_3'],
                      'peak_allocated_mib': result['peak_allocated_mib'],
                      'losses': [r['loss'] for r in result['steps']],
                      'cde_diagnostic': result.get('cde_after_three_steps_4_vs_8')}, ensure_ascii=False), flush=True)
    del model, opt, groups, decay, nodecay, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Preflight must own physical GPU0 only'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((ROOT / 'data_manifest.json').read_text())
    assert manifest['status'] == 'complete_and_audited'
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else None
    result = {'status': 'running', 'purpose': 'train-only numeric stability and rough compute preflight; no AP selection',
              'python': platform.python_version(), 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0),
              'physical_gpu': 0, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'precision': 'float32; TF32 disabled',
              'deterministic_algorithms': True, 'cpu_threads': 2, 'initialization_seed': 10000,
              'sampling': 'first 128 training rows in cache order, same fixed C3 batch for exactly three optimizer steps',
              'read_data_splits': ['train'], 'validation_test_metrics_read': False, 'weights_saved': False,
              'hashes': {name: sha(ROOT / name) for name in ['models.py', 'train_study.py', 'protocol.json', 'data_manifest.json']},
              'task_inputs': {}, 'models': [], 'timing_scope': 'synchronized wall time; forward+BCE, backward, clipping+optimizer; excludes extra audit finite checks; first step warmup excluded from steady estimate',
              'timing_limitations': 'two steady observations per model on one repeated batch; not a controlled full training benchmark; excludes validation, checkpoint I/O and seven-process contention'}
    if previous is not None:
        result['prior_model_version_preflight'] = previous
    result['neural_cde_fixed_substeps'] = 4
    result['neural_cde_refinement_reason'] = 'Coordinator selected four before code freeze based on train-only numerical refinement, not validation/test scores'
    write_record(result)
    for task in ('individual', 'relation'):
        train_path = ROOT / manifest['tasks'][task]['train']['path']
        assert sha(train_path) == manifest['tasks'][task]['train']['sha256']
        train = load_npz(train_path)
        n = len(train['y'])
        assert n >= 128
        norm = fit_normalizer(train, task)
        subset = {k: v[:128] if isinstance(v, np.ndarray) and len(v.shape) > 0 and len(v) == n else v for k, v in train.items()}
        xx, _ = normalize(subset, 'base__C3', norm)
        mm, tt = get_mask(subset), times_array(subset)
        ids = {k: subset[k].astype(str).tolist() for k in ('window_id', 'source_id', 'mmsi') if k in subset}
        result['task_inputs'][task] = {'train_path': str(train_path.relative_to(ROOT)), 'train_sha256': sha(train_path),
                                      'n_train': n, 'batch_rows': [0, 127], 'ids': ids, 'normalizer': norm,
                                      'input_dim': xx.shape[-1], 'valid_lengths_min_max': [int(mm.sum(1).min()), int(mm.sum(1).max())],
                                      'elapsed_seconds_min_max': [float(tt[mm].min()), float(tt[mm].max())],
                                      'continuous_abs_max': float(np.abs(xx[..., :-1]).max()), 'support_max': float(xx[..., -1].max())}
        x, mask, times, y = torch.tensor(xx, device='cuda'), torch.tensor(mm, device='cuda'), torch.tensor(tt, device='cuda'), torch.tensor(subset['y'], dtype=torch.float32, device='cuda')
        del train, subset, xx, mm, tt
        for name in MODEL_NAMES:
            try:
                item = run_one(name, task, x, mask, times, y, protocol)
                result['models'].append(item)
                write_record(result)
            except Exception as exc:
                result['status'] = 'failed'
                result['failure'] = {'task': task, 'model': name, 'exception': repr(exc)}
                write_record(result)
                raise
        del x, mask, times, y
        gc.collect()
        torch.cuda.empty_cache()
    # Training-compute-only extrapolation, deliberately not an end-to-end ETA.
    # Five configurations per task/backbone/seed: C0-C3 one pass; C4 two passes.
    per_model = [r['steady_compute_seconds_mean_steps_2_3'] for r in result['models']]
    one_pass_gpu_seconds = sum(per_model) * 25 * 64 * 3 * 5
    two_pass_C4_proxy = sum(per_model) * 25 * 64 * 3 * 6
    result['projection_requested_25epochs_64batches_240runs'] = {
        'single_pass_all_arms_training_compute_gpu_hours': one_pass_gpu_seconds / 3600,
        'C4_two_pass_proxy_training_compute_gpu_hours': two_pass_C4_proxy / 3600,
        'ideal_seven_gpu_wall_hours_C4_proxy': two_pass_C4_proxy / 7 / 3600,
        'assumption': '16 task/backbone timing cells x 3 seeds x (4 single-pass configs + one 2-pass C4 proxy); every run 25 epochs and 64 batches; ideal equal load',
        'not_end_to_end_eta': True,
        'excluded': ['validation inference every epoch', 'checkpoint writing', 'CPU and storage contention', 'load imbalance', 'C4 perturbation and JS overhead', 'distribution of real batch lengths and final batch size']}
    result['status'] = 'passed'
    result['gpu_released_on_process_exit'] = True
    write_record(result)
    print('PREFLIGHT_COMPLETE', json.dumps(result['projection_requested_25epochs_64batches_240runs']), flush=True)


if __name__ == '__main__':
    main()
