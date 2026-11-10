"""CPU/CUDA smoke and mathematical checks; does not load data or train models.

Run: python model_tests.py --device cpu --record
--record adds this run under validation_runs in model_inventory.json.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F
from models import MODEL_NAMES, make_model, cde_rk4_step, selective_scan_reference

ROOT = Path(__file__).resolve().parent


def _state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_scan():
    # Isolate only the unmodified upstream pure-PyTorch reference function.
    # This avoids importing its optional compiled CUDA module.
    source = ROOT / 'vendor/mamba/mamba_ssm/ops/selective_scan_interface.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'selective_scan_ref')
    namespace = {'torch': torch, 'F': F, 'rearrange': lambda x, pattern: x[:, None]}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    torch.manual_seed(13)
    b, t, d, n = 2, 5, 4, 3
    u, delta = torch.randn(b, t, d), torch.rand(b, t, d)
    A, B, C, D = -torch.rand(d, n), torch.randn(b, t, n), torch.randn(b, t, n), torch.randn(d)
    got = selective_scan_reference(u, delta, A, B, C, D, torch.ones(b, t, dtype=torch.bool))
    expected = namespace['selective_scan_ref'](u.transpose(1, 2), delta.transpose(1, 2), A, B.transpose(1, 2), C.transpose(1, 2), D).transpose(1, 2)
    error = (got - expected).abs().max().item()
    assert torch.allclose(got, expected, atol=2e-6, rtol=2e-6), error
    return {'upstream_max_abs_error': error, 'upstream_function': 'selective_scan_ref', 'passed': True}


def check_cde_solver():
    z = torch.tensor([[0.7]], dtype=torch.float64)
    dx = torch.tensor([[0.2]], dtype=torch.float64)
    def field(v):
        return v.unsqueeze(-1)
    expected = z * dx.exp()
    whole = cde_rk4_step(z, dx, field)
    half = cde_rk4_step(cde_rk4_step(z, dx / 2, field), dx / 2, field)
    whole_error = (whole - expected).abs().item()
    half_error = (half - expected).abs().item()
    assert half_error < 2e-7 and half_error < whole_error / 10
    constant = cde_rk4_step(z, dx, lambda v: torch.ones_like(v).unsqueeze(-1) * 2)
    assert torch.allclose(constant, z + 2 * dx, atol=1e-12)
    return {'analytic_exp_error_one_step': whole_error, 'analytic_exp_error_two_substeps': half_error, 'passed': True}


def check_model(name, input_dim, device):
    torch.manual_seed(101)
    model = make_model(name, input_dim).to(device)
    torch.manual_seed(101)
    duplicate = make_model(name, input_dim).to(device)
    assert _state_hash(model) == _state_hash(duplicate)
    initial_hash = _state_hash(model)
    del duplicate
    lengths = torch.tensor([20, 11, 1], device=device)
    mask = torch.arange(20, device=device)[None] < lengths[:, None]
    x = torch.randn(3, 20, input_dim, device=device, requires_grad=True)
    with torch.no_grad():
        x[..., -1].fill_(1)
    times = torch.arange(20, device=device, dtype=x.dtype)[None].expand(3, -1) * 60
    start = time.perf_counter()
    model.train()
    output = model(x, mask, times)
    assert output['logits'].shape == (3,) and output['embedding'].shape == (3, 64)
    loss = F.binary_cross_entropy_with_logits(output['logits'], torch.tensor([0., 1., 0.], device=device))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()
    assert x.grad[~mask].abs().max().item() == 0
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    forward_backward_seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        baseline = model(x, mask, times)
        extra_x = torch.cat((x, torch.full((3, 8, input_dim), float('nan'), device=device)), 1)
        extra_mask = F.pad(mask, (0, 8))
        extra_t = F.pad(times, (0, 8), value=float('nan'))
        padded = model(extra_x, extra_mask, extra_t)
        padding_error = max((baseline[k] - padded[k]).abs().max().item() for k in baseline)
        assert padding_error <= 1e-6, (name, padding_error)
        # Excluded interior observations must not affect the compact valid path.
        hole_x = x[:1].clone()
        hole_mask = torch.ones((1, 20), dtype=torch.bool, device=device)
        hole_mask[:, [2, 6, 13]] = False
        hole_x[~hole_mask] = float('nan')
        holes = model(hole_x, hole_mask, times[:1])
        compact_x = x[:1, hole_mask[0]]
        compact_t = times[:1, hole_mask[0]]
        compact = model(compact_x, torch.ones_like(compact_t, dtype=torch.bool), compact_t)
        hole_error = max((holes[k] - compact[k]).abs().max().item() for k in holes)
        assert hole_error <= 1e-6, (name, hole_error)
        single = torch.cat([model(x[i:i+1, :int(lengths[i])], mask[i:i+1, :int(lengths[i])], times[i:i+1, :int(lengths[i])])['logits'] for i in range(3)])
        batch_error = (baseline['logits'] - single).abs().max().item()
        assert torch.allclose(baseline['logits'], single, atol=1e-5, rtol=1e-5), (name, batch_error)
        if name == 'GraphTemporal':
            c0a = x.detach().clone()
            c0a[..., model.target_dim:] = 0
            c0b = c0a.clone()
            c0b[..., model.target_dim:-1] = torch.randn_like(c0b[..., model.target_dim:-1]) * 20
            graph_off_error = (model(c0a, mask, times)['logits'] - model(c0b, mask, times)['logits']).abs().max().item()
            assert graph_off_error == 0
        else:
            graph_off_error = None
    return {'name': name, 'input_dim': input_dim, 'parameters': sum(p.numel() for p in model.parameters()),
            'head_parameters': sum(p.numel() for p in model.head.parameters()),
            'initial_state_sha256_seed101': initial_hash, 'padding_max_abs_error': padding_error,
            'interior_mask_max_abs_error': hole_error, 'batch_vs_single_max_abs_error': batch_error,
            'graph_inactive_reference_max_abs_error': graph_off_error,
            'forward_backward_seconds_batch3': forward_backward_seconds, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = {'python': platform.python_version(), 'torch': torch.__version__, 'device': args.device,
               'torch_threads': 1, 'model_code_sha256': hashlib.sha256((ROOT / 'models.py').read_bytes()).hexdigest(),
               'mamba_reference': check_scan(), 'cde_solver': check_cde_solver(), 'models': []}
    for dim in (12, 7):
        for name in MODEL_NAMES:
            item = check_model(name, dim, args.device)
            results['models'].append(item)
            print(f"PASS {name:14s} input={dim} params={item['parameters']} padding={item['padding_max_abs_error']:.2g} fwd+bwd={item['forward_backward_seconds_batch3']:.3f}s", flush=True)
    results['passed'] = True
    if args.record:
        path = ROOT / 'model_inventory.json'
        inventory = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        inventory.setdefault('validation_runs', []).append(results)
        path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
