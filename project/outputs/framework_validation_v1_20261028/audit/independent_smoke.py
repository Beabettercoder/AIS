"""Independent CPU-only invariants; synthetic inputs, no fitted models or test scores."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utility_functions():
    # Load only audited pure utilities without sklearn or any training entrypoint.
    tree = ast.parse((ROOT / 'train_study.py').read_text(encoding='utf-8-sig'))
    wanted = {'get_mask', 'context_mask', 'fit_normalizer', 'normalize', 'times_array',
              'perturb', 'bernoulli_js', 'thresholds', 'operating', 'state_hash'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in nodes} == wanted
    namespace = dict(np=np, torch=torch, math=math, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'isolated_training_utilities', 'exec'), namespace)
    return namespace


def make_data(nf):
    rng = np.random.default_rng(42)
    target = 7 if nf == 12 else 4
    mask = np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0]], dtype=bool)
    support = np.array([[1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0]], dtype=bool)
    raw = rng.normal(10, 2, (2, 6, nf)).astype('float32')
    raw[..., target:] = 0
    raw[~mask] = 123456  # Must never enter fitted statistics or encoders.
    d = {'mask': mask, 'time_s': np.array([[100, 160, 220, 280, 340, 400], [0, 60, 120, 180, 240, 300]], float)}
    for i in range(5):
        x = raw.copy()
        x[..., target:] = 0
        if i:
            x[..., target:-1] = rng.normal(i, 1, (2, 6, nf - target - 1))
            x[..., target:-1][~support] = 0
            x[..., -1] = support.astype(float)
        d[f'base__C{i}'] = x
    d['base__C4'] = d['base__C3'].copy()
    return d


def test_utilities(f):
    result = []
    for nf, task in ((12, 'individual'), (7, 'relation')):
        d = make_data(nf)
        norm = f['fit_normalizer'](d, task)
        target = norm['target_dim']
        expected = d['base__C0'][d['mask']][:, :target].astype(float)
        np.testing.assert_allclose(norm['mean'][:target], expected.mean(0))
        np.testing.assert_allclose(norm['std'][:target], expected.std(0))
        normalized = {a: f['normalize'](d, 'base__' + a, norm) for a in ['C0', 'C1', 'C2', 'C3', 'C4']}
        for a, (x, fm) in normalized.items():
            assert not np.any(x[~d['mask']])
            np.testing.assert_array_equal(x[..., :target], normalized['C0'][0][..., :target])
            assert not np.any(fm[..., -1])
        assert not np.any(normalized['C0'][0][..., target:])
        np.testing.assert_array_equal(normalized['C3'][0], normalized['C4'][0])
        x, valid = normalized['C3']
        x, valid = torch.tensor(x), torch.tensor(valid)
        gen = torch.Generator().manual_seed(4)
        aug = f['perturb'](x, valid, gen)
        assert torch.equal(aug[~valid], x[~valid])
        assert torch.isfinite(aug).all()
        const = torch.ones_like(x)
        assert torch.equal(f['perturb'](const, valid, torch.Generator().manual_seed(4)), const)
        times = f['times_array'](d)
        np.testing.assert_array_equal(times[1], np.array([0, 240, 120, 180, 240, 240], dtype='float32'))
        result.append({'task': task, 'normalization_padding_support': 'passed'})
    a = torch.tensor([-100., -5., 0., 5., 100.], requires_grad=True)
    b = torch.tensor([100., 5., 0., -5., -100.], requires_grad=True)
    js = f['bernoulli_js'](a, b)
    assert 0 <= js.item() <= math.log(2) + 1e-6
    torch.testing.assert_close(js, f['bernoulli_js'](b, a))
    assert f['bernoulli_js'](a, a).item() == 0
    js.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    y = np.array([0] * 10 + [1] * 3)
    p = np.array([0., .1, .1, .1, .2, .3, .5, .5, .5, .9, .4, .8, 1.])
    th = f['thresholds'](y, p, [0, .01, .1, .2, .5])
    for k, v in th.items():
        assert v['validation_false_positives'] <= math.floor(float(k) * 10)
    return {'normalization': result, 'bernoulli_js': 'passed', 'threshold_ties': 'passed'}


def test_models(f):
    records = []
    torch.set_num_threads(2)
    for nf in (12, 7):
        for name in models.MODEL_NAMES:
            torch.manual_seed(111)
            model = models.make_model(name, nf, hidden=64)
            hash_a = f['state_hash'](model)
            torch.manual_seed(111)
            assert f['state_hash'](models.make_model(name, nf, hidden=64)) == hash_a
            model.eval()
            torch.manual_seed(222)
            x = torch.randn(3, 7, nf)
            mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
            x[..., -1] = torch.tensor([1., 0., 1., 0., 1., 1., 0.])
            times = torch.arange(7).float()[None].expand(3, -1) * 60
            with torch.no_grad():
                clean = model(x, mask, times)
                dirty = x.clone();dirty[~mask] = float('nan')
                dirty_t = times.clone();dirty_t[~mask] = float('nan')
                changed = model(dirty, mask, dirty_t)
                extended = model(torch.cat([dirty, torch.full((3, 8, nf), float('nan'))], 1),
                                 torch.cat([mask, torch.zeros(3, 8, dtype=torch.bool)], 1),
                                 torch.cat([dirty_t, torch.full((3, 8), float('nan'))], 1))
                for key in ('logits', 'embedding'):
                    assert torch.isfinite(clean[key]).all()
                    torch.testing.assert_close(clean[key], changed[key], rtol=0, atol=0)
                    torch.testing.assert_close(clean[key], extended[key], rtol=0, atol=0)
            x.requires_grad_()
            model.zero_grad(set_to_none=True)
            model(x, mask, times)['logits'].square().mean().backward()
            assert torch.isfinite(x.grad).all()
            assert not torch.count_nonzero(x.grad[~mask])
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            record = {'model': name, 'width': nf, 'parameters': sum(p.numel() for p in model.parameters()),
                      'padding_nan_and_append_invariance': 'exact', 'gradients': 'finite', 'initial_state_sha256': hash_a}
            if name == 'Neural-CDE':
                with torch.no_grad():
                    baseline_steps = model.rk4_substeps
                    model.rk4_substeps = baseline_steps * 2; fine = model(x, mask, times)['logits']
                    record[f'rk4_{baseline_steps}_vs_{2 * baseline_steps}_substeps_max_logit_difference'] = float((fine - clean['logits']).abs().max())
            records.append(record)
    z = torch.zeros(2, 3)
    dx = torch.tensor([[1., 2.], [3., 4.]])
    def field(state): return state.new_ones(state.shape[0], 3, 2)
    torch.testing.assert_close(models.cde_rk4_step(z, dx, field), dx.sum(-1, keepdim=True).expand(-1, 3))
    return records


def main():
    f = utility_functions()
    result = {'status': 'passed', 'execution': 'CPU synthetic invariants only; no training or source test-score read',
              'torch': torch.__version__, 'numpy': np.__version__,
              'source_sha256': {p: digest(ROOT / p) for p in ['models.py', 'train_study.py', 'protocol.json']},
              'utilities': test_utilities(f), 'models': test_models(f)}
    path = ROOT / 'audit' / 'independent_smoke_results.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
