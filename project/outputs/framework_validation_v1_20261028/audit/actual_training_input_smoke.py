"""Untrained model numerical acceptance on training inputs only; no fitting/scoring."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from independent_smoke import utility_functions, digest
sys.path.insert(0, str(ROOT))
from models import make_model, MODEL_NAMES


def main():
    torch.set_num_threads(2)
    f = utility_functions(); records = []
    for task in ['individual', 'relation']:
        with np.load(ROOT / 'data' / task / 'train.npz', allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        norm = f['fit_normalizer'](data, task)
        ids = np.linspace(0, len(data['y']) - 1, 8, dtype=int)
        mask = torch.tensor(data['mask'][ids])
        times = torch.tensor(f['times_array'](data)[ids])
        for arm in ['C0', 'C3']:
            normalized, fm = f['normalize'](data, 'base__' + arm, norm)
            x = torch.tensor(normalized[ids]); fm = torch.tensor(fm[ids])
            for name in MODEL_NAMES:
                torch.manual_seed(10000)
                model = make_model(name, x.shape[-1], hidden=64).eval()
                out = model(x, mask, times)['logits']
                assert torch.isfinite(out).all()
                model.zero_grad(set_to_none=True)
                out.square().mean().backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                row = {'task': task, 'arm': arm, 'model': name, 'train_samples': len(ids), 'finite_forward_backward': True}
                if name == 'Neural-CDE':
                    with torch.no_grad():
                        baseline_steps = model.rk4_substeps
                        model.rk4_substeps = baseline_steps * 2; refined = model(x, mask, times)['logits']
                        model.rk4_substeps = baseline_steps * 4; finer = model(x, mask, times)['logits']
                        row[f'rk4_{baseline_steps}_vs_{2 * baseline_steps}_max_logit_delta'] = float((refined - out).abs().max())
                        row[f'rk4_{2 * baseline_steps}_vs_{4 * baseline_steps}_max_logit_delta'] = float((finer - refined).abs().max())
                        model.rk4_substeps = baseline_steps
                if arm == 'C3':
                    augmented = f['perturb'](x, fm, torch.Generator().manual_seed(445))
                    with torch.no_grad():
                        aug_out = model(augmented, mask, times)['logits']
                        assert torch.isfinite(aug_out).all()
                    row['augmented_finite'] = True
                records.append(row)
    result = {'status': 'passed', 'scope': 'Untrained encoders on eight predetermined train indices per task; no optimizer or performance metric',
              'source_sha256': {p: digest(ROOT / p) for p in ['models.py', 'train_study.py', 'protocol.json']}, 'records': records}
    (HERE / 'actual_training_input_smoke_results.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({'status': result['status'], 'combinations': len(records), 'cde': [r for r in records if r['model'] == 'Neural-CDE']}, indent=2))


if __name__ == '__main__': main()
