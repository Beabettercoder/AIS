"""Train genuine masked Transformer four-arm development study; no test reads."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from pathlib import Path
import argparse, hashlib, json, math, sys
import numpy as np
import torch
from torch import nn

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'causal_gate_main_v1_20261027'))
import train_controls as source
hm, th = source.hm, source.th
ARMS = ['raw', 'causal', 'raw_bundle', 'causal_bundle']
hm.ARMS.clear()
hm.ARMS.update({a: dict(key=a, nf=12, graph=False, recipe=a.endswith('_bundle'), masked=False) for a in ARMS})
th.training_loss = source.controlled_loss

class MaskedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(12, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        pos = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, 64, 2).float() * (-math.log(10000.) / 64))
        pe = torch.zeros(20, 64)
        pe[:, 0::2] = torch.sin(pos * scale)
        pe[:, 1::2] = torch.cos(pos * scale)
        self.register_buffer('position_encoding', pe)

    def seq(self, x, mask):
        x = x * mask.unsqueeze(-1)
        h = self.encoder(self.project(x) + self.position_encoding[:x.shape[1]], src_key_padding_mask=~mask)
        return (h * mask.unsqueeze(-1)).transpose(1, 2)

class Core(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.enc = MaskedTransformer()
        self.bin = nn.Linear(128, 1)
        self.cls = nn.Linear(128, 5)
        if recipe:
            self.ctrl = nn.Linear(4, 2)

class TransformerModel(nn.Module):
    def __init__(self, kind, arm):
        super().__init__()
        if kind != 'transformer':
            raise ValueError('Unexpected backbone')
        self.kind, self.arm, self.config = kind, arm, hm.ARMS[arm].copy()
        self.core = Core(self.config['recipe'])

    @property
    def bin(self): return self.core.bin
    @property
    def cls(self): return self.core.cls
    @property
    def ctrl(self): return self.core.ctrl
    def sequence(self, x, mask): return self.core.enc.seq(x, mask)
    def forward(self, x, mask, nseq=None, nxy=None, nmask=None, nw=None, rf=None):
        h = self.sequence(x, mask)
        valid = mask.unsqueeze(1)
        mean = (h * valid).sum(-1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        maximum = h.masked_fill(~valid, -float('inf')).max(-1).values
        pooled = torch.cat((mean, maximum), -1)
        return self.bin(pooled).squeeze(-1), self.cls(pooled), pooled, torch.zeros_like(pooled)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def setup():
    th.HistoricalModel = TransformerModel
    data, proof = source.build_data()
    for split in ('train', 'val'):
        data[split]['features']['raw_bundle'] = data[split]['features']['raw'].copy()
        data[split]['features']['causal_bundle'] = data[split]['features']['causal'].copy()
        for plain in ('raw', 'causal'):
            assert np.array_equal(data[split]['features'][plain], data[split]['features'][plain + '_bundle'])
        mask = data[split]['mask']
        assert mask.dtype == bool and np.all(mask.sum(axis=1) > 0)
        assert np.array_equal(mask, np.arange(mask.shape[1])[None, :] < mask.sum(axis=1)[:, None])
    return data, proof

def preflight(device, data, stats):
    # Verify actual attention and pool ignore any arbitrary values in padded steps.
    for seed in (0, 1, 2):
        enc_hashes = []
        for arm in ARMS:
            th.seed_all(seed)
            model = TransformerModel('transformer', arm).to(device)
            enc_hashes.append(th.state_sha(model.core.enc.state_dict()))
            x = th.normalized(data['train']['features'][arm], data['train']['mask'], stats[arm], False)
            tensors = th.to_tensors(data['train'], x)
            b = th.batch(tensors, slice(0, 8), device)
            model.eval()
            with torch.no_grad():
                a = model(b['x'], b['mask'])[0]
                x2 = b['x'].clone()
                x2[~b['mask']] = 1000.
                c = model(x2, b['mask'])[0]
                assert torch.allclose(a, c, rtol=0, atol=1e-6), float((a-c).abs().max())
            model.train()
            loss, logs = source.controlled_loss(model, b, torch.ones(5, device=device))
            assert torch.isfinite(loss)
            loss.backward()
            assert model.core.enc.project.weight.grad is not None
            assert torch.isfinite(model.core.enc.project.weight.grad).all()
        assert len(set(enc_hashes)) == 1, (seed, enc_hashes)
    return {'status': 'passed', 'arms': ARMS, 'seeds': [0, 1, 2], 'tests': ['masked padded-value invariance', 'finite forward/backward each arm and seed', 'same initial encoder state within seed', 'identical bundle and nonbundle inputs']}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    protocol = json.loads((OUT / 'protocol.json').read_text(encoding='utf-8'))
    assert protocol['arms'] == ARMS
    data, proof = setup()
    stats = {a: th.fit_normalizer(data['train']['features'][a], data['train']['mask']) for a in ARMS}
    for a in ('raw', 'causal'):
        assert np.array_equal(stats[a]['mean'], stats[a+'_bundle']['mean'])
        assert np.array_equal(stats[a]['std'], stats[a+'_bundle']['std'])
    report = preflight(device, data, stats)
    report['encoder_parameters'] = sum(p.numel() for p in TransformerModel('transformer', 'raw').parameters())
    report['device'] = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    preflight_path = OUT/'preflight.json'
    if preflight_path.exists():
        if json.loads(preflight_path.read_text(encoding='utf-8')) != report:
            raise ValueError('Preflight changed between runs')
    else:
        th.jwrite(preflight_path, report)
    if args.preflight_only:
        print('PREFLIGHT_ONLY_COMPLETE', flush=True)
        return
    worker = OUT/'runs'/'transformer'
    worker.mkdir(parents=True, exist_ok=False)
    config = {'protocol_sha256': sha(OUT/'protocol.json'), 'script_sha256': sha(__file__), 'sources': proof,
              'source_train_controls_sha256': sha(ROOT/'causal_gate_main_v1_20261027/train_controls.py'),
              'source_historical_model_sha256': sha(ROOT/'gating_training_v2_20261027/historical/historical_model.py'),
              'source_training_sha256': sha(ROOT/'gating_training_v2_20261027/historical/train_historical.py'),
              'device': report['device'], 'torch': torch.__version__, 'numpy': np.__version__}
    th.jwrite(worker/'configuration.json', config)
    training = protocol['training']
    for seed in protocol['seeds']:
        for arm in ARMS:
            th.train_one('transformer', arm, seed, data, stats[arm], training, device, worker/f'seed_{seed}'/arm,
                         sha(OUT/'protocol.json'), sha(worker/'configuration.json'))
    print('ALL_TWELVE_COMPLETE', flush=True)

if __name__ == '__main__': main()
