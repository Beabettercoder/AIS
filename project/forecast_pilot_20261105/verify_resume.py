"""Small CPU-only exact-epoch resume check; never starts the 48-run pilot.

Uses existing smoke data. Compare a two-epoch TCN/P0 training with the same
two-epoch configuration interrupted immediately after epoch-one checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import os

import torch
import train as runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True,
                   help='Existing smoke data with train.npz, val.npz and data_manifest.json')
    p.add_argument('--out', type=Path, required=True, help='New, empty verification output directory')
    p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1: p.error('batch-size must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Use a new empty --out directory: {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'test': 'complete epoch checkpoint resume', 'status': 'running',
              'model': 'tcn', 'arm': 'P0', 'seed': 0, 'epochs': 2,
              'device': 'cpu', 'interruption': 'after saving complete epoch 1',
              'data': str(args.data.resolve()), 'formal_training_started': False}
    report_path = args.out/'resume_verification.json'
    runner.atomic_json(report_path, report)
    original_save = runner.save_checkpoint
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        source = Path(__file__).resolve().parent
        manifest, hashes, manifest_hash = runner.verify_data_manifest(args.data, source)
        train = runner.load_data(args.data/'train.npz', training=True)
        val = runner.load_data(args.data/'val.npz', training=False)
        normalizer = runner.fit_normalizer(train)
        identity = {'data_sha256': hashes, 'data_manifest_sha256': manifest_hash,
                    'dataset_fingerprint': manifest.get('fingerprint'),
                    'source_sha256': {n: runner.digest(source/n) for n in
                                     ('train.py','forecast_models.py','vendor_encoders.py',
                                      'features.py','build_data.py')},
                    'torch_version': torch.__version__}
        device = torch.device('cpu')
        common = dict(epochs=2, batch_size=args.batch_size, lr=.002, threads=1)
        uninterrupted = SimpleNamespace(**common, out=args.out/'uninterrupted', resume=False)
        interrupted = SimpleNamespace(**common, out=args.out/'interrupted_then_resumed', resume=False)
        print('Resume check 1/3: uninterrupted two epochs.', flush=True)
        runner.one_run(uninterrupted,train,val,normalizer,identity,'tcn',0,'P0',device)
        fired = False

        def interrupt_after_complete_epoch_one(path, payload):
            nonlocal fired
            original_save(path, payload)
            if not fired and payload['completed_epoch'] == 1:
                fired = True
                raise KeyboardInterrupt('Controlled verification interruption after complete epoch 1')

        print('Resume check 2/3: write epoch-one checkpoint, then controlled interruption.', flush=True)
        runner.save_checkpoint = interrupt_after_complete_epoch_one
        try:
            runner.one_run(interrupted,train,val,normalizer,identity,'tcn',0,'P0',device)
        except KeyboardInterrupt:
            if not fired: raise
        finally:
            runner.save_checkpoint = original_save
        if not fired: raise AssertionError('The interruption hook did not execute')
        folder = interrupted.out/'tcn'/'seed_0'/'P0'
        first = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        if first['completed_epoch'] != 1:
            raise AssertionError('Interrupted checkpoint is not a complete epoch-one checkpoint')
        if (folder/'result.json').exists():
            raise AssertionError('Interrupted run was incorrectly marked complete')
        interrupted.resume = True
        print('Resume check 3/3: same configuration resumes to epoch two.', flush=True)
        runner.one_run(interrupted,train,val,normalizer,identity,'tcn',0,'P0',device)
        left = torch.load(uninterrupted.out/'tcn'/'seed_0'/'P0'/'checkpoint.pt',
                          map_location='cpu',weights_only=False)
        right = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        if set(left['model']) != set(right['model']):
            raise AssertionError('Model state keys differ')
        mismatched = [k for k in left['model'] if not torch.equal(left['model'][k],right['model'][k])]
        history_equal = left['history'] == right['history']
        hashes_equal = left['initial_model_hash'] == right['initial_model_hash']
        config_equal = left['config_hash'] == right['config_hash']
        report.update(status='passed' if not mismatched and history_equal and hashes_equal and config_equal else 'failed',
                      model_tensor_count=len(left['model']), mismatched_model_tensors=mismatched,
                      all_model_tensors_bit_exact=not mismatched,
                      train_loss_and_val_ADE_FDE_history_exact=history_equal,
                      initial_model_hash_equal=hashes_equal,config_hash_equal=config_equal,
                      uninterrupted_history=left['history'],resumed_history=right['history'],
                      completed_epoch_uninterrupted=left['completed_epoch'],
                      completed_epoch_resumed=right['completed_epoch'],
                      data_manifest_sha256=manifest_hash,
                      verification_script_sha256=runner.digest(__file__))
        runner.atomic_json(report_path,report)
        if report['status'] != 'passed': raise AssertionError('Resume verification failed; inspect report JSON')
        print(f'PASS: every model tensor and loss/ADE/FDE history value matches exactly. {report_path.resolve()}',flush=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        runner.atomic_json(report_path,report)
        raise
    finally:
        runner.save_checkpoint = original_save


if __name__ == '__main__':
    main()
