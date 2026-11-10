"""Conditional entry point for the single-vessel eight-backbone expansion."""
import argparse
import json
from pathlib import Path
import scale_train

HERE = Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')

def gate():
    path = HERE / 'AUTHORIZED_STAGE2.json'
    if not path.exists() or json.loads(path.read_text(encoding='utf8')).get('relation_pilot_criteria_met') is not True:
        raise SystemExit('Stage 2 not authorized: no parent-issued qualifying pilot gate record.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True, choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'])
    p.add_argument('--seed', required=True, type=int, choices=[0,1,2])
    p.add_argument('--gpu', required=True, type=int)
    a = p.parse_args()
    if a.gpu == 4: raise SystemExit('GPU 4 is reserved by an existing service.')
    gate()
    scale_train.main(a.name, a.gpu, a.seed)
