#!/usr/bin/env python3
"""Replay the real 7D formal TCN at frozen weights on recorded validation set."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single')
sys.path.insert(0,str(ROOT))
from train_single import load,fit,convert,Model

def main():
    torch.set_num_threads(2);dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train=load('train');val=load('val');norm=fit(train);data=convert(val,norm,dev);out=[]
    for seed in range(3):
        for arm in ('raw','causal_bundle','f0_bundle','f1_bundle'):
            folder=ROOT/'runs/tcn'/f'seed{seed}'/arm
            net=Model('tcn').to(dev);net.load_state_dict(torch.load(folder/'final.pt',map_location=dev,weights_only=True));net.eval()
            key='raw' if arm=='raw' else 'self';fk='f0' if arm=='f0_bundle' else 'f1';hasfield=arm.startswith('f')
            p=[]
            with torch.no_grad():
                for i in range(0,len(val['y']),256):
                    sl=slice(i,i+256);z=net(data[key][sl],data['mask'][sl],data['times'][sl],data[fk][sl],data['rho'][sl],hasfield)[0];p.extend(torch.sigmoid(z).cpu().numpy())
            p=np.asarray(p)
            with np.load(folder/'final_val.npz',allow_pickle=False) as z:
                assert np.array_equal(z['window_id'],val['ids']);stored=z['score'];diff=np.abs(p-stored)
            out.append({'seed':seed,'arm':arm,'n':len(p),'max_abs_difference':float(diff.max()),'mean_abs_difference':float(diff.mean()),'within_1e-6':bool((diff<1e-6).all())})
    dest=Path('/home/ps/AIS_bridge_20261107/tcn');dest.mkdir(parents=True,exist_ok=True);(dest/'legacy_exact_validation_replay.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
