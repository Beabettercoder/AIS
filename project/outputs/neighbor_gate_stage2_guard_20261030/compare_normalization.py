"""Compare train-fitted baseline versus candidate normalization, without test access."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
HERE=ROOT/'neighbor_gate_stage2_guard_20261030'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module(ROOT/'heading_alignment_20261030/train_heading.py','old_model')
new=module(ROOT/'neighbor_gate_safe_20261030/train.py','new_model')
torch.set_num_threads(2)
o=old.read('train'); n=new.read('train')
om,os=old.fit(o);nm,ns=new.fit(n)
device=torch.device('cpu')
comparisons={}
for split, od, nd in (('train',o,n),('val',old.read('val'),new.read('val'))):
    for view in ('base','natural','suffix','combined_l100_m600'):
        ox,_,_=old.prep(od,view,'F0',om,os,device)
        nx,_,_=new.prep(nd,view,'F0',nm,ns,device)
        delta=(ox-nx).abs().numpy()
        target_valid=od['mask']
        _,context_valid=old.features(od,view,'F0')
        a=delta[target_valid];c=delta[context_valid][:,8:]
        comparisons[split+'/'+view]={
            'valid_target_steps':int(target_valid.sum()),
            'context_valid_steps':int(context_valid.sum()),
            'max_abs_difference_all_valid_channels':float(a.max()),
            'fraction_valid_values_over_1e_6':float((a>1e-6).mean()),
            'max_abs_difference_valid_context_channels':float(c.max()),
            'fraction_valid_context_values_over_1e_6':float((c>1e-6).mean()),
            'max_abs_difference_per_channel':a.max(0).tolist()}
out={'target_mean_equal':bool(np.array_equal(om[:8],nm[:8])),
     'target_std_equal':bool(np.array_equal(os[:8],ns[:8])),
     'changed_mean_channels':[int(i) for i in np.flatnonzero(np.abs(nm-om)>1e-12)],
     'changed_std_channels':[int(i) for i in np.flatnonzero(np.abs(ns-os)>1e-12)],
     'comparisons':comparisons,
     'note':'F0 physical fields equal but pooled F0/F1 training normalization can change; this is not a pure F1-only intervention.'}
HERE.joinpath('compare_normalization.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
