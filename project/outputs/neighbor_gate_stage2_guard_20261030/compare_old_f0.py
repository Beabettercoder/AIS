"""Train/validation-only comparison of baseline and candidate common support."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
OLD=ROOT/'relation_repair_20261030'
NEW=ROOT/'neighbor_gate_safe_20261030'
HERE=ROOT/'neighbor_gate_stage2_guard_20261030'
out={}
for split in ('train','val'):
    with np.load(OLD/f'{split}_relation_field.npz',allow_pickle=False) as z:
        old={k:z[k] for k in z.files}
    with np.load(NEW/f'{split}_relation_field.npz',allow_pickle=False) as z:
        new={k:z[k] for k in z.files}
    for k in ('source','case','y','length','mmsi','region','raw7','causal7','time_s'):
        np.testing.assert_array_equal(old[k],new[k],err_msg=f'{split} {k}')
    mask=np.arange(20)[None,:]<old['length'][:,None]
    details={}
    for view in ('base','natural','suffix','combined_l100_m600'):
        np.testing.assert_array_equal(old[view+'__count'],new[view+'__count'])
        o=old[view+'__field']; n=new[view+'__field']
        oldmode=o[:,0,:,7]; newmode=n[:,0,:,7]
        delta=np.max(np.abs(o[:,0,:,:6]-n[:,0,:,:6]),axis=-1)
        changed=(delta>1e-6)&mask
        details[view]={'old_new_f0_mode_disagreement_steps':int(((oldmode!=newmode)&mask).sum()),
                       'old_new_f0_values_differ_over_1e_6_steps':int(changed.sum()),
                       'max_f0_value_abs_difference':float(delta[mask].max()),
                       'old_f0_valid_steps':int(((oldmode>0)&mask).sum()),
                       'new_f0_valid_steps':int(((newmode>0)&mask).sum()),
                       'valid_target_steps':int(mask.sum())}
    out[split]=details
HERE.joinpath('compare_old_f0.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out,indent=2))
