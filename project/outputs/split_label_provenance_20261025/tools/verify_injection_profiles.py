"""Read-only inversion of the five stored clean injection profiles.

Run via stdin on 6000D; does not rebuild or modify any research assets.
Only label_cell==2 (behavior anomaly without observation corruption) is fitted.
Profiles need at least three post-onset observations and envelope range>=0.05.
The remainder are explicitly unidentified, never silently treated as matches.
"""
import json
import numpy as np
from pathlib import Path

BASE = Path('/mnt/sda/WangZijia/AIS/outputs/phase1')
PARAMS = {1: [120,135,150,165,180], 2: [20,40,60,90,120],
          3: [2,1.5,1,.7,.5], 4: [5,10,15,18,20], 5: [.6,.5,.4,.3,.2]}
NAMES = {1:'turn_back_delta_deg',2:'deviation_delta_deg',3:'stall_target_kn',
         4:'sinuous_amplitude_deg',5:'speed_drop_factor'}
results=[]
for path in [BASE/'eval_v2_graph2']+sorted(BASE.glob('*_dataset')):
    for split in ['train','val','test']:
        z=np.load(path/(split+'.npz'))
        raw,t_all,mask=z['raw'],z['t'],z['mask']
        row={'dataset':path.name,'split':split,'fits':{}}
        for c in range(1,6):
            matches={}; valid=0; skipped=0; bad=0; inferred=[]
            for i in np.where((z['label_class']==c)&(z['label_cell']==2))[0]:
                n=int(mask[i].sum()); t=t_all[i,:n].astype(float)
                t0=t[-1]-z['label_delay'][i]; i0=np.searchsorted(t,t0)
                t=t[i0:]; r=raw[i,i0:n].astype(float)
                u=np.clip((t-t0)/{1:600,2:180,3:240,4:1800,5:120}[c],0,1)
                env=u*u*(3-2*u)
                if c==4:
                    env=np.sin(np.pi*u)**2*np.sin(2*np.pi*(t-t0)/600)
                if len(t)<3 or np.ptp(env)<.05:
                    skipped+=1; continue
                y=np.unwrap(np.arctan2(r[:,5],r[:,6])) if c in [1,2,4] else r[:,4]
                X=np.c_[np.ones(len(t)),env]
                coef=np.linalg.lstsq(X,y,rcond=None)[0]
                err=float(np.max(np.abs(X@coef-y)))
                value=float(np.degrees(coef[1])) if c in [1,2,4] else (
                    float(coef.sum()) if c==3 else float(1+coef[1]/coef[0]))
                level=int(np.argmin(np.abs(np.array(PARAMS[c])-value)))+1
                if err<1e-4 and abs(PARAMS[c][level-1]-value)<.01:
                    matches[str(level)]=matches.get(str(level),0)+1; valid+=1
                else:
                    bad+=1
                inferred.append(value)
            row['fits'][NAMES[c]]={'levels':matches,'fit_count':valid,
                'unresolved':bad,'insufficient_points_or_envelope_variation':skipped,
                'median_parameter':round(float(np.median(inferred)),6) if inferred else None}
        results.append(row)
print(json.dumps({'method':'least-squares profile inversion from stored post-onset SOG/COG',
                  'results':results},ensure_ascii=False,indent=2))
