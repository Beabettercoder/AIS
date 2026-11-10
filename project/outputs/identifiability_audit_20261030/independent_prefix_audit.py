"""Independently perturb unseen future neighbor reports and verify prefix invariance."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
BASE=Path('/mnt/sda/WangZijia/AIS/outputs'); R=BASE/'balanced_global_v2/relation_dual_pressure_v2_20261027';C=BASE/'delayed_observation_20261030';O=BASE/'identifiability_audit_20261030'
sys.path.insert(0,str(C));import builder
m=json.loads((R/'dataset_manifest_combined.json').read_text());rows=sorted([r for r in m['samples'] if r['version']=='relaxed' and r['split']=='val'],key=lambda r:(r['region'],r['window_id']))[:16]
keys=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y')
nchecks=nfuture=0;maxdiff=0;invalid_order=0;nb_cases=0
for row in rows:
    data={}
    for view in ('base','natural','suffix'):
        with np.load(R/row['files'][view]['path'],allow_pickle=False) as z:data[view]={k:z[k].copy() for k in keys}
    comb={k:v.copy() for k,v in data['base'].items()};comb['neighbor_sog_kn']=data['suffix']['neighbor_sog_kn'].copy()
    for member in row['stress']['members']:
        lo=int(comb['member_offsets'][member['member_index']]);ix=lo+int(member['indices']['suffix']['report_index']);digest=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+member['mmsi']).encode()).digest();theta=2*np.pi*int.from_bytes(digest[:8],'big')/2**64
        comb['neighbor_pos'][:,ix:ix+2]+=600*np.array([np.cos(theta),np.sin(theta)])
    data['combined']=comb
    for case in (0,2):
        tt=data['base']['target_t'][case];invalid_order+=int(np.any(np.diff(tt)<=0))
        for lo,hi in zip(data['base']['member_offsets'][:-1],data['base']['member_offsets'][1:]):
            invalid_order+=int(np.any(np.diff(data['base']['neighbor_t'][case,int(lo):int(hi)])<=0));nb_cases+=1
        for view,o in data.items():
            full,_=builder.calc(o,case,row['mmsi']);kset=sorted(set((len(tt)//4,len(tt)//2,len(tt)-1)))
            for k in kset:
                modified={key:value.copy() for key,value in o.items()}; future=np.flatnonzero(o['neighbor_t'][case]>tt[k]);nfuture+=len(future)
                modified['neighbor_sog_kn'][case,future]+=123.;modified['neighbor_cog_deg'][case,future]=(modified['neighbor_cog_deg'][case,future]+137.)%360;modified['neighbor_pos'][case,future]+=np.array([123456.,-123456.])
                changed,_=builder.calc(modified,case,row['mmsi']);diff=float(np.max(np.abs(full[:,:k+1]-changed[:,:k+1])));maxdiff=max(maxdiff,diff);assert diff<=1e-6,(row['window_id'],case,view,k,diff)
                nchecks+=1
result={'scope':'16 held-out development validation sources, cases0/2, four views, three target cutoffs; perturbed future values only, no training or test labels','checks':nchecks,'future_report_perturbations':nfuture,'neighbor_case_sequences_checked':nb_cases,'unsorted_sequences':invalid_order,'max_absolute_prefix_diff':maxdiff,'asserted_tolerance':1e-6,'warning':'phase midpoint and fixed offline neighbor cohort are themselves precomputed using full window; this audit establishes absence of future neighbor-value influence on fitted fields, not an online end-to-end pipeline'}
assert invalid_order==0
(O/'independent_prefix_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
