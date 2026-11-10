"""Independently replay member records and saved background outputs."""
from pathlib import Path
import argparse, gzip, json, sys
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=ROOT/'robust_background_v1_20261027'
sys.path.insert(0,str(ROOT/'context_adapter_v1_20261026/neighborhood'))
from audit_neighborhood import Source
KNOT=1852/3600

def median_weighted_bruteforce(values,weights):
    if np.sum(weights)<=1e-12:return np.median(values,axis=0)
    result=[]
    for j in range(2):
        candidates=sorted(set(values[:,j]))
        result.append(next(float(q) for q in candidates if np.sum(weights[values[:,j]<=q])>=sum(weights)/2))
    return np.asarray(result)

def run(region):
    meta=json.loads((ROOT/'release_20261026/source_registry.json').read_text(encoding='utf8'))['regions'][region]
    source=Source(meta)
    overall={'region':region,'status':'running','splits':{}}
    for split in ('train','val'):
        orig=ROOT/'labeling/labeled_v1/features_scene'/region/f'{split}.npz'
        aud=orig.with_name(f'{split}_context_audit.jsonl.gz')
        new=OUT/region/f'{split}.npz'
        with np.load(orig,allow_pickle=False) as z:
            x={k:z[k].copy() for k in ['window_id','mmsi','mask','time_s','context_valid','target_sog_kn','target_cog_deg','residual_mps']}
        with np.load(new,allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
        assert np.array_equal(x['window_id'],y['window_id']) and np.array_equal(x['context_valid'],y['common_valid'])
        errors={n:0. for n in ['observation_velocity','weight','mean','median','weighted_median','trusted_mean','trusted_median','trusted_weighted_median','affine']}
        steps=0;members=0
        def compare(name,actual,expected):
            err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
            errors[name]=max(errors[name],err)
            assert err<1e-4,(region,split,name,err)
        with gzip.open(aud,'rt',encoding='utf8') as src,gzip.open(OUT/region/f'{split}_member_provenance.jsonl.gz','rt',encoding='utf8') as prov:
            it=iter(prov)
            for i,line in enumerate(src):
                md=json.loads(line)
                assert md['window_id']==str(x['window_id'][i])
                if not y['common_valid'][i].any():continue
                overlay=source.overlay(md)
                for st in md['steps']:
                    t=int(st['step'])
                    if not y['common_valid'][i,t]:continue
                    q=json.loads(next(it));assert q['window_id']==str(x['window_id'][i]) and q['step']==t
                    wanted=[e for e in st['neighbors'] if e['plain_accepted']]
                    assert len(q['members'])==len(wanted)==int(y['plain_count'][i,t])
                    assert sum(e['accepted'] for e in wanted)==int(y['trusted_count'][i,t])
                    velocity=[];quality=[];trusted=[]
                    for record,entry in zip(q['members'],wanted):
                        assert record['mmsi']==str(entry['mmsi']) and record['time_s']==entry['time_s']
                        assert record['time_s']<=x['time_s'][i,t] and x['time_s'][i,t]-record['time_s']<=300
                        assert record['mmsi']!=str(x['mmsi'][i])
                        pos,s,c,k,e,w=source.observation(record['mmsi'],record['time_s'],overlay)
                        true=np.array([s*KNOT*np.sin(np.deg2rad(c)),s*KNOT*np.cos(np.deg2rad(c))])
                        compare('observation_velocity',record['velocity_mps'],true)
                        compare('weight',record['w_abs'],w if k else 0.)
                        assert record['kinematic_valid']==bool(k) and record['historical_affine_accepted']==entry['accepted']
                        velocity.append(true);quality.append(w if k else 0.);trusted.append(entry['accepted']);members+=1
                    v=np.asarray(velocity);w=np.asarray(quality);tr=np.asarray(trusted,bool)
                    a=np.deg2rad(x['target_cog_deg'][i,t]);speed=x['target_sog_kn'][i,t]*KNOT
                    target=np.array([speed*np.sin(a),speed*np.cos(a)])
                    outcomes={'mean':np.mean(v,axis=0),'median':np.median(v,axis=0),
                       'weighted_median':median_weighted_bruteforce(v,w),
                       'trusted_mean':np.mean(v[tr],axis=0),'trusted_median':np.median(v[tr],axis=0),
                       'trusted_weighted_median':median_weighted_bruteforce(v[tr],w[tr])}
                    for name,bg in outcomes.items():
                        compare(name,y[f'background_{name}_mps'][i,t],bg)
                        compare(name,y[f'residual_{name}_mps'][i,t],target-bg)
                        compare(name,y[f'residual_{name}_norm_mps'][i,t],np.linalg.norm(target-bg))
                    compare('affine',y['residual_affine_mps'][i,t],x['residual_mps'][i,t])
                    steps+=1
            assert next(it,None) is None
        assert steps==int(y['common_valid'].sum())
        for key in y:
            if key.startswith(('background_','residual_')):
                assert np.all(y[key][~y['common_valid']]==0)
        overall['splits'][split]={'common_steps':steps,'recovered_members':members,'max_abs_error':errors}
        print('AUDIT',region,split,steps,members,flush=True)
    overall['status']='passed'
    (OUT/region/'independent_audit.json').write_text(json.dumps(overall,indent=2,ensure_ascii=False),encoding='utf8')
    print('REGION_AUDITED',region,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--regions',nargs='+',required=True);args=ap.parse_args()
    for region in args.regions:run(region)
