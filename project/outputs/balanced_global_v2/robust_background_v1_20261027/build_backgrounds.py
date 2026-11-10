"""Reconstruct training/development contextual velocity estimators from AIS observations.

Only train/val inputs are enabled here. The saved historical affine mask defines
common support; no future target time or hidden uncorrupted feature is used.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, sys, time
import numpy as np

ROOT = Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
HELPER = ROOT/'context_adapter_v1_20261026/neighborhood'
sys.path.insert(0,str(HELPER))
from audit_neighborhood import Source, sha

REGIONS = ['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
KNOT = 1852/3600

def weighted_median(v,w):
    result=[]
    if float(np.sum(w))<=1e-12:
        return np.median(v,axis=0),True
    for j in range(2):
        idx=np.argsort(v[:,j],kind='stable')
        pos=np.searchsorted(np.cumsum(w[idx]),sum(w)/2,side='left')
        result.append(v[idx[min(int(pos),len(idx)-1)],j])
    return np.asarray(result),False

def build_region(region,out):
    started=time.time()
    meta=json.loads((ROOT/'release_20261026/source_registry.json').read_text(encoding='utf8'))['regions'][region]
    source=Source(meta)
    d=out/region
    d.mkdir(parents=True,exist_ok=False)
    report={'region':region,'source_path':meta['canonical_file'],'source_sha256':meta['sha256'],
            'builder_sha256':sha(Path(__file__)),'splits':{}}
    for split in ('train','val'):
        path=ROOT/'labeling/labeled_v1/features_scene'/region/f'{split}.npz'
        audit=path.with_name(f'{split}_context_audit.jsonl.gz')
        with np.load(path,allow_pickle=False) as z:
            allowed=['window_id','mmsi','time_s','mask','context_valid','target_sog_kn',
                     'target_cog_deg','target_pos','background_field_mps','residual_mps']
            data={k:z[k].copy() for k in allowed}
        n,T=data['mask'].shape
        valid=data['context_valid'].astype(bool)
        assert not np.any(valid & ~data['mask'])
        names=['mean','median','weighted_median','trusted_mean','trusted_median','trusted_weighted_median']
        back={name:np.zeros((n,T,2),np.float32) for name in names}
        target=np.zeros((n,T,2),np.float64)
        angle=np.deg2rad(data['target_cog_deg'])
        target[:,:,0]=data['target_sog_kn']*KNOT*np.sin(angle)
        target[:,:,1]=data['target_sog_kn']*KNOT*np.cos(angle)
        plain_count=np.zeros((n,T),np.int32)
        trusted_count=np.zeros((n,T),np.int32)
        counts={'common_steps':0,'plain_members':0,'trusted_members':0,'unknown_reliability':0,
                'zero_weight_fallback_steps':0,'trusted_zero_weight_fallback_steps':0,'neighbor_reports_read':0}
        path_out=d/f'{split}_member_provenance.jsonl.gz'
        with gzip.open(audit,'rt',encoding='utf8') as inp, gzip.open(path_out,'xt',encoding='utf8') as proof:
            for i,line in enumerate(inp):
                md=json.loads(line)
                assert md['split']==split and str(md['window_id'])==str(data['window_id'][i])
                if not np.any(valid[i]):continue
                overlay=source.overlay(md)
                for st in md['steps']:
                    j=int(st['step'])
                    if not valid[i,j]:continue
                    assert st['boundary_context_valid']
                    now=float(data['time_s'][i,j]);target_id=str(data['mmsi'][i]);v=[];w=[];tr=[];ids=[];times=[];kin=[]
                    for ent in st.get('neighbors',[]):
                        if not ent['plain_accepted']:continue
                        mid=str(ent['mmsi']);stamp=float(ent['time_s'])
                        assert mid!=target_id and 0<=now-stamp<=300 and mid not in ids
                        pos,s,c,kv,e,ww=source.observation(mid,stamp,overlay)
                        assert np.isfinite(pos).all() and np.isfinite(s) and np.isfinite(c) and .5<=s<102.2 and 0<=c<360
                        assert bool(kv)==bool(ent['kinematic_valid']) and abs(float(ww)-float(ent['reliability']))<1e-8
                        assert abs(float(np.linalg.norm(pos-data['target_pos'][i,j]))-float(ent['distance_m']))<1e-6
                        a=np.deg2rad(c)
                        v.append([float(s*KNOT*np.sin(a)),float(s*KNOT*np.cos(a))]);w.append(float(ww) if kv else 0.)
                        tr.append(bool(ent['accepted']));ids.append(mid);times.append(stamp);kin.append(bool(kv))
                    v=np.asarray(v,np.float64);w=np.asarray(w,np.float64);tr=np.asarray(tr,bool)
                    assert len(v)>=4 and tr.sum()>=4 and len(set(ids))==len(ids)
                    plain_count[i,j]=len(v);trusted_count[i,j]=int(tr.sum())
                    back['mean'][i,j]=v.mean(0);back['median'][i,j]=np.median(v,axis=0)
                    back['weighted_median'][i,j],fallback=weighted_median(v,w)
                    back['trusted_mean'][i,j]=v[tr].mean(0);back['trusted_median'][i,j]=np.median(v[tr],axis=0)
                    back['trusted_weighted_median'][i,j],fallback_tr=weighted_median(v[tr],w[tr])
                    counts['common_steps']+=1;counts['plain_members']+=len(v);counts['trusted_members']+=int(tr.sum())
                    counts['unknown_reliability']+=int((~np.asarray(kin)).sum());counts['neighbor_reports_read']+=len(v)
                    counts['zero_weight_fallback_steps']+=int(fallback);counts['trusted_zero_weight_fallback_steps']+=int(fallback_tr)
                    proof.write(json.dumps({'window_id':str(data['window_id'][i]),'step':j,'target_mmsi':target_id,
                      'time_s':now,'members':[{'mmsi':mid,'time_s':times[k],'velocity_mps':v[k].tolist(),
                      'w_abs':float(w[k]),'kinematic_valid':kin[k], 'historical_affine_accepted':bool(tr[k])}
                      for k,mid in enumerate(ids)]},ensure_ascii=False,separators=(',',':'))+'\n')
                if (i+1)%200==0:print('REGION',region,split,i+1,flush=True)
        assert i+1==n and counts['common_steps']==int(valid.sum())
        result={'window_id':data['window_id'],'common_valid':valid,'plain_count':plain_count,'trusted_count':trusted_count}
        for name in names:
            residual=np.where(valid[...,None],target-back[name],0).astype(np.float32)
            result[f'background_{name}_mps']=back[name]
            result[f'residual_{name}_mps']=residual
            result[f'residual_{name}_norm_mps']=np.linalg.norm(residual,axis=-1).astype(np.float32)
        affine=np.where(valid[...,None],data['residual_mps'],0).astype(np.float32)
        result['background_affine_mps']=np.where(valid[...,None],data['background_field_mps'],0).astype(np.float32)
        result['residual_affine_mps']=affine
        result['residual_affine_norm_mps']=np.linalg.norm(affine,axis=-1).astype(np.float32)
        assert all(np.isfinite(x).all() for x in result.values() if x.dtype.kind in 'fi')
        dest=d/f'{split}.npz';np.savez_compressed(dest,**result)
        report['splits'][split]={'counts':counts,'windows':n,'valid_steps':int(data['mask'].sum()),
            'input_sha256':sha(path),'audit_sha256':sha(audit),'output_sha256':sha(dest),
            'member_provenance_sha256':sha(path_out)}
        print('DONE',region,split,json.dumps(counts),flush=True)
    report.update(status='passed',elapsed_seconds=time.time()-started)
    (d/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('REGION_COMPLETE',region,round(time.time()-started,2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--regions',nargs='+',choices=REGIONS,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for r in a.regions:build_region(r,a.out)
