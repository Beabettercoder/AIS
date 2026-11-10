from pathlib import Path
import hashlib,json,numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
SRC=ROOT/'original_method_and_relation_v1_20261026/data'
DEST=ROOT/'gating_training_v2_20261027/data'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    manifest=[]
    for region in REGIONS:
        (DEST/region).mkdir(parents=True,exist_ok=False)
        for split in ('train','val'):
            src=SRC/region/(split+'.npz'); before=sha(src)
            with np.load(src,allow_pickle=False) as z: d={k:z[k] for k in z.files}
            mask=d['mask'].astype(bool); valid=d['kinematic_valid'].astype(bool); raw=d['raw_legacy7']; w=d['w_self']; dt=np.diff(d['time_s'],axis=1)
            assert np.isfinite(raw).all() and np.isfinite(w).all()
            # Preserve old historical padding semantics and holding of velocity/heading slots.
            old=d['kg_compat8'][...,:7].copy(); alt=old.copy(); altered=np.zeros_like(mask)
            for i in range(len(mask)):
                n=int(mask[i].sum()); current=np.zeros(2,np.float64)
                for t in range(1,n):
                    dp=raw[i,t,:2].astype(np.float64)-raw[i,t-1,:2].astype(np.float64)
                    step=dp*float(w[i,t]); interval=dt[i,t-1]
                    if valid[i,t] and mask[i,t-1] and np.isfinite(interval) and interval>0 and np.isfinite(raw[i,t-1:t+1,2:4]).all():
                        vbar=.5*(raw[i,t-1,2:4].astype(np.float64)+raw[i,t,2:4].astype(np.float64))
                        step=step+(1.-float(w[i,t]))*vbar*interval
                        altered[i,t]=True
                    current+=step; alt[i,t,:2]=current
                # Same legacy carry into padding as original KG, for equal comparison.
                alt[i,n:,:2]=current
            wv=np.stack([w,valid.astype(np.float32)],-1)
            d['raw_equal9']=np.concatenate([raw,np.zeros_like(wv)],-1)
            d['old_gate9']=np.concatenate([old,wv],-1)
            d['comp_gate9']=np.concatenate([alt,wv],-1)
            d['old_lf10']=np.concatenate([old,d['rf'][...,None],wv],-1)
            d['comp_lf10']=np.concatenate([alt,d['rf'][...,None],wv],-1)
            d['comp_interval_valid']=altered
            for key in ('raw_equal9','old_gate9','comp_gate9','old_lf10','comp_lf10'):
                assert np.isfinite(d[key]).all()
            assert np.array_equal(old,d['kg_compat8'][...,:7])
            assert np.array_equal(d['old_gate9'][...,2:],d['comp_gate9'][...,2:])
            path=DEST/region/(split+'.npz');np.savez_compressed(path,**d)
            assert sha(src)==before
            manifest.append(dict(region=region,split=split,source=str(src),source_sha256=before,output=str(path),output_sha256=sha(path),windows=len(mask),valid_steps=int(mask.sum()),valid_compensation_intervals=int(altered.sum()),low_weight_eligible=int((altered&(w<.3)).sum())))
    (DEST/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
