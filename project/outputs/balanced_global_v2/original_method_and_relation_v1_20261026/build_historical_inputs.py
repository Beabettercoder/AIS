"""Bridge audited new observations into the actual historical Table-1 schema.

Restore physical channel meanings and legacy gate/padding behavior explicitly.
Keep corrected identity/causal/leave-one-out protections and real graph inputs.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys,time
import numpy as np
import pandas as pd

REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
CLASSES={'none':0,'turn_back':1,'deviation':2,'stall':3,'sinuous':4,'speed_drop':5}
KNOT=1852/3600

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False,
                  default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())

def gate(raw,w,mask,masked=False):
    out=raw.copy()
    for i in range(len(raw)):
        n=int(mask[i].sum()) if masked else raw.shape[1]
        # Historical diff uses prepend=0 and treats padding w=0 as carry-forward.
        dp=np.diff(raw[i,:n,:2],axis=0,prepend=np.zeros((1,2)))*w[i,:n,None]
        out[i,:n,:2]=np.cumsum(dp,axis=0)
        for t in range(1,n):
            if w[i,t]<.2:out[i,t,2:7]=out[i,t-1,2:7]
        if masked:out[i,n:]=0
    return out.astype(np.float32)

def target_bridge(z):
    mask=z['mask'].copy();n,T=mask.shape
    if not np.array_equal(mask,np.arange(T)[None,:]<mask.sum(1)[:,None]):raise ValueError('Right padding required')
    raw=np.zeros((n,T,7),np.float32)
    raw[...,:2]=(z['target_pos']-z['target_pos'][:,0:1,:]).astype(np.float32)
    angle=np.deg2rad(z['target_cog_deg']);speed=z['target_sog_kn']
    raw[...,2]=(speed*KNOT*np.sin(angle)).astype(np.float32);raw[...,3]=(speed*KNOT*np.cos(angle)).astype(np.float32)
    raw[...,4]=speed;raw[...,5]=np.sin(angle);raw[...,6]=np.cos(angle);raw[~mask]=0
    w=z['self_gate_weight'].astype(np.float32).copy();w[~mask]=0
    rf=np.linalg.norm(z['residual_mps'],axis=-1).astype(np.float32);rf[~mask]=0
    g=gate(raw,w,mask,False);gm=gate(raw,w,mask,True)
    ep=np.zeros((n,T,4),np.float32)
    ep[...,0]=np.linalg.norm(z['background_field_mps'],axis=-1)
    ep[...,2]=z['divergence_per_s'];ep[...,3]=z['curl_per_s']
    dt=np.diff(z['time_s'],axis=1);dv=np.abs(np.diff(ep[...,0],axis=1))
    ep[:,1:,1]=dv/np.maximum(dt,1.)
    ep[~mask]=0
    labels=np.array([CLASSES[str(x)] for x in z['behavior_type']],np.int64)
    assert np.array_equal(labels>0,z['y'].astype(bool))
    out={k:z[k].copy() for k in ['mask','time_s','y','window_id','base_window_id','mmsi','region','behavior_type',
                                 'noise_type','noise_scope','observation_noise','observation_noise_effective',
                                 'target_noise_effective','neighbor_noise_effective','target_pos','target_sog_kn','target_cog_deg',
                                 'kinematic_valid','context_valid','plain_context_valid','boundary_context_valid']}
    out.update(raw_legacy7=raw,raw_incremental7=z['raw'].astype(np.float32).copy(),
               kg_compat8=np.concatenate([g,w[...,None]],-1),
               kglf_compat9=np.concatenate([g,rf[...,None],w[...,None]],-1),
               kglf_masked9=np.concatenate([gm,rf[...,None],w[...,None]],-1),
               w_self=w,rf=rf,ep=ep,label_class=labels)
    for key in ['raw_legacy7','raw_incremental7','kg_compat8','kglf_compat9','kglf_masked9','rf','ep']:
        assert np.isfinite(out[key]).all(),key
    return out

def build_region(region,root,out,Source,allowed,protocol_sha):
    started=time.time();reg=json.loads((root/'release_20261026/source_registry.json').read_text())['regions'][region]
    source=Source(reg);dest=out/region;dest.mkdir(parents=True,exist_ok=False)
    report={'status':'running','region':region,'source_canonical':reg['canonical_file'],'source_sha256':reg['sha256'],
            'protocol_sha256':protocol_sha,'builder_sha256':sha(Path(__file__)),'splits':{}}
    for split in ['train','val']:
        path=root/'labeling/labeled_v1/features_scene'/region/f'{split}.npz'
        audit=path.with_name(f'{split}_context_audit.jsonl.gz');source_sha=sha(path);audit_sha=sha(audit)
        with np.load(path,allow_pickle=False) as f:z={k:f[k] for k in f.files}
        data=target_bridge(z);N,T=data['mask'].shape
        data.update(nb_seq=np.zeros((N,32,5,4),np.float32),nb_xy=np.zeros((N,32,2),np.float32),
                    nb_v=np.zeros((N,32,2),np.float32),nb_w=np.zeros((N,32),np.float32),nb_mask=np.zeros((N,32),bool),
                    nb_hist_mask=np.zeros((N,32,5),bool),nb_mmsi=np.full((N,32),'',dtype='<U16'),
                    nb_time=np.zeros((N,32),np.float64),nb_hist_time=np.zeros((N,32,5),np.float64))
        graph_reports=0;hist_reports=0;invalid_hist=0;graph_windows=0;counts=[]
        graph_audit=dest/f'{split}_graph_provenance.jsonl.gz'
        with gzip.open(audit,'rt',encoding='utf-8') as rows,gzip.open(graph_audit,'xt',encoding='utf-8') as evidence:
            for j,line in enumerate(rows):
                md=json.loads(line);assert md['window_id']==str(data['window_id'][j]) and md['split']==split
                length=int(data['mask'][j].sum());final=length-1;decision=float(data['time_s'][j,final]);target=str(data['mmsi'][j])
                assert decision==float(md['decision_time_s'])
                st=next(x for x in md['steps'] if x['step']==final)
                candidates=[x for x in st.get('neighbors',[]) if x['plain_accepted']]
                assert len({str(x['mmsi']) for x in candidates})==len(candidates)
                for x in candidates:
                    assert str(x['mmsi'])!=target and str(x['mmsi']) in allowed[split]
                    assert 0<=decision-float(x['time_s'])<=300
                enable=bool(data['boundary_context_valid'][j,final] and len(candidates)>=4)
                selected=sorted(candidates,key=lambda x:(float(x['distance_m']),str(x['mmsi'])))[:32] if enable else []
                overlay=source.overlay(md) if selected else {}
                identities=[]
                for k,entry in enumerate(selected):
                    mid=str(entry['mmsi']);stamp=float(entry['time_s'])
                    pos,sog,cog,kv,err,w=source.observation(mid,stamp,overlay)
                    assert np.isfinite(pos).all() and 0.5<=sog<102.2 and 0<=cog<360
                    assert bool(kv)==bool(entry['kinematic_valid']) and abs(float(w)-entry['reliability'])<1e-8
                    relative=pos-data['target_pos'][j,final]
                    assert abs(np.linalg.norm(relative)-entry['distance_m'])<1e-6
                    data['nb_mask'][j,k]=True;data['nb_mmsi'][j,k]=mid;data['nb_time'][j,k]=stamp
                    data['nb_xy'][j,k]=relative;data['nb_w'][j,k]=w
                    angle=np.deg2rad(cog);data['nb_v'][j,k]=sog*KNOT*np.array([np.sin(angle),np.cos(angle)])
                    ship=source.ship[mid];pin=int(np.searchsorted(source.t[ship],decision,side='right'));hist=ship[max(0,pin-5):pin]
                    assert len(hist)>0 and source.t[hist[-1]]==stamp and np.all(source.t[hist]<=decision)
                    pad=5-len(hist);bad=0
                    for q,ix in enumerate(hist,pad):
                        hp,hs,hc,*_=source.observation(mid,float(source.t[ix]),overlay)
                        good=bool(np.isfinite(hp).all() and np.isfinite(hs) and np.isfinite(hc) and 0<=hs<102.2 and 0<=hc<360)
                        data['nb_hist_time'][j,k,q]=source.t[ix];data['nb_hist_mask'][j,k,q]=good
                        if good:
                            a=np.deg2rad(hc);data['nb_seq'][j,k,q]=np.r_[hp-pos,hs*KNOT*np.sin(a),hs*KNOT*np.cos(a)]
                        else:bad+=1
                    hist_reports+=len(hist);invalid_hist+=bad;graph_reports+=1
                    identities.append({'mmsi':mid,'time_s':stamp,'distance_m':float(entry['distance_m']),'reliability':float(w),
                                       'history_time_s':source.t[hist].tolist(),'invalid_history_rows_zeroed':bad})
                counts.append(len(selected));graph_windows+=bool(selected)
                evidence.write(json.dumps({'window_id':str(data['window_id'][j]),'base_window_id':str(data['base_window_id'][j]),
                     'mmsi':target,'split':split,'decision_time_s':decision,'last_valid_step':final,
                     'plain_candidate_count':len(candidates),'boundary_context_valid':bool(data['boundary_context_valid'][j,final]),
                     'selected':identities},ensure_ascii=False)+'\n')
                if (j+1)%200==0:print('BRIDGE',region,split,j+1,flush=True)
        assert j+1==N
        for key in ['nb_seq','nb_xy','nb_v','nb_w']:assert np.isfinite(data[key]).all()
        file=dest/f'{split}.npz';np.savez_compressed(file,**data)
        assert sha(path)==source_sha and sha(audit)==audit_sha
        clean=~data['observation_noise'].astype(bool)
        report['splits'][split]={'source_path':str(path),'source_sha256':source_sha,'source_audit_sha256':audit_sha,
            'windows':N,'clean_gradient_windows':int(clean.sum()),'clean_positive_windows':int(data['y'][clean].sum()),
            'short_windows':int((data['mask'].sum(1)<20).sum()),'graph_windows':int(graph_windows),
            'selected_neighbor_reports':graph_reports,'neighbor_history_reports':hist_reports,'invalid_history_rows_zeroed':invalid_hist,
            'max_neighbors':max(counts),'output_sha256':sha(file),'graph_provenance_sha256':sha(graph_audit),
            'train_mask_counts':np.bincount(data['label_class'][clean],minlength=6).tolist()}
    report.update(status='passed',elapsed_seconds=time.time()-started)
    save(dest/'input_bridge_report.json',report);print('REGION_COMPLETE',region,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--source-helper',type=Path,required=True)
    ap.add_argument('--regions',nargs='+',default=list(REGIONS));args=ap.parse_args()
    sys.path.insert(0,str(args.source_helper.parent));from audit_neighborhood import Source
    registry=args.root/'release_20261026'
    allids=set(pd.read_parquet(registry/'global_all_mmsi_split.parquet',columns=['mmsi']).mmsi.astype(str))
    target=pd.read_parquet(registry/'selected_windows.parquet',columns=['mmsi','split'])
    val=set(target.loc[target.split=='val','mmsi'].astype(str));test=set(target.loc[target.split=='test','mmsi'].astype(str))
    allowed={'train':allids-val-test,'val':allids-test};args.out.mkdir(parents=True,exist_ok=True)
    for region in args.regions:build_region(region,args.root,args.out,Source,allowed,sha(args.protocol))
if __name__=='__main__':main()
