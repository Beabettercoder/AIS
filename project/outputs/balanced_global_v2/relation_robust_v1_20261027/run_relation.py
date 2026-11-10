"""Frozen development experiment. Only observed-array whitelist enters feature builders."""
from pathlib import Path
import csv, hashlib, importlib.util, json, math, os, sys
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OLD=ROOT/'original_method_and_relation_v1_20261026/relation'
OUT=ROOT/'relation_robust_v1_20261027'
sys.path.insert(0,str(OLD))
from observed_statistics_v2 import OBSERVED,score_observed
import build_speed_stress as stress

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def median_weighted(x,w):
    x=np.asarray(x,float);w=np.asarray(w,float);q=np.argsort(x,kind='stable');x=x[q];w=w[q]
    if w.sum()<=1e-12:return float(np.median(x))
    return float(x[min(np.searchsorted(np.cumsum(w),w.sum()/2,side='left'),len(x)-1)])
def weights(t,p,s,c):
    a=np.deg2rad(c);v=s[:,None]*(1852/3600)*np.column_stack((np.sin(a),np.cos(a)))
    dt=np.diff(t);valid=np.r_[False,(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])]
    pos=np.zeros_like(v);rep=np.zeros_like(v);ii=np.flatnonzero(valid);pos[ii]=(p[ii]-p[ii-1])/dt[ii-1,None];rep[ii]=(v[ii]+v[ii-1])/2
    e=np.linalg.norm(pos-rep,axis=1);rel=e/(np.minimum(np.linalg.norm(pos,axis=1),np.linalg.norm(rep,axis=1))+.3)
    w0=np.zeros(len(t));wm=np.zeros(len(t));w0[ii]=np.exp(-np.maximum(0,e[ii]-2.5)**2/8);wm[ii]=np.minimum(w0[ii],np.exp(-np.maximum(0,rel[ii]-1)**2/2))
    return valid,w0,wm,e,rel
def observed(z):return {k:np.asarray(z[k]).copy() for k in OBSERVED}
def extract(o,case):
    t=o['target_t'][case];s=o['target_sog_kn'][case];c=o['target_cog_deg'][case];n=len(t);anchor=max(2,n//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,n)[-3:]
    td=float(np.mean(s[post])/max(float(np.mean(s[pre])),.5)-1)
    changes=[];wabs=[];wmix=[];rchanges=[];grid_s=[];grid_w=[]
    for lo,hi in zip(o['member_offsets'][:-1],o['member_offsets'][1:]):
        lo=int(lo);hi=int(hi);nt=o['neighbor_t'][case,lo:hi];ns=o['neighbor_sog_kn'][case,lo:hi]
        valid,wa,wm,_,_=weights(nt,o['neighbor_pos'][case,lo:hi],ns,o['neighbor_cog_deg'][case,lo:hi]);ix=np.searchsorted(nt,t,side='right')-1;safe=np.maximum(ix,0)
        present=(ix>=0)&(t>=nt[safe])&(t-nt[safe]<=300);ai=pre[present[pre]];bi=post[present[post]]
        if len(ai)<2 or len(bi)<2:continue
        j=safe[ai];k=safe[bi];changes.append(float(ns[k].mean()/max(float(ns[j].mean()),.5)-1))
        wabs.append(float(np.sqrt(wa[j].mean()*wa[k].mean())));wmix.append(float(np.sqrt(wm[j].mean()*wm[k].mean())))
        x=float(np.average(ns[j],weights=wm[j])) if wm[j].sum()>1e-12 else float(ns[j].mean());y=float(np.average(ns[k],weights=wm[k])) if wm[k].sum()>1e-12 else float(ns[k].mean());rchanges.append(y/max(x,.5)-1)
        gs=np.full(n,np.nan);gw=np.zeros(n);gs[present]=ns[safe[present]];gw[present]=wm[safe[present]];grid_s.append(gs);grid_w.append(gw)
    if not changes:raise RuntimeError('No common observable members')
    changes=np.asarray(changes);wabs=np.asarray(wabs);wmix=np.asarray(wmix)
    mean=float(changes.mean());med=float(np.median(changes));wa=float(np.average(changes,weights=wabs)) if wabs.sum()>1e-12 else mean
    mixmean=float(np.average(changes,weights=wmix)) if wmix.sum()>1e-12 else mean;mixmed=median_weighted(changes,wmix)
    stats=dict(target_only=abs(td),neighbor_only=abs(mean),mean=abs(td-mean),median=abs(td-med),absolute_weighted_mean=abs(td-wa),mixed_weighted_mean=abs(td-mixmean),mixed_weighted_median=abs(td-mixmed),mixed_report_weighted_median=abs(td-float(np.median(rchanges))))
    grid_s=np.asarray(grid_s);grid_w=np.asarray(grid_w);ctx={mode:np.zeros((n,3),np.float32) for mode in ('mean','median','mixed_median')}
    for i in range(n):
        ok=np.isfinite(grid_s[:,i]);vals=grid_s[ok,i];ww=grid_w[ok,i]
        if not len(vals):continue
        for mode,value in [('mean',vals.mean()),('median',np.median(vals)),('mixed_median',median_weighted(vals,ww))]:ctx[mode][i,0]=value/10;ctx[mode][i,2]=len(vals)/8
    for v in ctx.values():v[1:,1]=np.diff(v[:,0])
    angle=np.deg2rad(c);target=np.column_stack((s/10,np.sin(angle),np.cos(angle),np.r_[0,np.diff(s)]/10)).astype(np.float32)
    features={m:np.concatenate((target,np.zeros((n,3),np.float32) if m=='target' else ctx[m]),axis=1) for m in ('target','mean','median','mixed_median')}
    return stats,features
def high_select(wid,members):
    return sorted(range(len(members)),key=lambda j:hashlib.sha256(('speed-stress-member:'+wid+':'+str(members[j])).encode()).digest())[:math.ceil(.5*len(members))]
def aggregate(rows,key):
    r=[x for x in rows if x['split']=='val'];y=np.array([x['y'] for x in r]);s=np.array([x[key] for x in r]);regions=sorted(set(x['region'] for x in r));reg={g:float(average_precision_score([x['y'] for x in r if x['region']==g],[x[key] for x in r if x['region']==g])) for g in regions}
    return {'AP':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s)),'macro_AP':float(np.mean(list(reg.values()))),'per_region_AP':reg}
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    manifest=json.loads((OLD/'all_sources_v1/relation_manifest.json').read_text());oldstress=json.loads((OLD/'neighbor_speed_stress_v1/stress_manifest.json').read_text());lookup={x['file']:x for x in oldstress['samples']}
    records=[];diagnostics=[];arrays={};provenance=[];observed_inputs=[];missing=[];modified=0;total=0;hits=0
    for item in manifest['samples']:
        path=OLD/'all_sources_v1'/item['file'];assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            o=observed(z);members=z['cohort_mmsi'].copy();wid=str(z['source_window_id']);metadata={k:str(z[k]) for k in ('target_mmsi','base_window_id','split','region')};y=z['y'].astype(int)
        se=lookup[item['file']];sp=OLD/'neighbor_speed_stress_v1'/se['file'];assert sha(sp)==se['sha256']
        with np.load(sp,allow_pickle=False) as z:existing=observed(z)
        stress.select_members=high_select
        high,proof=stress.stress_one({**o,'cohort_mmsi':members},wid)
        total+=len(members);modified+=proof['selected_member_count'];hits+=int(proof['source_suffix_overlap'])
        views={'base':o,'existing_speed':existing,'high_speed':high}
        for view,obs in views.items():
            for case in (0,1):
                for j,(lo,hi) in enumerate(zip(obs['member_offsets'][:-1],obs['member_offsets'][1:])):
                    lo=int(lo);hi=int(hi);tt=obs['neighbor_t'][case,lo:hi];ss=obs['neighbor_sog_kn'][case,lo:hi];cc=obs['neighbor_cog_deg'][case,lo:hi]
                    vv,wa,wm,e,er=weights(tt,obs['neighbor_pos'][case,lo:hi],ss,cc)
                    for i in np.flatnonzero(vv):
                        bas=float(o['neighbor_sog_kn'][case,lo+i]);changed=bool(ss[i]!=bas)
                        diagnostics.append(dict(view=view,split=metadata['split'],source=wid,region=metadata['region'],mmsi=metadata['target_mmsi'],neighbor=str(members[j]),neighbor_state=case,time=float(tt[i]),base_sog_kn=bas,changed=changed,w_abs=float(wa[i]),w_mixed=float(wm[i]),e_abs=float(e[i]),e_rel=float(er[i])))
            for case in range(4):
                st,feat=extract(obs,case);row=dict(view=view,source=wid,mmsi=metadata['target_mmsi'],region=metadata['region'],split=metadata['split'],case=case,y=int(y[case]),**st);records.append(row)
                for m,x in feat.items():arrays.setdefault((view,m),[]).append(x)
            if view=='base':
                ref=score_observed(obs)
                for k in range(4):
                    ours=records[-4+k]
                    for n,oldn in [('mean','relative_mean'),('median','relative_median'),('absolute_weighted_mean','relative_reliable')]:assert abs(ours[n]-ref[k][oldn])<1e-10,(wid,n)
        provenance.append(dict(source=wid,mmsi=metadata['target_mmsi'],region=metadata['region'],split=metadata['split'],base_file=str(path),base_sha256=sha(path),existing_file=str(sp),existing_sha256=sha(sp),selected_members=proof['members'],suffix_overlap=proof['source_suffix_overlap']))
    train_ids={r['mmsi'] for r in records if r['split']=='train'};val_ids={r['mmsi'] for r in records if r['split']=='val'};assert not train_ids&val_ids
    assert len(provenance)==385 and len(records)==385*4*3
    # Source/variant ordering is identical across views, then reindex by view.
    write_csv(OUT/'statistics_scores.csv',records);write_csv(OUT/'report_weight_diagnostics.csv',diagnostics)
    (OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf8')
    views=['base','existing_speed','high_speed'];methods=['target_only','neighbor_only','mean','median','absolute_weighted_mean','mixed_weighted_mean','mixed_weighted_median','mixed_report_weighted_median']
    summary={'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__)),'base_manifest_sha256':sha(OLD/'all_sources_v1/relation_manifest.json'),'old_stress_manifest_sha256':sha(OLD/'neighbor_speed_stress_v1/stress_manifest.json'),'counts':{'sources':len(provenance),'requested_high_members':modified,'all_members':total,'requested_member_fraction':modified/total,'high_speed_suffix_overlap_sources':hits,'train_mmsi':len(train_ids),'val_mmsi':len(val_ids),'train_val_mmsi_intersection':len(train_ids&val_ids)},'statistics':{}}
    for view in views:
        vr=[r for r in records if r['view']==view];summary['statistics'][view]={m:aggregate(vr,m) for m in methods}
    print(json.dumps({'counts':summary['counts'],'statistics':summary['statistics']},ensure_ascii=False),flush=True)
    # Save length and feature tensor explicitly so independent auditors can inspect allowed inputs.
    maxlen=max(x.shape[0] for lst in arrays.values() for x in lst)
    packed={};lengths={}
    for (view,mode),lst in arrays.items():
        z=np.zeros((len(lst),maxlen,7),dtype='float32');lens=np.array([len(x) for x in lst],dtype='int64')
        for i,x in enumerate(lst):z[i,:len(x)]=x
        packed[view+'__'+mode]=z;lengths[view]=lens
    np.savez_compressed(OUT/'feature_tensors.npz',**packed,**{'length__'+k:v for k,v in lengths.items()})
    # Every view has source/case order identical, target-only signatures balanced within quartet.
    byview={v:[r for r in records if r['view']==v] for v in views};vr=byview['base'];n=len(vr);splits=np.array([r['split'] for r in vr]);tr=np.flatnonzero(splits=='train');va=np.flatnonzero(splits=='val');y=np.array([r['y'] for r in vr]);assert np.array_equal(packed['base__target'][0::4],packed['base__target'][1::4]);assert np.array_equal(packed['base__target'][2::4],packed['base__target'][3::4])
    class Model(nn.Module):
        def __init__(self,arch):
            super().__init__();self.arch=arch
            self.body=nn.GRU(7,16,batch_first=True) if arch=='gru' else nn.Sequential(nn.Conv1d(7,16,3,padding=2),nn.ReLU(),nn.Conv1d(16,16,3,padding=2),nn.ReLU())
            self.out=nn.Linear(16,1)
        def forward(self,x,l):
            if self.arch=='gru':
                h,_=self.body(x);return self.out(h[torch.arange(len(l),device=l.device),l-1]).squeeze(-1)
            h=x.transpose(1,2)
            # Chopping each padded convolution keeps temporal outputs causal.
            h=self.body[1](self.body[0](h)[:,:,:x.shape[1]]);h=self.body[3](self.body[2](h)[:,:,:x.shape[1]])
            mask=(torch.arange(x.shape[1],device=x.device)[None,:]<l[:,None]).float();return self.out((h*mask[:,None,:]).sum(2)/l[:,None]).squeeze(-1)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');result=[];runmeta=[]
    for arch in ('tcn','gru'):
      for mode in ('target','mean','median','mixed_median'):
        base=packed['base__'+mode];lens=lengths['base'];mask=np.arange(maxlen)[None,:]<lens[:,None];valid=base[tr][mask[tr]];mu=valid.mean(0);sd=valid.std(0);sd[sd<1e-6]=1
        for seed in (0,1,2):
            torch.manual_seed(seed);np.random.seed(seed);model=Model(arch).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(seed);losslog=[]
            for epoch in range(20):
                model.train();batchloss=[]
                for ids in np.array_split(rng.permutation(tr),int(np.ceil(len(tr)/64))):
                    X=torch.tensor((base[ids]-mu)/sd,device=dev);L=torch.tensor(lens[ids],device=dev);Y=torch.tensor(y[ids],dtype=torch.float32,device=dev);opt.zero_grad();p=model(X,L);loss=nn.functional.binary_cross_entropy_with_logits(p,Y);loss.backward();opt.step();batchloss.append(float(loss.detach().cpu()))
                losslog.append(float(np.mean(batchloss)))
            model.eval();artifact=OUT/f'model_{arch}_{mode}_seed{seed}.pt';torch.save({'state_dict':model.state_dict(),'mean':mu,'std':sd,'architecture':arch,'mode':mode,'seed':seed,'loss':losslog},artifact)
            item=dict(architecture=arch,mode=mode,seed=seed,epochs=20,parameters=sum(p.numel() for p in model.parameters()),train_loss_first=losslog[0],train_loss_last=losslog[-1],checkpoint=artifact.name,checkpoint_sha256=sha(artifact),metrics={})
            with torch.no_grad():
                for view in views:
                    X=packed[view+'__'+mode];L=lengths[view];pred=[]
                    for ids in np.array_split(va,int(np.ceil(len(va)/64))):
                        out=model(torch.tensor((X[ids]-mu)/sd,device=dev),torch.tensor(L[ids],device=dev));pred.extend(torch.sigmoid(out).cpu().numpy().tolist())
                    rr=[]
                    for idx,p in zip(va,pred):
                        b=byview[view][idx];row=dict(view=view,architecture=arch,mode=mode,seed=seed,source=b['source'],mmsi=b['mmsi'],region=b['region'],split=b['split'],case=b['case'],y=b['y'],score=float(p));result.append(row);rr.append({**row,'score':float(p)})
                    item['metrics'][view]=aggregate(rr,'score')
            runmeta.append(item);print('DONE',arch,mode,seed,json.dumps({v:item['metrics'][v]['AP'] for v in views}),flush=True)
    write_csv(OUT/'neural_predictions.csv',result);summary['neural_runs']=runmeta
    (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
