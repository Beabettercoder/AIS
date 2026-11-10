"""Train matched original late-fusion backbones; test only after frozen manifest."""
from pathlib import Path
import argparse,hashlib,json,random,sys
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm
from radius_core import ARMS

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs');sys.path.insert(0,str(ROOT/'framework_validation_v1_20261028'));sys.path.insert(0,str(ROOT/'original_mainline_pilot_20261029'));sys.path.insert(0,str(ROOT/'formal_20261101/single'))
from eight_backbone_adapter import Model8
from pilot_historical_loss import historical_bundle

REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def data(root,split,view='base'):
    values=[];hashes={}
    for r in REGIONS:
        suffix='' if view=='base' else '_'+view
        p=root/'features'/f'{split}_{r}{suffix}.npz';side=p.with_suffix('.json')
        md=json.loads(side.read_text(encoding='utf8'))
        if sha(p)!=md['out_sha256']:raise ValueError('feature checksum mismatch '+str(p))
        hashes[str(p)]=md['out_sha256'];hashes[str(side)]=sha(side)
        with np.load(p,allow_pickle=False) as z:values.append({k:z[k] for k in z.files})
    d={k:np.concatenate([v[k] for v in values]) for k in values[0]}
    if len(set(d['window_id'].astype(str)))!=len(d['window_id']):raise ValueError('duplicate window IDs')
    return d,hashes

def normalization(d):
    m=d['mask'];selfs=np.concatenate([d['raw'][m],d['target'][m]]).astype(np.float64)
    plain=('tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_plain_deg6')
    fields=np.concatenate([d[arm+'__field'][d[arm+'__mode']>0] for arm in plain]).astype(np.float64)
    if not len(fields):raise ValueError('no valid training fields')
    def stats(x):
        mu=x.mean(0);sd=x.std(0);return mu,np.where(sd<1e-7,1.,sd)
    sm,ss=stats(selfs);fm,fs=stats(fields);return dict(sm=sm,ss=ss,fm=fm,fs=fs)

def prepare(d,n,arm,dev):
    mask=d['mask'].astype(bool);target=((d['target']-n['sm'])/n['ss']).astype('float32');target[~mask]=0
    mode=d[arm+'__mode'];field=((d[arm+'__field']-n['fm'])/n['fs']).astype('float32');field[mode==0]=0
    tm=(d['time_s']-d['time_s'][:,:1]).astype('float32');tm[~mask]=0
    cats={v:i for i,v in enumerate(('turn_back','deviation','stall','sinuous','speed_drop'),1)}
    yc=np.asarray([cats.get(str(v),0) for v in d['behavior_type']],np.int64)
    if not np.array_equal(yc>0,d['y']>0):raise ValueError('behavior category mismatch')
    vals={'x':target,'mask':mask,'times':tm,'field':field,'rho':d[arm+'__rho'].astype('float32'),
          'state':d[arm+'__state'].astype('float32'),'y':d['y'].astype('float32'),'yc':yc}
    return {k:torch.as_tensor(v,device=dev) for k,v in vals.items()}

def pred(model,z,batch):
    model.eval();out=[]
    with torch.no_grad():
        for st in range(0,len(z['y']),batch):
            ix=slice(st,st+batch);out.extend(torch.sigmoid(model(z['x'][ix],z['mask'][ix],z['times'][ix],z['field'][ix],z['rho'][ix],True)[0]).cpu().numpy())
    return np.asarray(out)

def metrics(y,p,thresholds=None):
    out={'pooled_AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p))}
    if thresholds:
        for q,t in thresholds.items():out['FAR@'+q]=float(np.mean(p[y==0]>=t));out['recall@'+q]=float(np.mean(p[y==1]>=t))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['tcn','gru','transformer','graph'],required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--root',type=Path,default=Path('/home/ps/AIS_adaptive_radius_relation_v2p1_20261109'));p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    tr,trhash=data(a.root,'train');va,vahash=data(a.root,'val')
    if set(tr['mmsi'].astype(str))&set(va['mmsi'].astype(str)):raise ValueError('train/val target identity overlap')
    norm=normalization(tr);train=prepare(tr,norm,a.arm,dev);val=prepare(va,norm,a.arm,dev)
    if a.smoke:
        for z in (train,val):
            for k in z:z[k]=z[k][:128]
    model=Model8(a.name).to(dev);out=a.root/('smoke_runs' if a.smoke else 'runs')/'single'/a.name/a.arm/f'seed{a.seed}'
    if a.mode=='test':
        manifest=json.loads((a.root/'frozen_manifest.json').read_text(encoding='utf8'));key=f'single/{a.name}/{a.arm}/seed{a.seed}'
        if key not in manifest['runs']:raise ValueError('run missing from frozen manifest')
        for path,digest in manifest['source_sha256'].items():
            if sha(path)!=digest:raise ValueError('source changed after freeze '+path)
        for k,v in (trhash|vahash).items():
            if manifest['input_sha256'].get(k)!=v:raise ValueError('data changed after freeze '+k)
        for ckname in ('last','best_val'):
            cp=out/f'{ckname}.pt'
            if sha(cp)!=manifest['runs'][key][ckname]:raise ValueError('checkpoint modified after freeze')
            ck=torch.load(cp,map_location=dev,weights_only=False);model.load_state_dict(ck['model'])
            pv=pred(model,val,256);yv=va['y'][:len(pv)];th={f'{q}%':float(np.quantile(pv[yv==0],1-q/100,method='higher')) for q in (1,5)}
            save={};res={'validation':metrics(yv,pv,th),'thresholds':th,'views':{}}
            test_reference=None
            for view in ('base','natural','suffix','combined_l100_m600'):
                te,h=data(a.root,'test',view)
                for path,digest in h.items():
                    if manifest['input_sha256'].get(path)!=digest:raise ValueError('test features changed after freeze '+path)
                if set(te['mmsi'].astype(str))&(set(tr['mmsi'].astype(str))|set(va['mmsi'].astype(str))):raise ValueError('test target identity overlap')
                if test_reference is None:test_reference=te;save={k:te[k] for k in ('y','mmsi','region','window_id','observation_noise','target_noise_effective','neighbor_noise_effective','eligible_count_8nm','mask')}
                else:
                    for k in ('window_id','mmsi','region','y','mask','time_s'):
                        if not np.array_equal(test_reference[k],te[k]):raise ValueError('stress view metadata mismatch '+k)
                    save[view+'_hit']=te['stress_hit']
                tz=prepare(te,norm,a.arm,dev);score=pred(model,tz,256);save[view]=score;res['views'][view]=metrics(te['y'],score,th)
            np.savez_compressed(out/f'test_{ckname}_scores.npz',**save)
            (out/f'test_{ckname}_metrics.json').write_text(json.dumps(res,indent=2),encoding='utf8');print('TEST',a.name,a.arm,a.seed,ckname,res,flush=True)
        return
    if out.exists() and not a.smoke:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
    init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.parameters())).hexdigest()
    counts=np.bincount(train['yc'][train['yc']>0].cpu().numpy()-1,minlength=5);cw=torch.as_tensor(1./np.maximum(counts,1),dtype=torch.float32,device=dev)
    spec={'name':a.name,'arm':a.arm,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'optimizer':'AdamW lr=.002, wd=.01; grad clip=1', 'model':'formal Model8 target causal7 and alpha-zero late 4D field adapter', 'loss':'historical_bundle plus per-arm physical field state','primary_checkpoint':'last epoch 25; best pooled validation AP also saved', 'normalization':'training-only pooled ordinary-field arms; no neighbor reliability in ordinary-field normalization; target raw/causal pooled', 'initial_parameter_sha256':init,'parameter_count':sum(p.numel() for p in model.parameters()), 'user_note':'observational area label and low-density stratum; no confirmed far-ocean assignment'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8');best=-1.;history=[]
    for epoch in range(a.epochs):
        model.train();order=np.random.default_rng(300000+epoch).permutation(len(train['y']));losses=[]
        bar=tqdm(range(0,len(order),a.batch_size),desc=f'{a.name}/{a.arm} seed={a.seed} {epoch+1}/{a.epochs}',file=sys.stdout,dynamic_ncols=True)
        for st in bar:
            ix=torch.as_tensor(order[st:st+a.batch_size],device=dev);b={k:v[ix] for k,v in train.items()}
            opt.zero_grad(set_to_none=True);pb,pc,h,seq=model(b['x'],b['mask'],b['times'],b['field'],b['rho'],True)
            keep=(torch.rand(b['x'].shape[:2],device=dev)>.15).to(b['x'].dtype);x2=b['x']*keep.unsqueeze(-1);lengths=b['mask'].sum(1)
            j=torch.floor(torch.rand(len(x2),device=dev)*lengths).long();x2=x2*b['mask'].unsqueeze(-1);x2[torch.arange(len(x2),device=dev),j,0]+=2.
            retention=(keep*b['mask']).sum(1)/lengths.clamp_min(1);_,_,h2,_=model(x2,b['mask'],b['times'],b['field'],b['rho'],True)
            loss,_=historical_bundle(model.backbone.head,model.typehead,model.ctrl,pb,pc,h2,seq,b['state'],b['mask'],b['y'],b['yc'],cw,retention)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()));bar.set_postfix(loss=f'{losses[-1]:.4f}')
        pv=pred(model,val,256);yv=va['y'][:len(pv)];m=metrics(yv,pv);row={'epoch':epoch+1,'loss':float(np.mean(losses)),**m};history.append(row)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'norm':norm,'spec':spec};torch.save(ck,out/'last.pt')
        if m['pooled_AP']>best:
            best=m['pooled_AP'];torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',score=pv,y=yv,mmsi=va['mmsi'][:len(pv)],region=va['region'][:len(pv)],window_id=va['window_id'][:len(pv)])
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(row),flush=True)
if __name__=='__main__':main()
