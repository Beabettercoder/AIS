"""Paired multi-encoder developmental study. No evaluation-set fitting."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, math, random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from models import make_model

ROOT=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)
def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def metrics(y,p,regions):
    y=np.asarray(y);p=np.asarray(p);rs={}
    for r in np.unique(regions):
        s=regions==r;rs[str(r)]={'n':int(s.sum()),'positive':int(y[s].sum()),'AP':float(average_precision_score(y[s],p[s])) if y[s].sum()>0 else None}
    return {'pooled_AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'macro_AP':float(np.mean([x['AP'] for x in rs.values() if x['AP'] is not None])),'per_region':rs}
def get_mask(data):
    if 'mask' in data:return data['mask'].astype(bool)
    return np.arange(data['base__C0'].shape[1])[None,:]<data['length'][:,None]
def context_mask(data,key):
    return get_mask(data)&(data[key][...,-1]>0)
def fit_normalizer(train,task):
    nf=train['base__C0'].shape[-1];target=7 if task=='individual' else 4
    mu=np.zeros(nf,np.float64);sd=np.ones(nf,np.float64)
    a=train['base__C0'][get_mask(train),:target].astype(np.float64)
    mu[:target]=a.mean(0);sd[:target]=a.std(0)
    pieces=[]
    for arm in ('C1','C2','C3'):
        key='base__'+arm;a=train[key];pieces.append(a[context_mask(train,key),target:-1].astype(np.float64))
    a=np.concatenate(pieces)
    if len(a):mu[target:-1]=a.mean(0);sd[target:-1]=a.std(0)
    sd[sd<1e-6]=1.
    return {'mean':mu.tolist(),'std':sd.tolist(),'target_dim':target,'fit_source':'train/base C0 target + pooled C1,C2,C3 context','context_rows':len(a),'clip':10.0}
def normalize(data,key,norm):
    x=data[key].copy().astype(np.float32);mask=get_mask(data);cm=context_mask(data,key);support=x[...,-1].copy();target=norm['target_dim']
    x=(x-np.asarray(norm['mean'],np.float32))/np.asarray(norm['std'],np.float32)
    x[...,:-1]=np.clip(x[...,:-1],-10,10);x[...,-1]=support
    x[...,target:-1]*=cm[...,None];x[~mask]=0
    feature_mask=np.broadcast_to(mask[...,None],x.shape).copy();feature_mask[...,target:-1]&=cm[...,None];feature_mask[...,-1]=False
    if key.endswith('C0'):x[...,target:]=0;feature_mask[...,target:]=False
    assert np.isfinite(x).all()
    return x,feature_mask
def times_array(data):
    for key in ('time_s','times'):
        if key in data:
            a=data[key].astype(np.float64).copy();m=get_mask(data)
            for i in range(len(a)):
                ix=np.flatnonzero(m[i]);assert len(ix)>1
                a[i,ix]-=a[i,ix[0]];a[i,~m[i]]=a[i,ix[-1]]
            assert np.isfinite(a).all()
            return a.astype(np.float32)
    raise RuntimeError('Actual source timestamps are required')
def tensor_data(data,key,norm,device):
    x,fm=normalize(data,key,norm)
    return {'x':torch.tensor(x,device=device),'fm':torch.tensor(fm,device=device),'mask':torch.tensor(get_mask(data),device=device),'times':torch.tensor(times_array(data),device=device),'y':torch.tensor(data['y'],dtype=torch.float32,device=device)}
def perturb(x,valid,generator):
    w=valid.to(x.dtype);count=w.sum(1,keepdim=True).clamp_min(1)
    mu=(x*w).sum(1,keepdim=True)/count
    var=(((x-mu)*w)**2).sum(1,keepdim=True)/count
    sd=var.sqrt();shape=(x.shape[0],1,x.shape[2])
    a=.9+.2*torch.rand(shape,device=x.device,generator=generator)
    b=-.1+.2*torch.rand(shape,device=x.device,generator=generator)
    aug=a*(x-mu)+mu+b*sd
    return torch.where(valid,aug,x)
def bernoulli_js(logits_a,logits_b):
    p=torch.sigmoid(logits_a).clamp(1e-6,1-1e-6);q=torch.sigmoid(logits_b).clamp(1e-6,1-1e-6);m=(p+q)/2
    def kl(a,b):return a*(a.log()-b.log())+(1-a)*((1-a).log()-(1-b).log())
    return .5*(kl(p,m)+kl(q,m)).mean()
@torch.no_grad()
def predict(model,td,batch_size=256):
    model.eval();out=[]
    for i in range(0,len(td['y']),batch_size):out.append(torch.sigmoid(model(td['x'][i:i+batch_size],td['mask'][i:i+batch_size],td['times'][i:i+batch_size])['logits']).cpu().numpy())
    p=np.concatenate(out).astype(np.float64);assert np.isfinite(p).all();return p
def get_regions(data):
    return np.asarray(data['region']).astype(str)
def save_predictions(path,data,p):
    keys=('y','mmsi','window_id','source','source_id','base_window_id','region','case','split','time_start_s','time_end_s','noise_type','noise_scope','behavior_type','observation_noise','target_noise_effective','neighbor_noise_effective')
    keep={k:data[k] for k in keys if k in data};keep['score']=np.asarray(p,np.float64)
    np.savez_compressed(path,**keep)
def thresholds(y,p,budgets):
    neg=np.sort(p[np.asarray(y)==0]);n=len(neg);out={}
    for alpha in budgets:
        allowed=int(math.floor(alpha*n));index=max(0,n-allowed-1);t=float(neg[index]);out[str(alpha)]={'threshold':t,'validation_negative_count':n,'validation_false_positives':int((neg>t).sum()),'nominal_FAR':alpha}
    return out
def operating(y,p,th):
    y=np.asarray(y);out={}
    for k,v in th.items():
        pr=p>v['threshold'];n0=int((y==0).sum());n1=int((y==1).sum());fp=int((pr&(y==0)).sum());tp=int((pr&(y==1)).sum())
        out[k]={'threshold':v['threshold'],'FAR':fp/n0 if n0 else None,'recall':tp/n1 if n1 else None,'false_positive':fp,'true_positive':tp,'negative_count':n0,'positive_count':n1}
    return out
def verify_lock():
    lock=json.loads((ROOT/'code_lock.json').read_text())
    for name,h in lock['files'].items():assert sha(ROOT/name)==h,name
    return lock
def verify_data(lock,task,splits):
    for split in splits:
        name=f'data/{task}/{split}.npz';assert sha(ROOT/name)==lock['data_files'][name],name
def train_one(task,arch,arm,seed,device,train,val,norm,protocol,lock):
    out=ROOT/'runs'/task/arch/f'seed_{seed}'/arm
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text());assert result['code_lock_sha256']==sha(ROOT/'code_lock.json');assert sha(out/'final.pt')==result['checkpoint_sha256']
        for key,expected in [('task',task),('backbone',arch),('arm',arm),('seed',seed),('data_manifest_sha256',sha(ROOT/'data_manifest.json'))]:assert result[key]==expected,key
        assert sha(out/'configuration.json')==result['configuration_sha256']
        assert sha(out/'validation_base.npz')==result['validation_predictions_sha256']
        return result
    out.mkdir(parents=True,exist_ok=True)
    # Initialization is identical for every arm of a backbone/task/seed.
    seed_all(10000+seed);model=make_model(arch,train['base__C0'].shape[-1],max_len=20,hidden=64).to(device)
    initial=state_hash(model)
    decay=[p for p in model.parameters() if not getattr(p,'_no_weight_decay',False)]
    nodecay=[p for p in model.parameters() if getattr(p,'_no_weight_decay',False)]
    groups=[{'params':decay,'weight_decay':protocol['training']['weight_decay']}]
    if nodecay:groups.append({'params':nodecay,'weight_decay':0.0})
    opt=torch.optim.AdamW(groups,lr=protocol['training']['learning_rate'])
    tr=tensor_data(train,'base__'+arm,norm,device);va=tensor_data(val,'base__'+arm,norm,device)
    config={'task':task,'backbone':arch,'arm':arm,'seed':seed,'initial_state_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),'code_lock_sha256':sha(ROOT/'code_lock.json'),'data_manifest_sha256':sha(ROOT/'data_manifest.json'),'normalization':norm,'torch':torch.__version__,'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'}
    write_json(out/'configuration.json',config)
    epochs=protocol['training']['epochs'];bs=protocol['training']['batch_size'];history=[];start=0;best=-float('inf')
    ck=out/'resume.pt'
    if ck.exists():
        resume=torch.load(ck,map_location=device,weights_only=False);assert resume['code_lock_sha256']==config['code_lock_sha256']
        for key in ('task','backbone','arm','seed','data_manifest_sha256','initial_state_sha256'):assert resume['configuration'][key]==config[key],key
        model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);start=resume['epoch'];history=resume['history'];best=resume['best']
    begin=time.time()
    for epoch in range(start,epochs):
        if (ROOT/'STOP').exists():
            print('STOP_BOUNDARY',task,arch,arm,seed,epoch,flush=True);return None
        model.train();seed_all(100000+seed*1000+epoch)
        gen=torch.Generator(device=device);gen.manual_seed(900000+seed*1000+epoch)
        order=np.random.default_rng(300000+seed*1000+epoch).permutation(len(tr['y']))
        losses=[];clean_terms=[];aug_terms=[];js_terms=[]
        for b in range(0,len(order),bs):
            ix=torch.tensor(order[b:b+bs],device=device);x=tr['x'][ix];mask=tr['mask'][ix];times=tr['times'][ix];y=tr['y'][ix]
            opt.zero_grad(set_to_none=True);logits=model(x,mask,times)['logits'];loss=F.binary_cross_entropy_with_logits(logits,y);clean_terms.append(float(loss.detach()))
            if arm=='C4':
                aug=perturb(x,tr['fm'][ix],gen)
                devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(8000000+seed*100000+epoch*1000+b)
                    aug_logits=model(aug,mask,times)['logits']
                aug_loss=F.binary_cross_entropy_with_logits(aug_logits,y);js=bernoulli_js(logits,aug_logits)
                loss=.5*(loss+aug_loss)+.1*js;aug_terms.append(float(aug_loss.detach()));js_terms.append(float(js.detach()))
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {task} {arch} {arm} {seed} epoch{epoch}')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        vp=predict(model,va);vm=metrics(val['y'],vp,get_regions(val));primary=vm['macro_AP'] if task=='individual' else vm['pooled_AP']
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),'clean_BCE':float(np.mean(clean_terms)),'aug_BCE':float(np.mean(aug_terms)) if aug_terms else 0.,'binary_JS':float(np.mean(js_terms)) if js_terms else 0.,'validation':vm,'elapsed_seconds':time.time()-begin};history.append(row)
        if primary>best:
            best=primary;torch.save({'model':model.state_dict(),'epoch':epoch+1,'config':config},out/'best_validation.pt')
        tmp=out/'resume.tmp.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'history':history,'best':best,'configuration':config,'code_lock_sha256':config['code_lock_sha256']},tmp);tmp.replace(ck)
        write_json(out/'history.json',history)
        if (epoch+1)%5==0 or epoch==0:print('EPOCH',task,arch,arm,seed,epoch+1,round(primary,6),round(time.time()-begin,1),flush=True)
    vp=predict(model,va);save_predictions(out/'validation_base.npz',val,vp)
    th=thresholds(val['y'],vp,protocol['evaluation']['false_alarm_budgets'])
    final={'model':{k:v.cpu() for k,v in model.state_dict().items()},'epoch':epochs,'config':config,'normalization':norm,'thresholds':th}
    torch.save(final,out/'final.pt')
    result={**config,'configuration_sha256':sha(out/'configuration.json'),'checkpoint_sha256':sha(out/'final.pt'),'validation_predictions_sha256':sha(out/'validation_base.npz'),'validation':metrics(val['y'],vp,get_regions(val)),'validation_operating':operating(val['y'],vp,th),'thresholds':th,'train_elapsed_seconds':time.time()-begin,'epochs':epochs,'status':'complete_training_no_test_yet'}
    write_json(out/'result.json',result);print('TRAIN_COMPLETE',task,arch,arm,seed,round(result['train_elapsed_seconds'],1),flush=True)
    return result
def train(args):
    lock=verify_lock();verify_data(lock,args.task,['train','val']);protocol=json.loads((ROOT/'protocol.json').read_text());device=torch.device(args.device)
    train=load_npz(ROOT/'data'/args.task/'train.npz');val=load_npz(ROOT/'data'/args.task/'val.npz');norm=fit_normalizer(train,args.task)
    for seed in args.seeds:
        for arm in args.arms:
            result=train_one(args.task,args.backbone,arm,seed,device,train,val,norm,protocol,lock)
            if result is None:return
def evaluate(args):
    lock=verify_lock();verify_data(lock,args.task,['test']);lockfile=ROOT/'checkpoint_lock.json';assert lockfile.exists(),'Final checkpoints must be locked before test'
    cl=json.loads(lockfile.read_text());assert cl['code_lock_sha256']==sha(ROOT/'code_lock.json');device=torch.device(args.device)
    protocol=json.loads((ROOT/'protocol.json').read_text());test=load_npz(ROOT/'data'/args.task/'test.npz')
    for seed in args.seeds:
        for arm in args.arms:
            out=ROOT/'runs'/args.task/args.backbone/f'seed_{seed}'/arm;cp=out/'final.pt';rel=str(cp.relative_to(ROOT)).replace('\\','/')
            if not cp.exists():continue
            assert cl['checkpoints'][rel]==sha(cp)
            if (out/'test_results.json').exists():
                previous=json.loads((out/'test_results.json').read_text());assert previous['checkpoint_sha256']==sha(cp);assert previous['code_lock_sha256']==sha(ROOT/'code_lock.json')
                for view in protocol['views'][args.task]:assert sha(out/f'test_{view}.npz')==previous['views'][view]['predictions_sha256']
                continue
            saved=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(args.backbone,test['base__C0'].shape[-1],max_len=20,hidden=64).to(device);model.load_state_dict(saved['model']);result={}
            for view in protocol['views'][args.task]:
                td=tensor_data(test,view+'__'+arm,saved['normalization'],device);p=predict(model,td);save_predictions(out/f'test_{view}.npz',test,p)
                vm=metrics(test['y'],p,get_regions(test));vm['operating']=operating(test['y'],p,saved['thresholds']);vm['predictions_sha256']=sha(out/f'test_{view}.npz');result[view]=vm
            write_json(out/'test_results.json',{'task':args.task,'backbone':args.backbone,'arm':arm,'seed':seed,'checkpoint_sha256':sha(cp),'code_lock_sha256':sha(ROOT/'code_lock.json'),'data_manifest_sha256':sha(ROOT/'data_manifest.json'),'status':'developmental_previously_exposed_test','views':result})
            print('TEST_COMPLETE',args.task,args.backbone,arm,seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','test'],required=True);p.add_argument('--task',choices=['individual','relation'],required=True);p.add_argument('--backbone',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--arms',nargs='+',default=['C0','C1','C2','C3','C4']);args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    (train if args.mode=='train' else evaluate)(args)
