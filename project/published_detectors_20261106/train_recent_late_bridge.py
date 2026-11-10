"""Recent method + historical late field adapter and physical-state bundle.

Exploratory bridge; not a claim of exact reproduction of the original paper or
of the historical TCN interface.  Upstream models lack mask support internally.
Train/validation do not load the heldout test set; test runs separately.
"""
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
sys.path.insert(0,str(ROOT/'formal_20261101/single'))
import train_single as old
from recent_late_bridge import RecentLateBridge
from recent_late_provenance import source_files,sha

def fingerprint(d):
    h=hashlib.sha256()
    for k in ('ids','mmsi','region','y','yc','mask','times','raw','self','f0','f1','rho','state0','state1'):
        a=np.ascontiguousarray(d[k]);h.update(k.encode());h.update(a.tobytes())
    return h.hexdigest()

def inputs(d,n,arm):
    key='raw' if arm=='raw' else 'self'; fkey='f0' if arm in ('raw','causal_bundle','f0_bundle') else 'f1'
    target=((d[key]-n['sm'])/n['ss']).astype(np.float32)
    field=((d[fkey]-n['fm'])/n['fs']).astype(np.float32)
    target[~d['mask']]=0.;field[~d['valid']]=0.
    return dict(x=target,field=field,mask=d['mask'],rho=d['rho'].astype(np.float32),
                state=d['state0' if fkey=='f0' else 'state1'].astype(np.float32),
                y=d['y'].astype(np.float32),yc=d['yc'].astype(np.int64))

def previous_windows(d):
    # Strictly earlier same-target window, in observed AIS absolute time.
    core=ROOT/'framework_validation_v1_20261028/data/individual/train.npz'
    with np.load(core,allow_pickle=False) as z:
        if not np.array_equal(d['ids'],z['window_id'].astype(str)):raise ValueError('source order mismatch')
        ts=z['time_s']; mask=z['mask']
    ends=ts[np.arange(len(ts)),mask.sum(1)-1]; ships=d['mmsi']; previous=np.full(len(ends),-1,dtype=np.int64)
    rng=np.random.default_rng(783)
    for ship in np.unique(ships):
        ix=np.flatnonzero(ships==ship);ix=ix[np.argsort(ends[ix],kind='stable')]
        for j in range(1,len(ix)):
            cand=ix[max(0,j-10):j];cand=cand[ends[cand]<ends[ix[j]]]
            if len(cand):previous[ix[j]]=rng.choice(cand)
    return previous

@torch.no_grad()
def embeddings(model,x,device,batch):
    model.eval();out=[]
    for i in range(0,len(x),batch):out.append(model.adapter.embed(torch.as_tensor(x[i:i+batch],device=device)).cpu())
    return torch.cat(out)

def neighbors(z,k=5):
    near=[];far=[]
    for st in range(0,len(z),256):
        dist=torch.cdist(z[st:st+256],z);rows=torch.arange(len(dist));dist[rows,rows+st]=float('inf')
        near.append(torch.topk(dist,k,largest=False).indices.numpy());dist[rows,rows+st]=float('-inf')
        far.append(torch.topk(dist,k,largest=True).indices.numpy())
    return np.concatenate(near),np.concatenate(far)

def metric(y,p,regions,thresholds=None):
    ret={'pooled_AP':float(average_precision_score(y,p)),'AUROC':float(roc_auc_score(y,p)),
         'macro_AP':float(np.mean([average_precision_score(y[regions==r],p[regions==r]) for r in np.unique(regions) if len(np.unique(y[regions==r]))==2]))}
    if thresholds:
        for name,t in thresholds.items():
            ret['FAR@'+name]=float(np.mean(p[y==0]>=t));ret['recall@'+name]=float(np.mean(p[y==1]>=t))
    return ret

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['tft_gcn','simad','dmrad','carla'])
    p.add_argument('--arm',required=True,choices=['raw','causal_bundle','f0_bundle','f1_bundle'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int,default=25);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.002);p.add_argument('--native-aux-weight',type=float,default=.1)
    p.add_argument('--mode',choices=['train','test'],default='train');p.add_argument('--smoke',action='store_true')
    p.add_argument('--checkpoint',choices=['best_val','last'],default='best_val')
    p.add_argument('--manifest',default=None,help='Frozen 48-run manifest required for actual test')
    p.add_argument('--bridge-gate',default='/home/ps/AIS_published_detectors_20261107/recent_original_late/bridge_gate_lock.json')
    p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/recent_original_late')
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    gate_sha=None
    if not a.smoke:
        gate=Path(a.bridge_gate);record=json.loads(gate.read_text(encoding='utf8'))
        for path,digest in record['evidence_sha256'].items():
            if sha(path)!=digest:raise ValueError('bridge evidence changed: '+path)
        if record['tcn_frozen_runs']!=21 or record['tft_frozen_runs']!=24 or record['tft_evaluation_batch']!=128:raise ValueError('bridge gate mismatch')
        gate_sha=sha(gate)
    torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    tr=old.load('train');va=old.load('val');norm=old.fit(tr);train_hash=fingerprint(tr);val_hash=fingerprint(va)
    if set(tr['mmsi']) & set(va['mmsi']):raise ValueError('MMSI train/val overlap')
    X=inputs(tr,norm,a.arm);V=inputs(va,norm,a.arm)
    if a.smoke:
        X={k:v[:64] for k,v in X.items()};V={k:v[:64] for k,v in V.items()}
        va={k:v[:64] for k,v in va.items()};epochs=2
    else:epochs=a.epochs
    net=RecentLateBridge(a.method,gpu=a.gpu).to(dev);use_field=a.arm in ('f0_bundle','f1_bundle')
    out=Path(a.out)/('smoke' if a.smoke else 'runs')/a.method/a.arm/f'seed{a.seed}'
    @torch.no_grad()
    def predict(d):
        net.eval();s=[]
        for st in range(0,len(d['y']),a.batch_size):
            sl=slice(st,st+a.batch_size)
            x=torch.as_tensor(d['x'][sl],device=dev);m=torch.as_tensor(d['mask'][sl],device=dev)
            f=torch.as_tensor(d['field'][sl],device=dev);rho=torch.as_tensor(d['rho'][sl],device=dev)
            s.extend(torch.sigmoid(net(x,m,f,rho,use_field)[0]).cpu().numpy())
        return np.asarray(s)
    if a.mode=='test':
        ckpath=out/(a.checkpoint+'.pt')
        if not a.smoke:
            if not a.manifest:raise ValueError('Actual test requires the frozen manifest')
            manifest=json.loads(Path(a.manifest).read_text(encoding='utf8'))
            key=f'{a.method}/{a.arm}/seed{a.seed}'
            if manifest['runs'][key][a.checkpoint]['sha256']!=sha(ckpath):raise ValueError('frozen checkpoint hash mismatch')
            if manifest['sources'][a.method]!=source_files(a.method,include_test=True):raise ValueError('frozen code/data hash mismatch')
        ck=torch.load(ckpath,map_location=dev,weights_only=False)
        if ck['train_hash']!=train_hash or ck['val_hash']!=val_hash:raise ValueError('frozen source data mismatch')
        if ck['spec']['batch_size']!=a.batch_size:raise ValueError('evaluation batch must match frozen validation batch')
        net.load_state_dict(ck['model']);pv=predict(V);th={f'{q}%':float(np.quantile(pv[V['y']==0],1-q/100,method='higher')) for q in (1,5)}
        if a.smoke:
            # Exercise reload, score persistence and metric path using validation
            # only. Never inspect the real heldout test during implementation.
            te=va;T=V;prefix='smoke_validation_as_test_'+a.checkpoint
        else:
            # Match the original frozen formal evaluation: test rho belongs to
            # eight_backbone_preparation, not the train/val rho directory.
            # The manifest already locks IDs, masks, member provenance and SHA.
            import evaluate_fixed
            _,te=evaluate_fixed.load_test()
            if set(te['mmsi'])&(set(tr['mmsi'])|set(va['mmsi'])):raise ValueError('MMSI test overlap')
            T=inputs(te,norm,a.arm);prefix='test_'+a.checkpoint
            if (out/(prefix+'_scores.npz')).exists() or (out/(prefix+'_metrics.json')).exists():raise FileExistsError('test output exists; refusing overwrite')
        pt=predict(T);result={'selection':a.checkpoint,'thresholds':th,'validation':metric(V['y'],pv,va['region'],th),prefix:metric(T['y'],pt,te['region'],th),'epoch':ck['epoch'],'phase':ck['phase'],'checkpoint_sha256':sha(ckpath)}
        np.savez_compressed(out/(prefix+'_scores.npz'),score=pt,y=te['y'],mmsi=te['mmsi'],region=te['region'],window_id=te['ids'])
        (out/(prefix+'_metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(prefix.upper(),json.dumps(result),flush=True);return
    out.mkdir(parents=True,exist_ok=a.smoke)
    if (out/'best_val.pt').exists() and not a.smoke:raise FileExistsError(out)
    init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in net.parameters())).hexdigest()
    ncat=np.bincount(X['yc'][X['yc']>0]-1,minlength=5);cw=torch.tensor(1./np.maximum(ncat,1),dtype=torch.float32,device=dev)
    sequence_note=net.sequence_provenance
    sources=source_files(a.method,include_test=False)
    spec=vars(a)|{'train_count':len(X['y']),'validation_count':len(V['y']),'target':'historical raw7 or causal gated7','field':'historical F0/F1 four-dimensional feature, weighted by frozen rho',
     'rho':'historical min(n/4,1) times average member reliability; unavailable=0', 'loss':'raw BCE; other arms unchanged historical bundle; every arm additionally has native auxiliary weight',
     'sequence_proxy':sequence_note,'important_mask':'historical prefix masks are retained in side fusion and bundle; published backbone and native auxiliary loss observe zero-padded steps and do not explicitly consume irregular timestamps',
     'time_policy':'original sample order; 20 slots, variable prefix mask and original AIS times; no resampling',
     'validation':'macro AP on previous development validation; best_val and last','test':'previously exposed; separately evaluated','init_sha256':init,
     'train_sha256':train_hash,'val_sha256':val_hash,'sources':sources,'bridge_gate_sha256':gate_sha,'carla_phases':'12 pretext + 13 clustering when 25 epochs, both BCE/bundle and native stage objective; otherwise 25 epochs if auxiliary weight is zero'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    history=[];best=-1.;opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.01)
    prev=previous_windows(tr)[:len(X['y'])] if a.method=='carla' and a.native_aux_weight>0 else None
    if prev is not None:prev[prev>=len(X['y'])]=-1
    bundle=a.arm!='raw'
    def pack(d,ids):return {k:torch.as_tensor(v[ids],device=dev) for k,v in d.items()}
    def base_loss(b):
        if bundle:return net.bundle_loss(b['x'],b['mask'],b['field'],b['rho'],use_field,b['state'],b['y'],b['yc'],cw)[0]
        return F.binary_cross_entropy_with_logits(net(b['x'],b['mask'],b['field'],b['rho'],False)[0],b['y'])
    def save(epoch,phase,losses):
        nonlocal best
        val=predict(V);scores=metric(V['y'],val,va['region']);row={'epoch':epoch+1,'phase':phase,'loss':float(np.mean(losses)),**scores};history.append(row)
        ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'phase':phase,'spec':spec,'metrics':scores,'train_hash':train_hash,'val_hash':val_hash}
        if a.method=='carla':ck.update(carla_previous_loss=net.adapter.prev_loss,carla_margin=net.adapter.pretext_loss.margin)
        torch.save(ck,out/'last.pt')
        if scores['macro_AP']>best:
            best=scores['macro_AP'];torch.save(ck,out/'best_val.pt');np.savez_compressed(out/'val_scores.npz',score=val,y=V['y'],mmsi=va['mmsi'],region=va['region'],window_id=va['ids'])
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('VALIDATION',json.dumps(row),flush=True)
    def normal_epoch(epoch):
        if hasattr(net.adapter,'begin_epoch'):net.adapter.begin_epoch(epoch,epochs)
        net.train();order=np.random.default_rng(300000+epoch).permutation(len(X['y']));losses=[]
        for st in tqdm(range(0,len(order),a.batch_size),desc=f'{a.method}/{a.arm} {epoch+1}/{epochs}',file=sys.stdout,dynamic_ncols=True):
            ids=order[st:st+a.batch_size];b=pack(X,ids);opt.zero_grad(set_to_none=True);loss=base_loss(b)
            if a.native_aux_weight>0:
                if a.method=='carla':
                    positive=b['x']+torch.randn_like(b['x'])*.01;has=prev[ids]>=0
                    # A same-ship prior window may have a different valid length.
                    if has.any():
                        ix=prev[ids[has]];px=torch.as_tensor(X['x'][ix],device=dev);pm=torch.as_tensor(X['mask'][ix],device=dev)
                        positive[torch.as_tensor(has,device=dev)]=px.masked_fill(~pm[...,None],0.)
                    fallback=torch.as_tensor(~has,device=dev)
                    positive[fallback]=positive[fallback].masked_fill(~b['mask'][fallback,...,None],0.)
                    negative=net.adapter.synthetic(b['x']).masked_fill(~b['mask'][...,None],0.)
                    aux=net.native_loss(b['x'],b['mask'],indices=ids,epoch=epoch,positive=positive,negative=negative)
                else:aux=net.native_loss(b['x'],b['mask'],indices=ids,epoch=epoch)
                loss=loss+a.native_aux_weight*aux
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
        return losses
    pre=epochs if a.method!='carla' or a.native_aux_weight==0 else max(1,epochs//2)
    for epoch in range(pre):save(epoch,'pretext' if a.method=='carla' else 'joint',normal_epoch(epoch))
    if a.method=='carla' and a.native_aux_weight>0 and pre<epochs:
        # Preserve native two-phase prior adaptation: best-pretext snapshot,
        # synthetic negatives and fixed nearest/farthest pretext embeddings.
        ck=torch.load(out/'best_val.pt',map_location=dev,weights_only=False);net.load_state_dict(ck['model'])
        net.adapter.prev_loss=ck['carla_previous_loss'];net.adapter.pretext_loss.margin=ck['carla_margin']
        augmented=[]
        for st in range(0,len(X['x']),a.batch_size):
            xx=torch.as_tensor(X['x'][st:st+a.batch_size],device=dev)
            mm=torch.as_tensor(X['mask'][st:st+a.batch_size],device=dev)
            augmented.append(net.adapter.synthetic(xx).masked_fill(~mm[...,None],0.).cpu().numpy())
        cluster_x=np.concatenate([X['x'],*augmented]);z=embeddings(net,cluster_x,dev,a.batch_size);near,far=neighbors(z)
        net.adapter.begin_clustering();opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=.01)
        for epoch in range(pre,epochs):
            net.train();order=np.random.default_rng(300000+epoch).permutation(len(cluster_x));losses=[]
            for st in tqdm(range(0,len(order),a.batch_size),desc=f'CARLA/{a.arm} cluster {epoch+1}/{epochs}',file=sys.stdout,dynamic_ncols=True):
                ids=order[st:st+a.batch_size];n=near[ids,np.random.randint(near.shape[1],size=len(ids))];f=far[ids,np.random.randint(far.shape[1],size=len(ids))]
                x=torch.as_tensor(cluster_x[ids],device=dev);nx=torch.as_tensor(cluster_x[n],device=dev);fx=torch.as_tensor(cluster_x[f],device=dev)
                # Synthetic perturbations are defined on the same shape; masks
                # refer to their original source window (ids modulo N).
                m=torch.as_tensor(X['mask'][ids%len(X['y'])],device=dev)
                nm=torch.as_tensor(X['mask'][n%len(X['y'])],device=dev);fm=torch.as_tensor(X['mask'][f%len(X['y'])],device=dev)
                nx=nx.masked_fill(~nm[...,None],0.);fx=fx.masked_fill(~fm[...,None],0.)
                opt.zero_grad(set_to_none=True);loss=a.native_aux_weight*net.native_loss(x,m,near=nx,far=fx)
                original=ids<len(X['y'])
                if original.any():loss=loss+base_loss(pack(X,ids[original]))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            save(epoch,'clustering',losses)
if __name__=='__main__':main()
