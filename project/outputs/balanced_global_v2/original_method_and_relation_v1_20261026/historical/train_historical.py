"""Actual Table-1 historical architectures on an explicit repaired-input bridge.

Read train/val only. Fixed epoch25 is primary; validation-best is secondary.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,copy,csv,hashlib,json,random,sys,time
import numpy as np
import torch
from torch.nn import functional as F
from historical_model import HistoricalModel,ARMS,training_loss

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import pilot_metrics_frozen as metrics_lib
PROTOCOL_SHA='1147e459e9b8617b30b793e571f16631376627ed5eb6d3e0e3584305ecf4d6a6'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
SPLITS=('train','val')
META=('window_id','base_window_id','mmsi','region','observation_noise','noise_type','noise_scope','target_noise_effective','neighbor_noise_effective','observation_noise_effective')
INPUT_KEYS=tuple(dict.fromkeys(a['key'] for a in ARMS.values()))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def jwrite(path,x):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def state_sha(state):
    h=hashlib.sha256()
    for k in sorted(state):
        x=state[k].detach().cpu().contiguous().numpy()
        h.update(k.encode()+b'\0');h.update(str(x.dtype).encode()+b'\0');h.update(canon(list(x.shape)).encode()+b'\0');h.update(x.tobytes())
    return h.hexdigest()
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def read_protocol(path):
    if sha(path)!=PROTOCOL_SHA:raise ValueError('Frozen protocol hash mismatch')
    p=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if p['arms']!=list(ARMS) or p['regions']!=list(REGIONS) or p['splits_read']!=list(SPLITS):raise ValueError('Unexpected matrix')
    for arm,cfg in ARMS.items():
        s=p['arm_specs'][arm]
        if (s['input'],s['graph'],s['augmentation_bundle'],s['padding'])!=(cfg['key'],cfg['graph'],cfg['recipe'],'masked' if cfg['masked'] else 'compat'):raise ValueError('Arm formula changed')
    if p['training']['epochs']!=25 or p['training']['batch_size']!=128:raise ValueError('Historical budget changed')
    return p
def ids_sha(d):
    return hashlib.sha256(canon([[str(d[k][i]) for k in ['window_id','base_window_id','mmsi','region']]+[int(d['y'][i]),int(d['label_class'][i])] for i in range(len(d['y']))]).encode()).hexdigest()


def load_data(root,regions=REGIONS,expected=None):
    if expected is None:expected={'train':980,'val':210}
    parts={s:[] for s in SPLITS};proof=[];rows=[]
    for rg in regions:
        for split in SPLITS:
            path=Path(root)/rg/f'{split}.npz';before=sha(path)
            with np.load(path,allow_pickle=False) as z:
                required=set(INPUT_KEYS)|set(META)|{'mask','y','label_class','ep','rf','nb_seq','nb_xy','nb_w','nb_mask'}
                if required-set(z.files):raise ValueError(f'Missing fields {path}: {sorted(required-set(z.files))}')
                mask=metrics_lib.validate_mask(z['mask']);n,T=mask.shape
                if T!=20 or (expected and n!=expected[split]):raise ValueError('Unexpected source dimensions')
                y=z['y'].astype(np.int64);yc=z['label_class'].astype(np.int64)
                if y.shape!=(n,) or yc.shape!=(n,) or not np.isin(y,[0,1]).all() or not np.isin(yc,range(6)).all() or not np.array_equal(y,yc>0):raise ValueError('Label semantics mismatch')
                d={'y':y,'label_class':yc,'mask':mask,'features':{}}
                for key in META:
                    if z[key].shape!=(n,):raise ValueError('Metadata dimension mismatch')
                    d[key]=z[key].astype(bool) if key in ['observation_noise','target_noise_effective','neighbor_noise_effective','observation_noise_effective'] else z[key].astype(str)
                if not np.all(d['region']==rg):raise ValueError('Wrong region')
                for key in INPUT_KEYS:
                    width=next(c['nf'] for c in ARMS.values() if c['key']==key)
                    x=z[key].astype(np.float32)
                    if x.shape!=(n,20,width) or not np.isfinite(x).all():raise ValueError(f'Bad feature {key}')
                    d['features'][key]=x
                for key,shape in [('ep',(n,20,4)),('rf',(n,20)),('nb_seq',(n,32,5,4)),('nb_xy',(n,32,2)),('nb_w',(n,32)),('nb_mask',(n,32))]:
                    x=z[key].astype(bool if key=='nb_mask' else np.float32)
                    if x.shape!=shape or not np.isfinite(x).all():raise ValueError(f'Bad field {key}')
                    d[key]=x
            if sha(path)!=before:raise ValueError('Input changed during read')
            parts[split].append(d);proof.append(dict(region=rg,split=split,path=str(path.resolve()),sha256=before,windows=n,valid_steps=int(mask.sum()),clean_windows=int((~d['observation_noise']).sum())))
            for i in range(n):rows.append(dict(split=split,source_row=i,source_path=str(path.resolve()),source_sha256=before,**{k:str(d[k][i]) for k in META},y=int(y[i]),label_class=int(yc[i]),gradient_eligible=bool(split=='train' and not d['observation_noise'][i])))
    output={}
    keys=('y','label_class','mask','ep','rf','nb_seq','nb_xy','nb_w','nb_mask')+META
    for split,pp in parts.items():
        output[split]={k:np.concatenate([d[k] for d in pp]) for k in keys}
        output[split]['features']={k:np.concatenate([d['features'][k] for d in pp]) for k in INPUT_KEYS}
        for k in ['window_id','base_window_id']:
            if len(set(output[split][k]))!=len(output[split][k]):raise ValueError('Duplicate window')
    for k in ['window_id','base_window_id','mmsi']:
        if set(output['train'][k])&set(output['val'][k]):raise ValueError(f'{k} crosses splits')
    return output,proof,rows


def fit_normalizer(x,mask):
    # Match historical NumPy float32 mean/std + 1e-6 on all train valid steps.
    v=x[mask]
    return dict(mean=v.mean(0,keepdims=True),std=v.std(0,keepdims=True)+np.float32(1e-6),valid_steps=int(len(v)))
def normalized(x,mask,stats,masked):
    out=((x-stats['mean'])/stats['std']).astype(np.float32)
    if masked:out[~mask]=0
    if not np.isfinite(out).all():raise ValueError('Nonfinite normalized input')
    return out
def make_orders(clean_indices,seed,epochs):
    rng=np.random.RandomState(seed);current=np.array(clean_indices,np.int64,copy=True);orders=[];digests=[];h=hashlib.sha256()
    for ep in range(1,epochs+1):
        rng.shuffle(current);order=current.copy();orders.append(order)
        payload=np.asarray([ep],dtype='<i8').tobytes()+order.astype('<i8').tobytes();digests.append(hashlib.sha256(payload).hexdigest());h.update(payload)
    return orders,h.hexdigest(),digests
def to_tensors(d,x):
    return dict(x=torch.from_numpy(x),mask=torch.from_numpy(d['mask']),y=torch.from_numpy(d['y'].astype(np.float32)),label_class=torch.from_numpy(d['label_class']),
                **{k:torch.from_numpy(d[k]) for k in ['ep','rf','nb_seq','nb_xy','nb_w','nb_mask']})
def batch(d,indices,device):return {k:v[indices].to(device) for k,v in d.items()}

@torch.inference_mode()
def predict(model,d,device,batch_size=128):
    model.eval();prob=[];types=[]
    for lo in range(0,len(d['y']),batch_size):
        b=batch(d,slice(lo,lo+batch_size),device)
        pb,pc,_,_=model(b['x'],b['mask'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'])
        prob.append(torch.sigmoid(pb).cpu().numpy());types.append(torch.softmax(pc,-1).cpu().numpy())
    return np.concatenate(prob),np.concatenate(types)
def scores_metrics(d,scores,threshold):return metrics_lib.region_metrics(d['y'],scores,d['region'],threshold)
def prediction_rows(d,scores,threshold,kind,arm,seed,which):
    rows=[]
    for i,s in enumerate(scores):
        row=dict(backbone=kind,arm=arm,seed=seed,checkpoint=which,y=int(d['y'][i]),label_class=int(d['label_class'][i]),score=float(s),threshold=threshold,prediction=int(s>=threshold))
        row.update({k:str(d[k][i]) for k in META});rows.append(row)
    return rows
def subset_metrics(d,scores,threshold):
    out={}
    for label,mask in [('clean',~d['observation_noise']),('requested_corruption',d['observation_noise'])]:
        out[label]=metrics_lib.region_metrics(d['y'][mask],scores[mask],d['region'][mask],threshold)
    return out


def train_one(kind,arm,seed,data,stats,training,device,out,protocol_hash,config_hash):
    cfg=ARMS[arm];seed_all(seed);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    clean=np.flatnonzero(~data['train']['observation_noise'])
    if len(clean)<2:raise ValueError('No clean training examples')
    cnt=np.bincount(data['train']['label_class'][clean][data['train']['label_class'][clean]>0]-1,minlength=5)
    cw=torch.tensor(1./np.maximum(cnt,1),dtype=torch.float32,device=device)
    spec=dict(backbone=kind,arm=arm,seed=seed,protocol_sha256=protocol_hash,configuration_sha256=config_hash)
    jwrite(out/'run_configuration.json',dict(run_spec=spec,arm_spec=cfg,training=training,gradient_clean_indices_sha256=hashlib.sha256(clean.astype('<i8').tobytes()).hexdigest(),clean_class_counts=cnt.tolist()))
    np.savez_compressed(out/'normalizer.npz',**stats)
    tensors={s:to_tensors(d,normalized(d['features'][cfg['key']],d['mask'],stats,cfg['masked'])) for s,d in data.items()}
    model=HistoricalModel(kind,arm).to(device)
    initial=dict(full=state_sha(model.state_dict()),encoder=state_sha(model.core.enc.state_dict()),binary_head=state_sha(model.bin.state_dict()),type_head=state_sha(model.cls.state_dict()))
    params=sum(p.numel() for p in model.parameters())
    if cfg['graph'] and params!={'tcn':71125,'gru':65237}[kind]:raise AssertionError('Historical full parameter count changed')
    optimizer=torch.optim.AdamW(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    orders,order_hash,each_order=make_orders(clean,seed,training['epochs'])
    history=[];all_predictions=[];best=None;best_ap=-float('inf');best_epoch=None;start=time.time()
    for ep,order in enumerate(orders,1):
        model.train();totals={k:0. for k in ['binary','type_ce','js','aug_binary','hidden_smooth','epsilon_mean','total']}
        for lo in range(0,len(order),training['batch_size']):
            ii=order[lo:lo+training['batch_size']];b=batch(tensors['train'],ii,device)
            optimizer.zero_grad(set_to_none=True);loss,logs=training_loss(model,b,cw)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();optimizer.step()
            for k,v in logs.items():totals[k]+=v*len(ii)
        prob,_=predict(model,tensors['val'],device,training['batch_size'])
        m=scores_metrics(data['val'],prob,.5);ap=m['macro']['AP']
        history.append(dict(epoch=ep,validation_macro_AP=ap,**{f'train_{k}':v/len(order) for k,v in totals.items()}));all_predictions.append(prob.copy())
        if ap>best_ap:best_ap=ap;best_epoch=ep;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"EPOCH {kind}/{arm}/seed{seed} {ep}/{training['epochs']} train={history[-1]['train_total']:.6f} valAP={ap:.6f}",flush=True)
    last={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};last_prob,last_types=predict(model,tensors['val'],device,training['batch_size'])
    if not np.array_equal(last_prob,all_predictions[-1]):raise AssertionError('Final replay mismatch')
    model.load_state_dict(best);best_prob,best_types=predict(model,tensors['val'],device,training['batch_size'])
    if not np.array_equal(best_prob,all_predictions[best_epoch-1]):raise AssertionError('Best replay mismatch')
    th_last,_=metrics_lib.choose_threshold(data['val']['y'],last_prob);th_best,_=metrics_lib.choose_threshold(data['val']['y'],best_prob)
    final_m=scores_metrics(data['val'],last_prob,th_last);best_m=scores_metrics(data['val'],best_prob,th_best)
    for which,prob,th in [('final',last_prob,th_last),('best',best_prob,th_best)]:
        rows=prediction_rows(data['val'],prob,th,kind,arm,seed,which);metrics_lib.write_csv(out/f'val_predictions_{which}.csv',rows,list(rows[0]))
    np.savez_compressed(out/'epoch_validation_predictions.npz',scores=np.stack(all_predictions),window_id=data['val']['window_id'],y=data['val']['y'],label_class=data['val']['label_class'],region=data['val']['region'])
    np.savez_compressed(out/'type_predictions.npz',final=last_types,best=best_types,window_id=data['val']['window_id'],label_class=data['val']['label_class'])
    common=dict(backbone=kind,arm=arm,arm_config=cfg,normalizer_mean=torch.from_numpy(stats['mean']),normalizer_std=torch.from_numpy(stats['std']),protocol_sha256=protocol_hash,configuration_sha256=config_hash)
    torch.save(dict(state_dict=last,epoch=training['epochs'],threshold=th_last,**common),out/'checkpoint_final.pt')
    torch.save(dict(state_dict=best,epoch=best_epoch,threshold=th_best,**common),out/'checkpoint_best.pt')
    jwrite(out/'history.json',history)
    result=dict(status='completed_validation_only',run_spec=spec,**spec,parameters=params,device=str(device),epochs=training['epochs'],primary_epoch=training['epochs'],best_epoch=best_epoch,
                final_validation=final_m,best_validation=best_m,final_threshold=th_last,best_threshold=th_best,
                final_strata=subset_metrics(data['val'],last_prob,th_last),best_strata=subset_metrics(data['val'],best_prob,th_best),
                gradient_training_windows=len(clean),all_training_windows=len(data['train']['y']),normalizer_valid_steps=stats['valid_steps'],positive_type_counts=cnt.tolist(),
                initial_state_hashes=initial,final_state_sha256=state_sha(last),best_state_sha256=state_sha(best),batch_order_sha256=order_hash,epoch_batch_order_sha256=each_order,
                history=history,elapsed_seconds=time.time()-start,
                definition='Historical35 execution model/training transferred to corrected data; fixed25 primary, best-validation secondary; no held-out test use')
    result['output_hashes']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    jwrite(out/'result.json',result);jwrite(out/'completion.json',dict(status='complete',result_sha256=sha(out/'result.json')))
    print('RUN_COMPLETE',kind,seed,arm,'fixed',final_m['macro']['AP'],'best',best_m['macro']['AP'],flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--backbone',choices=['tcn','gru'],required=True);ap.add_argument('--device',default='cpu');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    p=read_protocol(args.protocol);torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise ValueError('Requested CUDA unavailable; no silent device fallback')
    data,proof,rows=load_data(args.data);stats={key:fit_normalizer(data['train']['features'][key],data['train']['mask']) for key in INPUT_KEYS}
    code_files=[Path(__file__),HERE/'historical_model.py',HERE/'vendor/35_graph_ours_final.py',HERE/'vendor/34_fair_benchmark.py',HERE/'vendor/pilot_metrics_frozen.py']
    config=dict(protocol_sha256=sha(args.protocol),backbone=args.backbone,data_path=str(args.data.resolve()),sources=proof,input_signatures={s:ids_sha(d) for s,d in data.items()},
                training=p['training'],arms=p['arms'],device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',cpu_threads=4,
                deterministic=True,tf32=False,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),code_sha256={str(f.relative_to(HERE)):sha(f) for f in code_files},
                counts={s:dict(windows=len(d['y']),clean=int((~d['observation_noise']).sum()),valid_steps=int(d['mask'].sum())) for s,d in data.items()},
                versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,sklearn=__import__('sklearn').__version__))
    worker=args.out/args.backbone
    if worker.exists():
        if not args.resume or json.loads((worker/'configuration.json').read_text())!=config:raise ValueError('Output exists or resume configuration changed')
    else:
        worker.mkdir(parents=True,exist_ok=False);jwrite(worker/'configuration.json',config);metrics_lib.write_csv(worker/'input_manifest.csv',rows,list(rows[0]))
    config_sha=sha(worker/'configuration.json')
    if args.prepare_only:
        jwrite(worker/'preflight.json',dict(status='passed',counts=config['counts'],feature_stats={k:dict(mean=v['mean'].tolist(),std=v['std'].tolist(),valid_steps=v['valid_steps']) for k,v in stats.items()}));print('PREPARED_ONLY',args.backbone,flush=True);return
    results=[]
    for seed in p['training']['seeds']:
        for arm in p['arms']:
            dest=worker/f'seed_{seed}'/arm
            if dest.exists():
                if not args.resume:raise ValueError('Run exists')
                result=json.loads((dest/'result.json').read_text());done=json.loads((dest/'completion.json').read_text())
                if done['result_sha256']!=sha(dest/'result.json') or result['configuration_sha256']!=config_sha:raise ValueError('Resume result mismatch')
                for name,h in result['output_hashes'].items():
                    if sha(dest/name)!=h:raise ValueError('Resume payload mismatch')
            else:result=train_one(args.backbone,arm,seed,data,stats[ARMS[arm]['key']],p['training'],device,dest,sha(args.protocol),config_sha)
            results.append(result)
    for seed in p['training']['seeds']:
        rr=[r for r in results if r['seed']==seed]
        if len({r['batch_order_sha256'] for r in rr})!=1:raise AssertionError('Unpaired sample orders')
    for source in proof:
        if sha(source['path'])!=source['sha256']:raise ValueError('Input source changed')
    summary=dict(status='completed_validation_only',backbone=args.backbone,runs=len(results),source_data_unchanged=True,protocol_sha256=sha(args.protocol),
                 results=[{k:r[k] for k in ['arm','seed','parameters','primary_epoch','best_epoch','final_validation','best_validation','elapsed_seconds']} for r in results])
    out=worker/'worker_summary.json'
    if out.exists():
        if json.loads(out.read_text())!=summary:raise ValueError('Changed existing summary')
    else:jwrite(out,summary)
    print('BACKBONE_COMPLETE',args.backbone,len(results),flush=True)
if __name__=='__main__':main()
