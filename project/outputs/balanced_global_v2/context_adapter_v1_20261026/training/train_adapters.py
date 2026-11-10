"""Frozen baseline plus zero-output context adapter; train/validation only."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse, copy, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'vendor'))
import pilot_train_frozen as base

ARMS = ('self_quality', 'mean_context', 'affine_context', 'reliable_context')
ARM_SOURCES = dict(zip(ARMS, ('raw_meta', 'mean_meta', 'affine_meta', 'reliable_raw')))
REGIONS = ('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
SPLITS = ('train', 'val')
META = ('mmsi','window_id','base_window_id','region','behavior_type','noise_type','noise_scope',
        'target_noise_effective','neighbor_noise_effective','observation_noise_effective')
PROTOCOL_SHA = '7c52304e5517da6c749e6a2f025ee61d77f8739671931fdd8bb78086b31c80b0'
PARENT_PROTOCOL_SHA = 'fd882a0b27f51c8fbb76c327353379dd5d35caeb642fee42d54a383a3909d489'
VENDOR_SHA = '4ee25368063f92a2be6aa5991fb228c2771ca4e981123deb13e763cc732f5281'


def canon(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def write_json(path, value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def digest_state(state):
    h=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous().numpy()
        for v in (name.encode(),str(x.dtype).encode(),canon(list(x.shape)).encode()): h.update(v+b'\0')
        h.update(x.tobytes())
    return h.hexdigest()


def digest_ids(d):
    rows=[[str(d[k][i]) for k in ('window_id','base_window_id','mmsi','region')]+[int(d['y'][i])] for i in range(len(d['y']))]
    return hashlib.sha256(canon(rows).encode()).hexdigest()


def make_orders(n,seed,epochs):
    rng=np.random.default_rng(seed+100)
    orders=[rng.permutation(n).astype('<i8') for _ in range(epochs)]
    h=hashlib.sha256();per=[]
    for e,order in enumerate(orders,1):
        payload=np.asarray([e],dtype='<i8').tobytes()+order.tobytes()
        h.update(payload);per.append(hashlib.sha256(payload).hexdigest())
    return orders,h.hexdigest(),per


def read_protocol(path):
    p=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if base.sha(path)!=PROTOCOL_SHA:raise ValueError('Frozen protocol changed')
    if p['arms']!=list(ARMS) or p['arm_sources']!=ARM_SOURCES:raise ValueError('Arms changed')
    if p['regions']!=list(REGIONS) or p['splits_read']!=list(SPLITS):raise ValueError('Data scope changed')
    if base.sha(HERE/'vendor/pilot_train_frozen.py')!=VENDOR_SHA:raise ValueError('Backbone definitions changed')
    return p


def construct_arms(z):
    """Only residual/derivative slots differ after common support is imposed."""
    mask=base.validate_mask(z['mask'])
    common=mask & z['context_valid'].astype(bool) & z['plain_affine_valid'].astype(bool) & z['plain_context_valid'].astype(bool)
    raw=z['raw16'].astype(np.float32)
    if raw.shape!=mask.shape+(16,) or np.any(raw[...,7:]!=0):raise ValueError('Parent raw16 is not raw7 plus zeros')
    if not np.isfinite(raw).all() or np.any(raw[~mask]!=0):raise ValueError('Invalid raw baseline values')
    inputs={}
    for arm,source in ARM_SOURCES.items():
        x=z[source].astype(np.float32).copy()
        if x.shape!=raw.shape or not np.isfinite(x).all() or np.any(x[~mask]!=0):raise ValueError('Bad adapter source')
        if not np.array_equal(x[...,:7],raw[...,:7]):raise ValueError('Arm has changed raw motion')
        x[...,12]=common;x[...,13]=common
        inputs[arm]=x
    ref=inputs[ARMS[0]]
    keep=[0,1,2,3,4,5,6,9,10,11,12,13]
    for x in inputs.values():
        if not np.array_equal(ref[...,keep],x[...,keep]):raise ValueError('Non-context information differs across arms')
    return raw,mask,common,inputs


def load_data(parent, regions=REGIONS, counts=None):
    if counts is None:counts={'train':980,'val':210}
    chunks={s:[] for s in SPLITS};sources=[];manifest=[]
    for region in regions:
        for split in SPLITS:
            path=Path(parent)/'data'/region/(split+'.npz')
            before=base.sha(path)
            with np.load(path,allow_pickle=False) as z:
                required={'raw16','mask','context_valid','plain_affine_valid','plain_context_valid','y'}|set(META)|set(ARM_SOURCES.values())
                if required-set(z.files):raise ValueError(f'Missing source fields {path}: {required-set(z.files)}')
                raw,mask,common,features=construct_arms(z)
                y=z['y'].astype(np.int64);n=len(y)
                if y.shape!=(n,) or n!=counts[split] or not np.isin(y,[0,1]).all() or len(np.unique(y))!=2:raise ValueError('Source labels/counts')
                part={'y':y,'raw':raw,'mask':mask,'common_support':common,'trusted_support':mask & z['context_valid'].astype(bool),'features':features}
                for key in META:
                    if z[key].shape!=(n,):raise ValueError('Source metadata shape')
                    part[key]=z[key].astype(str)
                if not np.all(part['region']==region):raise ValueError('Region mismatch')
                chunks[split].append(part)
                sources.append(dict(region=region,split=split,path=str(path.resolve()),sha256=before,samples=n,
                                    positive=int(y.sum()),valid_steps=int(mask.sum()),trusted_steps=int(part['trusted_support'].sum()),
                                    common_steps=int(common.sum()),zero_support_windows=int((common.sum(1)==0).sum())))
                for i in range(n):
                    manifest.append(dict(split=split,region=region,window_id=str(part['window_id'][i]),base_window_id=str(part['base_window_id'][i]),
                                         mmsi=str(part['mmsi'][i]),y=int(y[i]),source_row=i,source_path=str(path.resolve()),source_sha256=before))
            if base.sha(path)!=before:raise ValueError('Source changed while reading')
    data={}
    for split,parts in chunks.items():
        data[split]={k:np.concatenate([p[k] for p in parts]) for k in ('y','raw','mask','common_support','trusted_support')+META}
        data[split]['features']={a:np.concatenate([p['features'][a] for p in parts]) for a in ARMS}
        for key in ('window_id','base_window_id'):
            if len(set(data[split][key]))!=len(data[split][key]):raise ValueError('Repeated window')
    for key in ('mmsi','window_id','base_window_id'):
        if set(data['train'][key]) & set(data['val'][key]):raise ValueError(f'Cross-split {key}')
    return data,sources,manifest


def fit_adapter_scaler(x,support):
    values=np.asarray(x,np.float64)[support]
    if len(values)==0:raise ValueError('No supported training steps')
    return dict(mean=values.mean(0).astype(np.float32),std=np.maximum(values.std(0),1e-6).astype(np.float32),valid_steps=int(len(values)))


def adapter_normalized(x,support,scaler):
    y=((x-scaler['mean'])/scaler['std']).astype(np.float32)
    y[~support]=0
    if not np.isfinite(y).all():raise ValueError('Nonfinite normalized adapter input')
    return y


class ResidualAdapter(nn.Module):
    def __init__(self,input_channels=16,hidden=32,output_channels=64):
        super().__init__()
        self.first=nn.Linear(input_channels,hidden)
        self.last=nn.Linear(hidden,output_channels)
        nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,x,mask,support):
        per=self.last(F.gelu(self.first(x)))
        weight=(mask & support).unsqueeze(-1)
        return (per*weight).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1)


def load_parent(parent,backbone,seed,data,sources):
    parent=Path(parent)
    if base.sha(parent/'protocol.json')!=PARENT_PROTOCOL_SHA:raise ValueError('Parent protocol changed')
    folder=parent/'runs'/backbone/f'seed_{seed}'/'raw16'
    result=json.loads((folder/'result.json').read_text())
    completion=json.loads((folder/'completion.json').read_text())
    if completion['result_sha256']!=base.sha(folder/'result.json'):raise ValueError('Parent result mismatch')
    for name,h in result['output_hashes'].items():
        if base.sha(folder/name)!=h:raise ValueError(f'Parent artifact changed: {name}')
    conf_path=parent/'runs'/backbone/'configuration.json'
    conf=json.loads(conf_path.read_text())
    if base.sha(conf_path)!=result['configuration_sha256']:raise ValueError('Parent worker config mismatch')
    if {(x['region'],x['split']):x['sha256'] for x in conf['sources']}!={(x['region'],x['split']):x['sha256'] for x in sources}:raise ValueError('Parent trained on other data')
    for split in SPLITS:
        if result['input_signatures'][split]!=digest_ids(data[split]):raise ValueError('Parent sample order differs')
    checkpoint=folder/'checkpoint.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=base.BackboneClassifier(backbone,16,ck['hidden'],ck['dropout'])
    model.load_state_dict(ck['state_dict']);model.requires_grad_(False);model.eval()
    if digest_state(model.state_dict())!=result['selected_state_sha256']:raise ValueError('Parent state digest differs')
    with np.load(folder/'normalizer.npz',allow_pickle=False) as z:
        scaler={k:z[k].copy() for k in ('mean','std','valid_steps')}
    if not np.array_equal(scaler['mean'],ck['normalizer_mean'].numpy()) or not np.array_equal(scaler['std'],ck['normalizer_std'].numpy()):raise ValueError('Parent normalizer mismatches')
    with (folder/'val_predictions.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    for i,row in enumerate(rows):
        for key in ('window_id','base_window_id','mmsi','region','y'):
            if str(row[key])!=str(data['val'][key][i]):raise ValueError('Parent predictions IDs/labels differ')
    if len(rows)!=len(data['val']['y']):raise ValueError('Parent validation length differs')
    scores=np.asarray([float(x['score']) for x in rows])
    proof=dict(path=str(folder),checkpoint_sha256=base.sha(checkpoint),result_sha256=base.sha(folder/'result.json'),
               state_sha256=digest_state(model.state_dict()),normalizer_sha256=base.sha(folder/'normalizer.npz'),
               val_predictions_sha256=base.sha(folder/'val_predictions.csv'),threshold=result['threshold'],
               validation=result['validation'],raw_normalizer_mean=scaler['mean'].tolist(),raw_normalizer_std=scaler['std'].tolist())
    return model,scaler,scores,proof


@torch.inference_mode()
def encode_cache(model,data,scaler,batch_size):
    x=torch.from_numpy(base.normalized(data['raw'],data['mask'],scaler));mask=torch.from_numpy(data['mask'])
    hs=[];logits=[];scores=[]
    for lo in range(0,len(x),batch_size):
        h=base.masked_pool(model.encode(x[lo:lo+batch_size],mask[lo:lo+batch_size]),mask[lo:lo+batch_size])
        z=model.head(h).squeeze(-1)
        hs.append(h.numpy().copy());logits.append(z.numpy().copy());scores.append(torch.sigmoid(z).numpy().copy())
    return dict(h=np.concatenate(hs),baseline_logits=np.concatenate(logits),baseline_scores=np.concatenate(scores))


def prepare_cache(folder,model,data,scaler,parent_proof,parent_scores,batch_size):
    folder=Path(folder)
    expected=dict(parent_checkpoint_sha256=parent_proof['checkpoint_sha256'],parent_state_sha256=parent_proof['state_sha256'],
                  input_signatures={s:digest_ids(d) for s,d in data.items()},batch_size=batch_size)
    if folder.exists():
        manifest=json.loads((folder/'cache_manifest.json').read_text())
        if manifest['spec']!=expected:raise ValueError('Cache spec changed')
        for name,h in manifest['files'].items():
            if base.sha(folder/name)!=h:raise ValueError('Cache payload changed')
        cache={}
        for split in SPLITS:
            with np.load(folder/(split+'.npz'),allow_pickle=False) as z:cache[split]={k:z[k].copy() for k in ('h','baseline_logits','baseline_scores')}
    else:
        cache={s:encode_cache(model,d,scaler,batch_size) for s,d in data.items()}
        folder.mkdir(parents=True,exist_ok=False)
        for split in SPLITS:
            d=data[split]
            np.savez_compressed(folder/(split+'.npz'),**cache[split],window_id=d['window_id'],base_window_id=d['base_window_id'],
                                y=d['y'],region=d['region'],mask=d['mask'],common_support=d['common_support'])
        manifest=dict(spec=expected,files={s+'.npz':base.sha(folder/(s+'.npz')) for s in SPLITS})
        write_json(folder/'cache_manifest.json',manifest)
    error=float(np.max(np.abs(cache['val']['baseline_scores'].astype(float)-parent_scores)))
    if error!=0:raise AssertionError(f'Parent prediction replay error {error}')
    return cache,dict(cache_path=str(folder),cache_manifest_sha256=base.sha(folder/'cache_manifest.json'),parent_replay_max_abs_error=error)


@torch.inference_mode()
def adapter_predict(model,adapter,h,x,mask,support,batch_size,enabled=True):
    model.eval();adapter.eval();scores=[];norms=[]
    for lo in range(0,len(h),batch_size):
        hi=lo+batch_size
        delta=adapter(x[lo:hi],mask[lo:hi],support[lo:hi]) if enabled else torch.zeros_like(h[lo:hi])
        scores.append(torch.sigmoid(model.head(h[lo:hi]+delta).squeeze(-1)).numpy())
        norms.append(torch.linalg.vector_norm(delta,dim=-1).numpy())
    return np.concatenate(scores),np.concatenate(norms)


def select_epoch(history):
    return max(history,key=lambda x:x['validation_macro_AP'])['epoch']


def prediction_rows(data,scores,delta_norm,threshold,backbone,arm,seed):
    rows=[]
    fractions=data['common_support'].sum(1)/data['mask'].sum(1)
    for i,score in enumerate(scores):
        row=dict(backbone=backbone,arm=arm,seed=seed,y=int(data['y'][i]),score=float(score),threshold=threshold,
                 prediction=int(score>=threshold),delta_norm=float(delta_norm[i]),common_support_fraction=float(fractions[i]))
        row.update({k:str(data[k][i]) for k in META});rows.append(row)
    return rows


def train_arm(model,cache,data,arm,seed,training,scaler,out,parent_proof,cache_proof,config_sha,protocol_sha):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    spec=dict(backbone=model.kind,arm=arm,seed=seed,configuration_sha256=config_sha,protocol_sha256=protocol_sha,
              parent_checkpoint_sha256=parent_proof['checkpoint_sha256'])
    write_json(out/'run_configuration.json',dict(run_spec=spec,training=training,parent=parent_proof,cache=cache_proof))
    np.savez_compressed(out/'adapter_normalizer.npz',**scaler)
    # Seed is reset before every arm, independent of the cache/parent loading RNG.
    base.seed_all(seed+2000)
    adapter=ResidualAdapter();initial_hash=digest_state(adapter.state_dict())
    initial_state=copy.deepcopy(adapter.state_dict())
    tensors={s:dict(h=torch.from_numpy(cache[s]['h']),x=torch.from_numpy(adapter_normalized(d['features'][arm],d['common_support'],scaler)),
                    mask=torch.from_numpy(d['mask']),support=torch.from_numpy(d['common_support']),y=torch.from_numpy(d['y'].astype(np.float32))) for s,d in data.items()}
    model.eval();model.requires_grad_(False);baseline_before=digest_state(model.state_dict())
    if baseline_before!=parent_proof['state_sha256']:raise AssertionError('Baseline changed before training')
    identity={};epoch_scores=[];history=[]
    for split,d in tensors.items():
        scores,_=adapter_predict(model,adapter,d['h'],d['x'],d['mask'],d['support'],training['batch_size'])
        error=float(np.max(np.abs(scores-cache[split]['baseline_scores'])))
        if error!=0:raise AssertionError('Zero adapter is not exactly baseline')
        identity[f'initial_{split}_max_abs_error']=error
    scores=cache['val']['baseline_scores'].copy()
    initial_metrics=base.region_metrics(data['val']['y'],scores,data['val']['region'],.5)
    baseline_train_loss=float(F.binary_cross_entropy_with_logits(torch.from_numpy(cache['train']['baseline_logits']),tensors['train']['y']))
    history.append(dict(epoch=0,train_loss=baseline_train_loss,validation_macro_AP=initial_metrics['macro']['AP']))
    epoch_scores.append(scores.copy());best_ap=history[0]['validation_macro_AP'];best_epoch=0;best=copy.deepcopy(adapter.state_dict())
    orders,order_hash,per_order_hash=make_orders(len(data['train']['y']),seed,training['epochs'])
    opt=torch.optim.AdamW(adapter.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    started=time.time();first_grad=None
    for epoch,order in enumerate(orders,1):
        adapter.train();model.eval();loss_sum=0.;tr=tensors['train']
        for lo in range(0,len(order),training['batch_size']):
            ix=order[lo:lo+training['batch_size']]
            opt.zero_grad(set_to_none=True)
            delta=adapter(tr['x'][ix],tr['mask'][ix],tr['support'][ix])
            logits=model.head(tr['h'][ix]+delta).squeeze(-1)
            loss=F.binary_cross_entropy_with_logits(logits,tr['y'][ix])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adapter loss')
            loss.backward()
            if first_grad is None and tr['support'][ix].any():
                first_grad=float(adapter.last.weight.grad.norm())+float(adapter.last.bias.grad.norm())
            nn.utils.clip_grad_norm_(adapter.parameters(),training['gradient_clip_norm']);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        v=tensors['val'];scores,_=adapter_predict(model,adapter,v['h'],v['x'],v['mask'],v['support'],training['batch_size'])
        m=base.region_metrics(data['val']['y'],scores,data['val']['region'],.5)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(order),validation_macro_AP=m['macro']['AP']))
        epoch_scores.append(scores.copy())
        if m['macro']['AP']>best_ap:
            best_ap=m['macro']['AP'];best_epoch=epoch;best=copy.deepcopy(adapter.state_dict())
        print(f"EPOCH {model.kind}/{arm}/seed{seed} {epoch}/{training['epochs']} val_macro_AP={m['macro']['AP']:.6f}",flush=True)
    if best_epoch!=select_epoch(history):raise AssertionError('Selection rule changed')
    last_state=copy.deepcopy(adapter.state_dict());last_scores=epoch_scores[-1]
    last_threshold,_=base.choose_threshold(data['val']['y'],last_scores)
    last_metrics=base.region_metrics(data['val']['y'],last_scores,data['val']['region'],last_threshold)
    adapter.load_state_dict(best);v=tensors['val']
    selected_scores,delta_norm=adapter_predict(model,adapter,v['h'],v['x'],v['mask'],v['support'],training['batch_size'])
    if not np.array_equal(selected_scores,epoch_scores[best_epoch]):raise AssertionError('Best replay mismatch')
    for split,d in tensors.items():
        bypass,_=adapter_predict(model,adapter,d['h'],d['x'],d['mask'],d['support'],training['batch_size'],enabled=False)
        error=float(np.max(np.abs(bypass-cache[split]['baseline_scores'])))
        if error!=0:raise AssertionError('Disabled adapter altered baseline')
        identity[f'disabled_{split}_max_abs_error']=error
        active,_=adapter_predict(model,adapter,d['h'],d['x'],d['mask'],d['support'],training['batch_size'])
        none=data[split]['common_support'].sum(1)==0
        error=float(np.max(np.abs(active[none]-cache[split]['baseline_scores'][none]))) if none.any() else 0.
        if error!=0:raise AssertionError('No-support window does not bypass')
        identity[f'zero_support_{split}_windows']=int(none.sum());identity[f'zero_support_{split}_max_abs_error']=error
    baseline_after=digest_state(model.state_dict())
    if baseline_after!=baseline_before or any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen baseline changed/received gradients')
    threshold,_=base.choose_threshold(data['val']['y'],selected_scores)
    baseline_threshold,_=base.choose_threshold(data['val']['y'],epoch_scores[0])
    selected_metrics=base.region_metrics(data['val']['y'],selected_scores,data['val']['region'],threshold)
    baseline_metrics=base.region_metrics(data['val']['y'],epoch_scores[0],data['val']['region'],baseline_threshold)
    rows=prediction_rows(data['val'],selected_scores,delta_norm,threshold,model.kind,arm,seed)
    base.write_csv(out/'val_predictions.csv',rows,list(rows[0]))
    np.savez_compressed(out/'epoch_validation_predictions.npz',scores=np.stack(epoch_scores),window_id=data['val']['window_id'],y=data['val']['y'],region=data['val']['region'])
    cp=dict(backbone=model.kind,input_channels=16,hidden=32,baseline_dropout=.1,adapter_hidden=32,
            baseline_state_dict=copy.deepcopy(model.state_dict()),adapter_state_dict=best,initial_adapter_state_dict=initial_state,
            adapter_normalizer_mean=torch.from_numpy(scaler['mean']),adapter_normalizer_std=torch.from_numpy(scaler['std']),
            raw_normalizer_mean=torch.tensor(parent_proof['raw_normalizer_mean']),raw_normalizer_std=torch.tensor(parent_proof['raw_normalizer_std']),
            selected_epoch=best_epoch,threshold=threshold,parent_checkpoint_sha256=parent_proof['checkpoint_sha256'],protocol_sha256=protocol_sha,configuration_sha256=config_sha)
    torch.save(cp,out/'checkpoint.pt')
    torch.save(dict(adapter_state_dict=last_state,epoch=training['epochs'],threshold=last_threshold,parent_checkpoint_sha256=parent_proof['checkpoint_sha256']),out/'checkpoint_last.pt')
    write_json(out/'history.json',history)
    result=dict(status='completed_validation_only',run_spec=spec,**spec,adapter_parameters=sum(p.numel() for p in adapter.parameters()),
                frozen_parameters=sum(p.numel() for p in model.parameters()),epochs_budget=training['epochs'],selected_epoch=best_epoch,
                baseline_validation=baseline_metrics,validation=selected_metrics,final_epoch_validation=last_metrics,
                threshold=threshold,final_epoch_threshold=last_threshold,baseline_threshold=baseline_threshold,history=history,
                initial_adapter_state_sha256=initial_hash,selected_adapter_state_sha256=digest_state(best),last_adapter_state_sha256=digest_state(last_state),
                frozen_state_sha256_before=baseline_before,frozen_state_sha256_after=baseline_after,
                batch_order_sha256=order_hash,epoch_batch_order_sha256=per_order_hash,identity_checks=identity,
                first_supported_batch_output_gradient_norm=first_grad,normalizer_supported_training_steps=int(scaler['valid_steps']),
                selected_delta_norm=dict(mean=float(delta_norm.mean()),max=float(delta_norm.max()),nonzero_windows=int((delta_norm!=0).sum())),
                parent=parent_proof,cache=cache_proof,elapsed_seconds=time.time()-started,
                identity_scope='Identity checks concern logits/probabilities. A reselected validation-F1 threshold can change binary labels even for unchanged probabilities.',
                control_scope='self_quality receives the shared neighborhood-availability mask and reliability metadata; it is not a context-information-free single-agent model.',
                scope='Validation development; epoch0 retained as candidate, no test access or generalization floor guarantee')
    if first_grad is None or first_grad<=0:raise AssertionError('Zero-output adapter failed to receive a learning signal')
    result['output_hashes']={p.name:base.sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out/'result.json',result)
    write_json(out/'completion.json',dict(status='complete',result_sha256=base.sha(out/'result.json')))
    print('ARM_COMPLETE',model.kind,seed,arm,'epoch',best_epoch,'AP',selected_metrics['macro']['AP'],flush=True)
    return result


def resume_run(path,spec):
    result=json.loads((path/'result.json').read_text());complete=json.loads((path/'completion.json').read_text())
    if complete['result_sha256']!=base.sha(path/'result.json') or result['run_spec']!=spec:raise ValueError('Resume run changed')
    for name,h in result['output_hashes'].items():
        if base.sha(path/name)!=h:raise ValueError('Resume artifact changed')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--backbone',choices=base.BACKBONES,required=True)
    ap.add_argument('--cache',type=Path);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();protocol=read_protocol(args.protocol);training=protocol['training']
    torch.set_num_threads(training['cpu_threads_per_worker']);torch.set_num_interop_threads(1)
    data,sources,manifest=load_data(args.parent)
    scales={a:fit_adapter_scaler(data['train']['features'][a],data['train']['common_support']) for a in ARMS}
    cache_root=args.cache or args.out.parent/'cache'
    config=dict(stage=protocol['stage'],backbone=args.backbone,protocol_sha256=base.sha(args.protocol),driver_sha256=base.sha(Path(__file__)),
                vendor_sha256=VENDOR_SHA,parent_stage=str(args.parent.resolve()),sources=sources,training=training,adapter=protocol['adapter'],arms=list(ARMS),
                input_signatures={s:digest_ids(d) for s,d in data.items()},
                support_summary={s:dict(windows=len(d['y']),valid_steps=int(d['mask'].sum()),trusted_steps=int(d['trusted_support'].sum()),common_steps=int(d['common_support'].sum()),zero_support_windows=int((d['common_support'].sum(1)==0).sum())) for s,d in data.items()},
                versions=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,sklearn=__import__('sklearn').__version__))
    worker=args.out/args.backbone
    if worker.exists():
        if not args.resume or json.loads((worker/'configuration.json').read_text())!=config:raise ValueError('Worker exists or resume configuration changed')
    else:
        worker.mkdir(parents=True,exist_ok=False);write_json(worker/'configuration.json',config)
        base.write_csv(worker/'input_manifest.csv',manifest,list(manifest[0]))
    config_sha=base.sha(worker/'configuration.json');results=[];preflight=[]
    for seed in training['seeds']:
        model,raw_scaler,parent_scores,parent_proof=load_parent(args.parent,args.backbone,seed,data,sources)
        cache,cache_proof=prepare_cache(cache_root/args.backbone/f'seed_{seed}',model,data,raw_scaler,parent_proof,parent_scores,training['batch_size'])
        preflight_row=dict(seed=seed,parent=parent_proof,cache=cache_proof)
        if args.prepare_only:
            checks={}
            for arm in ARMS:
                base.seed_all(seed+2000);adapter=ResidualAdapter()
                checks[arm]={}
                for split in SPLITS:
                    d=data[split]
                    pred,_=adapter_predict(model,adapter,torch.from_numpy(cache[split]['h']),
                        torch.from_numpy(adapter_normalized(d['features'][arm],d['common_support'],scales[arm])),
                        torch.from_numpy(d['mask']),torch.from_numpy(d['common_support']),training['batch_size'])
                    error=float(np.max(np.abs(pred-cache[split]['baseline_scores'])))
                    if error!=0:raise AssertionError('Preflight zero adapter mismatch')
                    checks[arm][split]=error
            preflight_row['zero_adapter_max_abs_error']=checks
            preflight.append(preflight_row)
            continue
        preflight.append(preflight_row)
        for arm in ARMS:
            path=worker/f'seed_{seed}'/arm
            spec=dict(backbone=args.backbone,arm=arm,seed=seed,configuration_sha256=config_sha,protocol_sha256=base.sha(args.protocol),parent_checkpoint_sha256=parent_proof['checkpoint_sha256'])
            if path.exists():
                if not args.resume:raise ValueError('Run exists; refusing overwrite')
                result=resume_run(path,spec)
            else:
                result=train_arm(model,cache,data,arm,seed,training,scales[arm],path,parent_proof,cache_proof,config_sha,base.sha(args.protocol))
            results.append(result)
    if args.prepare_only:
        write_json(worker/'preflight.json',dict(status='passed',baselines=preflight,support_summary=config['support_summary']))
        print('PREPARED_ONLY',args.backbone,flush=True);return
    for seed in training['seeds']:
        paired=[r for r in results if r['seed']==seed]
        for key in ('initial_adapter_state_sha256','batch_order_sha256','frozen_state_sha256_before','adapter_parameters'):
            if len({r[key] for r in paired})!=1:raise AssertionError(f'Unpaired {key}')
    if not all(base.sha(Path(s['path']))==s['sha256'] for s in sources):raise ValueError('Parent feature files changed')
    summary=dict(status='completed_validation_only',backbone=args.backbone,runs=len(results),protocol_sha256=base.sha(args.protocol),source_data_unchanged=True,
                 epoch0_selections=sum(r['selected_epoch']==0 for r in results),support_summary=config['support_summary'],
                 results=[{k:r[k] for k in ('backbone','arm','seed','selected_epoch','validation','baseline_validation','final_epoch_validation','identity_checks','elapsed_seconds')} for r in results])
    if (worker/'worker_summary.json').exists():
        if json.loads((worker/'worker_summary.json').read_text())!=summary:raise ValueError('Summary changed on resume')
    else:write_json(worker/'worker_summary.json',summary)
    print('BACKBONE_COMPLETE',args.backbone,len(results),flush=True)


if __name__=='__main__':main()
