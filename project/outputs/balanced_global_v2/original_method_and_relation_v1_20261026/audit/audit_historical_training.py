"""Independent fixed/best checkpoint and metric replay for historical transfer.

Uses only the archived step35 layer definitions; it does not call the new
trainer's loader, normalization, forward wrapper, metrics or selection code.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score,roc_auc_score

METRICS=('AP','AUROC','F1','precision','recall')
META=('window_id','base_window_id','mmsi','region','observation_noise','noise_type','noise_scope','target_noise_effective','neighbor_noise_effective','observation_noise_effective')
def check(ok,msg):
    if not ok:raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def state_hash(sd):
    h=hashlib.sha256()
    for k in sorted(sd):
        x=sd[k].detach().cpu().contiguous().numpy()
        for v in (k.encode(),str(x.dtype).encode(),canon(list(x.shape)).encode()):h.update(v+b'\0')
        h.update(x.tobytes())
    return h.hexdigest()
def ids_hash(d):
    rows=[[str(d[k][i]) for k in ('window_id','base_window_id','mmsi','region')]+[int(d['y'][i]),int(d['label_class'][i])] for i in range(len(d['y']))]
    return hashlib.sha256(canon(rows).encode()).hexdigest()
def order_hash(clean,seed,epochs):
    rng=np.random.RandomState(seed);indices=clean.copy();h=hashlib.sha256();per=[]
    for ep in range(1,epochs+1):
        rng.shuffle(indices);payload=np.array([ep],dtype='<i8').tobytes()+indices.astype('<i8').tobytes();h.update(payload);per.append(hashlib.sha256(payload).hexdigest())
    return h.hexdigest(),per
def threshold(y,s):
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(yy)[ends];fp=(ends+1)-tp;fn=y.sum()-tp;f1=2*tp/np.maximum(2*tp+fp+fn,1)
    ix=np.flatnonzero(np.isclose(f1,f1.max(),rtol=0,atol=1e-12))[0]
    return float(ss[ends[ix]])
def metric(y,s,th):
    pred=s>=th;tp=int((pred&(y==1)).sum());fp=int((pred&(y==0)).sum());fn=int((~pred&(y==1)).sum());tn=int((~pred&(y==0)).sum())
    return dict(AP=float(average_precision_score(y,s)),AUROC=float(roc_auc_score(y,s)),F1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),confusion={'tn':tn,'fp':fp,'fn':fn,'tp':tp},samples=len(y),positive_samples=int(y.sum()),positive_fraction=float(y.mean()))
def regions(y,s,rg,th):
    per={str(r):metric(y[rg==r],s[rg==r],th) for r in sorted(set(rg))}
    return dict(macro={m:float(np.mean([v[m] for v in per.values()])) for m in METRICS},pooled=metric(y,s,th),per_region=per)
def compare(actual,recorded,label):
    for m in METRICS:check(abs(actual['macro'][m]-recorded['macro'][m])<1e-10,label+'/macro/'+m)
    for r,v in actual['per_region'].items():
        for m in METRICS:check(abs(v[m]-recorded['per_region'][r][m])<1e-10,label+'/'+r+'/'+m)
        check(v['confusion']==recorded['per_region'][r]['confusion'],label+'/confusion')
    for m in METRICS:check(abs(actual['pooled'][m]-recorded['pooled'][m])<1e-10,label+'/pooled/'+m)

def load(data_path,protocol):
    keys=sorted(set(x['input'] for x in protocol['arm_specs'].values()));parts={s:[] for s in ('train','val')};proof=[]
    for r in protocol['regions']:
        for split in ('train','val'):
            p=data_path/r/f'{split}.npz'
            with np.load(p,allow_pickle=False) as z:
                d={k:z[k].copy() for k in tuple(keys)+META+('mask','y','label_class','rf','ep','nb_seq','nb_xy','nb_mask','nb_w')}
                check(len(d['y'])=={'train':980,'val':210}[split],'count')
                check(np.array_equal(d['y']>0,d['label_class']>0),'type labels')
                check(np.array_equal(d['mask'],np.arange(20)[None,:]<d['mask'].sum(1)[:,None]),'mask')
            parts[split].append(d);proof.append(dict(region=r,split=split,path=str(p),sha256=sha(p),windows=len(d['y']),valid_steps=int(d['mask'].sum()),clean_windows=int((~d['observation_noise']).sum())))
    data={s:{k:np.concatenate([p[k] for p in pp]) for k in pp[0]} for s,pp in parts.items()}
    for key in ('mmsi','window_id','base_window_id'):check(not set(data['train'][key])&set(data['val'][key]),'split overlap')
    scalers={}
    for k in keys:
        valid=data['train'][k][data['train']['mask']]
        scalers[k]=dict(mean=valid.mean(0,keepdims=True),std=valid.std(0,keepdims=True)+np.float32(1e-6),valid_steps=len(valid))
    return data,proof,scalers

def original_core(legacy,kind,nf,graph,recipe):
    if graph:return legacy.Net(kind,nf,'ours',64)
    core=nn.Module();core.enc=legacy.Enc(kind,nf,64);core.bin=nn.Linear(core.enc.dim,1);core.cls=nn.Linear(core.enc.dim,5)
    if recipe:core.ctrl=nn.Linear(4,2)
    return core

@torch.inference_mode()
def replay(core,x,d,padding,graph,device):
    core.eval();probs=[];types=[]
    for lo in range(0,len(x),128):
        hi=lo+128;xb=torch.from_numpy(x[lo:hi]).to(device);mask=torch.from_numpy(d['mask'][lo:hi]).to(device)
        if padding=='compat' or bool(mask.all()):h=core.enc(xb)
        else:
            lengths=mask.sum(1);h=torch.empty((len(xb),core.enc.dim),device=device,dtype=xb.dtype)
            for length in torch.unique(lengths).tolist():
                rows=torch.where(lengths==length)[0];h[rows]=core.enc(xb[rows,:length])
        if graph:
            fields={k:torch.from_numpy(d[k][lo:hi]).to(device) for k in ('nb_seq','nb_xy','nb_mask','nb_w','rf')}
            ri=fields['rf'][:,-1] if padding=='compat' else fields['rf'][torch.arange(len(xb),device=device),mask.sum(1)-1]
            h,_=core.graph(h,fields['nb_seq'],fields['nb_xy'],fields['nb_mask'],fields['nb_w'],ri)
        probs.append(torch.sigmoid(core.bin(h).squeeze(-1)).cpu().numpy());types.append(torch.softmax(core.cls(h),-1).cpu().numpy())
    return np.concatenate(probs),np.concatenate(types)

def distribution(x):
    a=np.array(x,float);return dict(values=a.tolist(),mean=float(a.mean()) if len(a) else None,sample_std=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--allow-partial',action='store_true');ap.add_argument('--device',default='cpu');args=ap.parse_args()
    check(not args.out.exists(),'Refusing overwrite');root=args.stage;protocol=json.loads((root/'protocol.json').read_text());ph=sha(root/'protocol.json')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    spec=importlib.util.spec_from_file_location('original35_independent',root/'historical/vendor/35_graph_ours_final.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    data,proof,scalers=load(root/'data',protocol);clean=np.flatnonzero(~data['train']['observation_noise']);cnt=np.bincount(data['train']['label_class'][clean][data['train']['label_class'][clean]>0]-1,minlength=5)
    results=[];missing=[]
    for kind in protocol['backbones']:
        worker=root/'runs'/kind
        if not (worker/'configuration.json').exists() and args.allow_partial:continue
        config=json.loads((worker/'configuration.json').read_text());ch=sha(worker/'configuration.json')
        check(config['protocol_sha256']==ph and config['training']==protocol['training'],'worker protocol')
        check(config['arms']==protocol['arms'],'worker arms')
        check({(x['region'],x['split']):x['sha256'] for x in config['sources']}=={(x['region'],x['split']):x['sha256'] for x in proof},'source hashes')
        for name,h in config['code_sha256'].items():check(sha(root/'historical'/name)==h,'training code changed')
        for split in ('train','val'):check(config['input_signatures'][split]==ids_hash(data[split]),'IDs signature')
        for seed in protocol['training']['seeds']:
            oh,epochs_order=order_hash(clean,seed,25)
            for arm in protocol['arms']:
                path=worker/f'seed_{seed}'/arm
                if not (path/'completion.json').exists():missing.append(str(path));continue
                result=json.loads((path/'result.json').read_text());done=json.loads((path/'completion.json').read_text());check(done['result_sha256']==sha(path/'result.json'),'result hash')
                check(result['protocol_sha256']==ph and result['configuration_sha256']==ch,'run config')
                for name,h in result['output_hashes'].items():check(sha(path/name)==h,'output hash '+name)
                cfg=protocol['arm_specs'][arm];nf=data['train'][cfg['input']].shape[-1]
                check(result['gradient_training_windows']==len(clean) and result['all_training_windows']==7840,'clean train population')
                check(result['positive_type_counts']==cnt.tolist(),'class weights population')
                check(result['batch_order_sha256']==oh and result['epoch_batch_order_sha256']==epochs_order,'sample order')
                with np.load(path/'normalizer.npz',allow_pickle=False) as z:
                    for k in ('mean','std'):check(np.array_equal(z[k],scalers[cfg['input']][k]),'normalizer '+k)
                    check(int(z['valid_steps'])==scalers[cfg['input']]['valid_steps'],'normalizer count')
                torch.manual_seed(seed);core=original_core(legacy,kind,nf,cfg['graph'],cfg['augmentation_bundle']);wrapped=nn.Module();wrapped.core=core
                initial=wrapped.state_dict();params=sum(x.numel() for x in core.parameters());check(params==result['parameters'],'parameter count')
                check(state_hash(initial)==result['initial_state_hashes']['full'],'initial state')
                for name,module in [('encoder',core.enc),('binary_head',core.bin),('type_head',core.cls)]:check(state_hash(module.state_dict())==result['initial_state_hashes'][name],'initial '+name)
                if cfg['graph']:check(params==protocol['historical_reference']['full_parameters'][kind],'full source identity')
                with np.load(path/'epoch_validation_predictions.npz',allow_pickle=False) as z:
                    scores=z['scores'];check(scores.shape==(25,1680),'epoch predictions shape')
                    for k in ('window_id','y','label_class','region'):check(np.array_equal(z[k],data['val'][k]),'epoch rows')
                values=[]
                for ep,pred in enumerate(scores,1):
                    value=regions(data['val']['y'],pred,data['val']['region'],.5)['macro']['AP'];values.append(value)
                    check(abs(value-result['history'][ep-1]['validation_macro_AP'])<1e-10,'epoch AP')
                    history=result['history'][ep-1]
                    expected=history['train_binary']+.5*history['train_type_ce']
                    if cfg['augmentation_bundle']:
                        expected+=.2*history['train_js']+.3*history['train_aug_binary']+.01*history['train_hidden_smooth']
                        check(.05-1e-8<=history['train_epsilon_mean']<=.25+1e-8,'epsilon range')
                        check(history['train_js']>=-1e-8 and history['train_aug_binary']>=0 and history['train_hidden_smooth']>=0,'loss values')
                    else:
                        check(all(history['train_'+k]==0 for k in ('js','aug_binary','hidden_smooth','epsilon_mean')),'disabled recipe nonzero')
                    check(abs(expected-history['train_total'])<2e-5,'loss decomposition')
                best=int(np.argmax(values))+1;check(best==result['best_epoch'] and result['primary_epoch']==25,'checkpoint selection')
                s=scalers[cfg['input']];x=((data['val'][cfg['input']]-s['mean'])/s['std']).astype(np.float32)
                if cfg['padding']=='masked':x[~data['val']['mask']]=0
                outcomes={};gamma={};errors={};unused=0
                for which,ep in [('final',25),('best',best)]:
                    pred=scores[ep-1];th=threshold(data['val']['y'],pred);check(abs(th-result[which+'_threshold'])<1e-10,'threshold')
                    actual=regions(data['val']['y'],pred,data['val']['region'],th);compare(actual,result[which+'_validation'],kind+'/'+arm+'/'+which)
                    for label,mask in [('clean',~data['val']['observation_noise']),('requested_corruption',data['val']['observation_noise'])]:compare(regions(data['val']['y'][mask],pred[mask],data['val']['region'][mask],th),result[which+'_strata'][label],'stratum')
                    with (path/f'val_predictions_{which}.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
                    check(len(rows)==1680,'csv length')
                    for i,row in enumerate(rows):
                        for k in META:check(row[k]==str(data['val'][k][i]),'csv metadata '+k)
                        check(int(row['y'])==data['val']['y'][i] and int(row['label_class'])==data['val']['label_class'][i],'csv labels')
                        check(abs(float(row['score'])-float(pred[i]))<1e-12 and abs(float(row['threshold'])-th)<1e-12,'csv score')
                        check(int(row['prediction'])==int(pred[i]>=th),'csv binary')
                    ck=torch.load(path/f'checkpoint_{which}.pt',map_location='cpu',weights_only=True)
                    check(state_hash(ck['state_dict'])==result[which+'_state_sha256'],'state hash')
                    check(ck['epoch']==ep and ck['threshold']==th and ck['protocol_sha256']==ph,'checkpoint metadata')
                    for k in ('mean','std'):check(np.array_equal(ck['normalizer_'+k].numpy(),s[k]),'checkpoint normalization')
                    if cfg['graph']:
                        for name in ('core.rec.weight','core.rec.bias'):check(torch.equal(ck['state_dict'][name],initial[name]),'historical unused rec changed')
                        unused=sum(initial[n].numel() for n in ('core.rec.weight','core.rec.bias'))
                        gamma[which]=float(ck['state_dict']['core.gamma'].item())
                    core.load_state_dict({k.removeprefix('core.'):v for k,v in ck['state_dict'].items()});core.to(device)
                    binary,typed=replay(core,x,data['val'],cfg['padding'],cfg['graph'],device)
                    error=float(np.max(abs(binary-pred)));check(error<5e-6,'independent forward replay '+str(error))
                    with np.load(path/'type_predictions.npz',allow_pickle=False) as z:
                        type_error=float(np.max(abs(typed-z[which])));check(type_error<5e-6,'type head replay')
                    errors[which]=dict(binary_max_abs=error,type_max_abs=type_error);outcomes[which]=actual
                    core.to('cpu')
                results.append(dict(backbone=kind,arm=arm,seed=seed,parameters=params,structurally_unused_rec_parameters=unused,
                                    parameters_with_training_computation_path=params-unused,primary_epoch=25,best_epoch=best,
                                    final_validation=outcomes['final'],best_validation=outcomes['best'],replay_errors=errors,graph_gamma=gamma,
                                    eps_epoch_means=[h['train_epsilon_mean'] for h in result['history']],js_epoch_means=[h['train_js'] for h in result['history']],
                                    batch_order_sha256=oh,result_sha256=sha(path/'result.json'),initial_state_hashes=result['initial_state_hashes']))
                print('RUN_AUDITED',kind,seed,arm,flush=True)
    check(args.allow_partial or len(results)==48,'Missing completed runs '+str(len(missing)))
    aggregate=[];contrasts=[]
    idx={(x['backbone'],x['arm'],x['seed']):x for x in results}
    for kind in protocol['backbones']:
        for arm in protocol['arms']:
            rr=[idx[kind,arm,s] for s in protocol['training']['seeds'] if (kind,arm,s) in idx]
            aggregate.append(dict(backbone=kind,arm=arm,seeds=[x['seed'] for x in rr],**{which:{m:distribution([x[which+'_validation']['macro'][m] for x in rr]) for m in METRICS} for which in ('final','best')}))
        for fr,to in protocol['predeclared_contrasts']:
            pairs=[(idx[kind,fr,s],idx[kind,to,s]) for s in protocol['training']['seeds'] if (kind,fr,s) in idx and (kind,to,s) in idx]
            contrasts.append(dict(backbone=kind,from_arm=fr,to_arm=to,**{which:{m:distribution([b[which+'_validation']['macro'][m]-a[which+'_validation']['macro'][m] for a,b in pairs]) for m in METRICS} for which in ('final','best')}))
    args.out.mkdir(parents=True)
    status='passed' if len(results)==48 else 'partial'
    audit=dict(status=status,protocol_sha256=ph,source_files=proof,clean_gradient_windows=len(clean),positive_type_counts=cnt.tolist(),runs=results,missing=missing)
    summary=dict(status=status,scope='Fixed epoch25 primary; best-validation secondary; safe data migration of historical network. No old test replay or causal single-factor data-effect claim.',completed_runs=len(results),all_configurations=aggregate,paired_contrasts=contrasts,protocol_sha256=ph)
    for name,obj in [('independent_historical_training_audit.json',audit),('historical_reproduction_summary.json',summary)]:
        (args.out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':status,'runs':len(results),'out':str(args.out)}))

if __name__=='__main__':main()
