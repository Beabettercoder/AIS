"""Independent audit of frozen-backbone adapters, with no test-set access.

Only the previously frozen backbone architecture is imported for weight replay.
Feature construction, normalization, pooling, adapter forward, metrics, epoch
selection and threshold selection are independently implemented here.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,math
import numpy as np
import torch
from torch import nn
from audit_helpers import *
from independent_adapter_math import *


def expected_orders(n,seed,epochs):
    rng=np.random.default_rng(seed+100);h=hashlib.sha256();per=[]
    for e in range(1,epochs+1):
        payload=np.asarray([e],dtype='<i8').tobytes()+np.asarray(rng.permutation(n),dtype='<i8').tobytes()
        h.update(payload);per.append(hashlib.sha256(payload).hexdigest())
    return h.hexdigest(),per


def expected_initial_adapter(seed):
    torch.manual_seed(seed+2000)
    first=nn.Linear(16,32);last=nn.Linear(32,64)
    nn.init.zeros_(last.weight);nn.init.zeros_(last.bias)
    return {'first.weight':first.weight.detach(),'first.bias':first.bias.detach(),
            'last.weight':last.weight.detach(),'last.bias':last.bias.detach()}


def compare_state(a,b,label):
    require(set(a)==set(b),f'{label}: state keys')
    for k in a:require(torch.equal(a[k].cpu(),b[k].cpu()),f'{label}: {k} changed')


def check_worker(folder,parent,protocol,protocol_hash,training_dir,source_proof,data):
    conf=json.loads((folder/'configuration.json').read_text())
    require(conf['protocol_sha256']==protocol_hash,f'{folder}: protocol')
    require(conf['training']==protocol['training'] and conf['adapter']==protocol['adapter'],f'{folder}: settings')
    require(conf['arms']==protocol['arms'],f'{folder}: arms')
    require(conf['driver_sha256']==sha(training_dir/'train_adapters.py'),f'{folder}: driver changed')
    require(conf['vendor_sha256']==sha(training_dir/'vendor/pilot_train_frozen.py'),f'{folder}: vendor changed')
    actual={(x['region'],x['split']):x['sha256'] for x in conf['sources']}
    expected={(x['region'],x['split']):x['sha256'] for x in source_proof}
    require(actual==expected and len(conf['sources'])==16,f'{folder}: input source list')
    require(all(x['split'] in ('train','val') and Path(x['path']).name==x['split']+'.npz' for x in conf['sources']),f'{folder}: forbidden split')
    with (folder/'input_manifest.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    require(len(rows)==9520,f'{folder}: manifest row count')
    for split in ('train','val'):
        rr=[x for x in rows if x['split']==split]
        require(len(rr)==len(data[split]['y']),f'{folder}: manifest {split}')
        for i,r in enumerate(rr):
            for k in ('region','window_id','base_window_id','mmsi','y'):
                require(str(r[k])==str(data[split][k][i]),f'{folder}: manifest {split}/{i}/{k}')
    return sha(folder/'configuration.json')


def verify_parent_cache(parent,stage,b,seed,data,vendor,batch):
    folder=parent/'runs'/b/f'seed_{seed}'/'raw16'
    cp=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=vendor.BackboneClassifier(b,16,32,.1);model.load_state_dict(cp['state_dict']);model.eval();model.requires_grad_(False)
    baseline_state=state_hash(model.state_dict())
    with np.load(folder/'normalizer.npz',allow_pickle=False) as z:
        mean=z['mean'].copy();std=z['std'].copy()
    require(np.array_equal(mean,cp['normalizer_mean'].numpy()) and np.array_equal(std,cp['normalizer_std'].numpy()),f'{b}/{seed}: parent scaler differs')
    parent_score,parent_threshold=load_csv_predictions(folder/'val_predictions.csv',data['val'])
    cache_folder=stage/'cache'/b/f'seed_{seed}'
    cache_manifest=json.loads((cache_folder/'cache_manifest.json').read_text())
    require(cache_manifest['spec']['parent_checkpoint_sha256']==sha(folder/'checkpoint.pt'),f'{b}/{seed}: cache parent hash')
    require(cache_manifest['spec']['parent_state_sha256']==baseline_state,f'{b}/{seed}: cache parent state')
    require(cache_manifest['spec']['batch_size']==batch,f'{b}/{seed}: cache batch')
    replay={};proof={}
    for split in ('train','val'):
        cache_path=cache_folder/f'{split}.npz'
        require(sha(cache_path)==cache_manifest['files'][split+'.npz'],f'{cache_path}: cache hash')
        x=normalized_baseline(data[split]['raw16'],data[split]['mask'],mean,std)
        h,logits,scores=reference_baseline(model,x,data[split]['mask'],batch)
        with np.load(cache_path,allow_pickle=False) as z:
            for k in ('window_id','base_window_id','y','region','mask','common_support'):
                require(np.array_equal(z[k],data[split][k]),f'{cache_path}: {k}')
            for key,actual in [('h',h),('baseline_logits',logits),('baseline_scores',scores)]:
                require(np.array_equal(z[key],actual),f'{cache_path}: {key} replay')
        replay[split]=dict(h=h,logits=logits,scores=scores)
        proof[split]=dict(windows=len(scores),cached_latent_and_scores_exact=True)
    error=float(np.max(np.abs(replay['val']['scores'].astype(float)-parent_score)))
    require(error==0,f'{b}/{seed}: original parent prediction replay')
    require(sha(parent/'protocol.json')==json.loads((stage/'protocol.json').read_text())['parent_protocol_sha256'],'Parent protocol changed')
    proof.update(parent_checkpoint_sha256=sha(folder/'checkpoint.pt'),parent_state_sha256=baseline_state,
                 parent_replay_max_absolute_error=error,cache_manifest_sha256=sha(cache_folder/'cache_manifest.json'),
                 raw_normalizer_mean=mean.tolist(),raw_normalizer_std=std.tolist(),parent_threshold=parent_threshold)
    return model,replay,proof,cp


def audit_run(path,b,seed,arm,data,scaler,protocol,ph,config_hash,model,replay,parent_proof,parent_cp):
    result=json.loads((path/'result.json').read_text());complete=json.loads((path/'completion.json').read_text())
    require(result['status']=='completed_validation_only' and complete['result_sha256']==sha(path/'result.json'),f'{path}: completion')
    require(result['protocol_sha256']==ph and result['configuration_sha256']==config_hash,f'{path}: configuration')
    require(result['backbone']==b and result['seed']==seed and result['arm']==arm,f'{path}: identity')
    require(result['parent_checkpoint_sha256']==parent_proof['parent_checkpoint_sha256'],f'{path}: parent')
    for name,wanted in result['output_hashes'].items():
        require(Path(name).name==name and 'test' not in name.lower(),f'{path}: unexpected output')
        require(sha(path/name)==wanted,f'{path}: output hash {name}')
    with np.load(path/'adapter_normalizer.npz',allow_pickle=False) as z:
        for k in ('mean','std'):require(np.array_equal(z[k],scaler[k]),f'{path}: train-only {k}')
        require(int(z['valid_steps'])==scaler['valid_steps'],f'{path}: scaler step count')
    cp=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    last=torch.load(path/'checkpoint_last.pt',map_location='cpu',weights_only=True)
    compare_state(cp['baseline_state_dict'],parent_cp['state_dict'],str(path)+' frozen baseline')
    expected_state=expected_initial_adapter(seed)
    compare_state(cp['initial_adapter_state_dict'],expected_state,str(path)+' initial adapter')
    require(state_hash(expected_state)==result['initial_adapter_state_sha256'],f'{path}: initial hash')
    require(state_hash(cp['adapter_state_dict'])==result['selected_adapter_state_sha256'],f'{path}: selected adapter hash')
    require(state_hash(last['adapter_state_dict'])==result['last_adapter_state_sha256'],f'{path}: last adapter hash')
    expected_frozen=parent_proof['parent_state_sha256']
    require(result['frozen_state_sha256_before']==result['frozen_state_sha256_after']==expected_frozen,f'{path}: frozen hash')
    require(result['first_supported_batch_output_gradient_norm']>0,f'{path}: missing learning signal')
    require(result['adapter_parameters']==2656,f'{path}: unexpected parameter count')
    require(np.array_equal(cp['raw_normalizer_mean'].numpy(),parent_cp['normalizer_mean'].numpy()) and np.array_equal(cp['raw_normalizer_std'].numpy(),parent_cp['normalizer_std'].numpy()),f'{path}: parent scaler persisted')
    for key in ('mean','std'):require(np.array_equal(cp[f'adapter_normalizer_{key}'].numpy(),scaler[key]),f'{path}: checkpoint scaler')
    order,per=expected_orders(len(data['train']['y']),seed,protocol['training']['epochs'])
    require(result['batch_order_sha256']==order and result['epoch_batch_order_sha256']==per,f'{path}: batch order')
    with np.load(path/'epoch_validation_predictions.npz',allow_pickle=False) as z:
        epochs=z['scores'].copy()
        for k in ('window_id','y','region'):require(np.array_equal(z[k],data['val'][k]),f'{path}: epoch source {k}')
    require(epochs.shape==(13,len(data['val']['y'])),f'{path}: epoch array shape')
    require(np.array_equal(epochs[0],replay['val']['scores']),f'{path}: epoch0 differs')
    history=json.loads((path/'history.json').read_text())
    require(history==result['history'] and [x['epoch'] for x in history]==list(range(13)),f'{path}: history')
    independently_computed=[]
    for e in range(13):
        ap=region_metrics(data['val']['y'],epochs[e],data['val']['region'],.5)['macro']['AP']
        require(math.isclose(ap,history[e]['validation_macro_AP'],rel_tol=0,abs_tol=1e-12),f'{path}: epoch {e} AP')
        independently_computed.append(ap)
    selected=max(range(13),key=lambda e:independently_computed[e])
    require(selected==result['selected_epoch']==cp['selected_epoch'],f'{path}: epoch selection')
    require(independently_computed[selected]>=independently_computed[0],f'{path}: selection below epoch0')
    scores,threshold=load_csv_predictions(path/'val_predictions.csv',data['val'])
    require(np.array_equal(scores,epochs[selected].astype(float)),f'{path}: chosen scores')
    actual={};thresholds={}
    for key,e in [('baseline_validation',0),('validation',selected),('final_epoch_validation',12)]:
        th,_=threshold_from_counts(data['val']['y'],epochs[e]);thresholds[key]=th
        actual[key]=region_metrics(data['val']['y'],epochs[e],data['val']['region'],th)
        compare_metrics(actual[key],result[key],str(path)+'/'+key)
    require(threshold==thresholds['validation']==result['threshold']==cp['threshold'],f'{path}: selected threshold')
    require(last['threshold']==result['final_epoch_threshold']==thresholds['final_epoch_validation'],f'{path}: last threshold')
    require(result['baseline_threshold']==thresholds['baseline_validation'],f'{path}: baseline threshold')
    identity={};replayed_selected=None;selected_norm=None
    for split in ('train','val'):
        d=data[split];x=normalized_adapter(d['features'][arm],d['common_support'],scaler)
        initial,_=reference_predictions(model.head,replay[split]['h'],x,d['mask'],d['common_support'],expected_state)
        err=float(np.max(np.abs(initial-replay[split]['scores'])))
        require(err==0,f'{path}: zero-init identity {split}');identity[f'initial_{split}_max_abs_error']=err
        selected_scores,deltas=reference_predictions(model.head,replay[split]['h'],x,d['mask'],d['common_support'],cp['adapter_state_dict'])
        unsupported=d['common_support'].sum(1)==0
        error=float(np.max(np.abs(selected_scores[unsupported]-replay[split]['scores'][unsupported]))) if unsupported.any() else 0.
        require(error==0 and np.all(deltas[unsupported]==0),f'{path}: no-support bypass {split}')
        identity[f'zero_support_{split}_windows']=int(unsupported.sum());identity[f'zero_support_{split}_max_abs_error']=error
        # Disabled branch is the same unchanged head applied to unchanged cache.
        disabled=[]
        with torch.inference_mode():
            for lo in range(0,len(replay[split]['h']),128):
                disabled.append(torch.sigmoid(model.head(torch.from_numpy(replay[split]['h'][lo:lo+128])).squeeze(-1)).numpy())
        bypass=np.concatenate(disabled)
        error=float(np.max(np.abs(bypass-replay[split]['scores'])))
        require(error==0,f'{path}: disabled identity {split}');identity[f'disabled_{split}_max_abs_error']=error
        if split=='val':
            replayed_selected=selected_scores;selected_norm=np.linalg.norm(deltas,axis=1)
            final_scores,_=reference_predictions(model.head,replay[split]['h'],x,d['mask'],d['common_support'],last['adapter_state_dict'])
            require(np.array_equal(final_scores,epochs[12]),f'{path}: epoch12 checkpoint replay')
    require(identity==result['identity_checks'],f'{path}: recorded identity proof')
    require(np.array_equal(replayed_selected,epochs[selected]),f'{path}: selected checkpoint replay')
    with (path/'val_predictions.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    recorded_norm=np.asarray([float(r['delta_norm']) for r in rows])
    require(np.allclose(recorded_norm,selected_norm,rtol=1e-6,atol=1e-7),f'{path}: delta norm')
    require(np.array_equal([float(r['common_support_fraction']) for r in rows],data['val']['common_support'].sum(1)/data['val']['mask'].sum(1)),f'{path}: support fraction')
    return dict(backbone=b,seed=seed,arm=arm,selected_epoch=selected,adapter_parameters=result['adapter_parameters'],
                **actual,identity_checks=identity,initial_adapter_state_sha256=result['initial_adapter_state_sha256'],
                batch_order_sha256=order,epoch_batch_order_sha256=per,frozen_state_sha256=expected_frozen,
                independently_computed_epoch_macro_AP=independently_computed,checkpoint_replay_max_absolute_error=0.,
                result_sha256=sha(path/'result.json'),checkpoint_sha256=sha(path/'checkpoint.pt'),
                val_predictions_sha256=sha(path/'val_predictions.csv'),validation_strata=strata(data['val'],scores,threshold))


def summarize(results,protocol):
    table=[];contrasts=[];index={(r['backbone'],r['seed'],r['arm']):r for r in results}
    comparisons=[('self_quality','mean_context'),('self_quality','affine_context'),('self_quality','reliable_context'),('mean_context','reliable_context'),('affine_context','reliable_context')]
    for b in protocol['backbones']:
        for arm in protocol['arms']:
            rr=[index[b,s,arm] for s in protocol['training']['seeds'] if (b,s,arm) in index]
            table.append(dict(backbone=b,arm=arm,seeds=[r['seed'] for r in rr],selected_epoch0_count=sum(r['selected_epoch']==0 for r in rr),
                selected_epochs=[r['selected_epoch'] for r in rr],
                selected={m:distribution([r['validation']['macro'][m] for r in rr]) for m in METRICS},
                baseline={m:distribution([r['baseline_validation']['macro'][m] for r in rr]) for m in METRICS},
                final_epoch={m:distribution([r['final_epoch_validation']['macro'][m] for r in rr]) for m in METRICS},
                selected_minus_baseline={m:distribution([r['validation']['macro'][m]-r['baseline_validation']['macro'][m] for r in rr]) for m in METRICS},
                final_epoch_minus_baseline={m:distribution([r['final_epoch_validation']['macro'][m]-r['baseline_validation']['macro'][m] for r in rr]) for m in METRICS}))
        for fr,to in comparisons:
            paired=[(s,index[b,s,fr],index[b,s,to]) for s in protocol['training']['seeds'] if (b,s,fr) in index and (b,s,to) in index]
            contrasts.append(dict(backbone=b,from_arm=fr,to_arm=to,seeds=[s for s,_,_ in paired],
                selected={m:distribution([v['validation']['macro'][m]-u['validation']['macro'][m] for _,u,v in paired]) for m in METRICS},
                final_epoch={m:distribution([v['final_epoch_validation']['macro'][m]-u['final_epoch_validation']['macro'][m] for _,u,v in paired]) for m in METRICS}))
    return dict(all_configurations=table,paired_contrasts=contrasts,
                epoch0_selections=sum(r['selected_epoch']==0 for r in results),
                selection_scope='Epoch0 is eligible: selected validation AP not below its paired baseline is a selection property, not an architectural or generalization guarantee.',
                score_identity_scope='Identity assertions concern logits/scores. A separately reselected validation F1 threshold may change hard classifications even when scores are identical.',
                control_scope='self_quality includes the shared support/availability policy, not a strictly context-blind extra network. Nominal parameter counts match, inactive slots do not establish equal effective capacity.',
                causal_scope='Frozen pretrained pooled adapters versus earlier from-scratch early concat differ in more than fusion position; interpret current paired arms within this stage.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args();require(not args.out.exists(),'Audit output exists')
    protocol=json.loads((args.stage/'protocol.json').read_text());ph=sha(args.stage/'protocol.json')
    require(protocol['splits_read']==['train','val'],'Forbidden split scope')
    data,proof=load_inputs(args.parent,protocol)
    scales={arm:fit_adapter_scaler(data['train']['features'][arm],data['train']['common_support']) for arm in protocol['arms']}
    spec=importlib.util.spec_from_file_location('frozen_backbone_definitions',args.parent/'training/vendor/pilot_train_frozen.py')
    vendor=importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    results=[];missing=[];baseline_proofs=[]
    for b in protocol['backbones']:
        worker=args.stage/'runs'/b
        if not (worker/'configuration.json').exists() and args.allow_partial:
            missing.extend(f'{b}/{s}/{a}' for s in protocol['training']['seeds'] for a in protocol['arms']);continue
        conf_hash=check_worker(worker,args.parent,protocol,ph,args.stage/'training',proof,data)
        for seed in protocol['training']['seeds']:
            existing=[a for a in protocol['arms'] if (worker/f'seed_{seed}'/a/'completion.json').exists()]
            if not existing:
                missing.extend(f'{b}/{seed}/{a}' for a in protocol['arms']);continue
            model,replay,parent_proof,parent_cp=verify_parent_cache(args.parent,args.stage,b,seed,data,vendor,128)
            baseline_proofs.append(dict(backbone=b,seed=seed,**parent_proof))
            for arm in protocol['arms']:
                folder=worker/f'seed_{seed}'/arm
                if arm not in existing:missing.append(f'{b}/{seed}/{arm}');continue
                result=audit_run(folder,b,seed,arm,data,scales[arm],protocol,ph,conf_hash,model,replay,parent_proof,parent_cp)
                results.append(result);print('AUDITED',b,seed,arm,flush=True)
    require(args.allow_partial or not missing,f'Missing runs: {missing}')
    for b in protocol['backbones']:
        for s in protocol['training']['seeds']:
            rr=[r for r in results if r['backbone']==b and r['seed']==s]
            for k in ('initial_adapter_state_sha256','batch_order_sha256','frozen_state_sha256','adapter_parameters'):
                require(len({r[k] for r in rr})<=1,f'{b}/{s}: unpaired {k}')
    summary=summarize(results,protocol)
    summary.update(status='partial' if missing else 'passed',completed_runs=len(results),expected_runs=36,
                   protocol_sha256=ph,missing_runs=missing,availability=availability(data))
    args.out.mkdir(parents=True)
    jwrite(args.out/'independent_adapter_audit.json',dict(status=summary['status'],protocol_sha256=ph,source_files=proof,baseline_proofs=baseline_proofs,runs=results))
    jwrite(args.out/'adapter_summary.json',summary)
    rows=[]
    for r in results:
        rows.append({k:r[k] for k in ('backbone','seed','arm','selected_epoch')}|
                    {f'{scope}_{m}':r[key]['macro'][m] for scope,key in [('selected','validation'),('baseline','baseline_validation'),('epoch12','final_epoch_validation')] for m in METRICS})
    with (args.out/'per_seed_validation_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['backbone']);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(status=summary['status'],runs=len(results),out=str(args.out))),flush=True)


if __name__=='__main__':main()
