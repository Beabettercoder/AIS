"""Read-only independent completion, selection, frozen-model and data audit."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math
import numpy as np
from tqdm.auto import tqdm

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_reanchor_run_20261106/work_v2'))
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();root=a.root;base=root.parent
    issues=[];checks=collections.Counter();experiments=[];run_rows=[]
    def check(condition,label,detail=None):
        checks[label]+=1
        if not bool(condition):issues.append(dict(check=label,detail=detail))
    plan=read(root/'plan.json');progress=read(root/'progress.json')
    check(progress['stage']=='complete','pipeline_completed')
    check(len(plan['experiments'])==5,'five_declared_experiments')
    for relative,digest in plan['code_hashes'].items():check(sha(base/relative)==digest,'predeclared_code_hash',relative)
    data_cache={};hash_cache={}
    def cached_sha(p):
        key=str(p)
        if key not in hash_cache:hash_cache[key]=sha(p)
        return hash_cache[key]
    def get_data(directory):
        key=str(directory)
        if key not in data_cache:
            data_cache[key]={s:load(directory/f'{s}.npz') for s in ['train','val','test']}
            for s in ['train','val','test']:
                manifest=read(directory/f'{s}_manifest.json')
                check(cached_sha(directory/f'{s}.npz')==manifest['npz_sha256'],'data_manifest_hash',f'{directory}/{s}')
            ids={s:set(map(int,z['mmsi'])) for s,z in data_cache[key].items()}
            check(not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']),'target_mmsi_disjoint',key)
        return data_cache[key]
    all_normalizers=set();selection_counts=collections.Counter()
    for spec in plan['experiments']:
        name=spec['name'];exp=root/name;metadata=read(exp/'experiment.json');data_dir=Path(metadata['data_dir']);data=get_data(data_dir)
        freeze=read(exp/'freeze.json');freeze_hash=sha(exp/'freeze.json');frozen={(r['model'],r['seed'],r['arm']):r for r in freeze['models']}
        check(len(frozen)==48,'freeze_model_count',name)
        check(freeze['code_hashes']==plan['code_hashes'],'freeze_matches_predeclared_code',name)
        norms=set();init_groups=collections.defaultdict(set);param_groups=collections.defaultdict(set);run_count=0;early=0
        for model in spec['models']:
            for seed in spec['seeds']:
                for arm in ['P0','P1','P2','P3']:
                    folder=exp/f'runs/{model}/seed_{seed}/{arm}';r=read(folder/'result.json');pr=read(folder/'protocol.json')
                    test_dir=exp/f'test_runs/{model}/seed_{seed}/{arm}';tested=read(test_dir/'result.json');eval_plan=read(test_dir/'evaluation_plan.json')
                    history=list(csv.DictReader((folder/'history.csv').open()));epochs=spec['itransformer_epochs'] if model=='itransformer' else spec['epochs']
                    key=f'{name}/{model}/s{seed}/{arm}';run_count+=1
                    check(r['status']==tested['status']=='complete','run_and_test_complete',key)
                    check(len(history)==epochs==r['completed_epochs']==r['expected_epochs'],'full_budget',key)
                    check([int(x['epoch']) for x in history]==list(range(1,epochs+1)),'contiguous_history',key)
                    minimum=min(history,key=lambda x:float(x['val_ADE_m']));best_epoch=int(minimum['epoch'])
                    check(best_epoch==r['best_val']['epoch'],'base_validation_argmin',key)
                    check(float(minimum['val_ADE_m'])==r['best_val']['ADE_m'],'best_val_metric',key)
                    check(float(history[-1]['val_ADE_m'])==r['last_epoch']['ADE_m'],'last_epoch_metric',key)
                    expected=best_epoch if spec['policy']=='cosine_best' else epochs
                    check(r['selected_epoch']==tested['selected_epoch']==expected,'selected_epoch_policy',key)
                    selected=Path(r['selected_checkpoint']);selected_sha=cached_sha(selected)
                    f=frozen[model,seed,arm]
                    check(selected_sha==r['selected_checkpoint_sha256']==tested['selected_checkpoint_sha256']==f['selected_checkpoint_sha256'],'selected_checkpoint_hash',key)
                    check(Path(f['selected_checkpoint'])==selected,'frozen_selected_path',key)
                    for choice in ['best_val','last_epoch']:
                        checkpoint=Path(r[choice]['checkpoint']);check(cached_sha(checkpoint)==r[choice]['sha256'],choice+'_checkpoint_hash',key)
                    check(eval_plan['model_selected_sha256']==selected_sha and eval_plan['freeze_sha256']==freeze_hash,'test_uses_frozen_selected_checkpoint',key)
                    check(eval_plan['data_sha256']==cached_sha(data_dir/'test.npz'),'test_data_hash',key)
                    check(tested['evaluation_identity']==eval_plan,'test_record_matches_evaluation_plan',key)
                    norm_hash=cached_sha(folder/'normalizer.json');norms.add(norm_hash);all_normalizers.add(norm_hash)
                    check(eval_plan['norm_sha256']==norm_hash==freeze['norm_sha256']==metadata['norm_sha256'],'frozen_scaler_hash',key)
                    check(pr['initial_model_hash']==r['initial_model_hash'],'initial_weight_record',key)
                    init_groups[model,seed].add(r['initial_model_hash']);param_groups[model,seed].add(r['parameter_count'])
                    lr=[float(x['lr']) for x in history];start=spec['learning_rate']
                    expected_lr=[start*.01+(start-start*.01)*(1+math.cos(math.pi*j/epochs))/2 for j in range(epochs)] if spec['policy']=='cosine_best' else [start]*epochs
                    check(np.allclose(lr,expected_lr,rtol=1e-11,atol=1e-14),'learning_rate_schedule',key)
                    check(r['settings']['weight_decay']==.0001 and r['settings']['gradient_clip']==1.,'regularization_config',key)
                    epoch_files=list((folder/'weights').glob('epoch_*.pt'));check(len(epoch_files)==epochs,'per_epoch_weights_saved',key)
                    for split,where,prefix,expected_epoch,metrics in [
                            ('val',folder,'predictions',expected,r['views']),('val',folder,'last_epoch_predictions',epochs,r['last_epoch_views']),
                            ('test',test_dir,'predictions',expected,tested['views'])]:
                        base_pred=None
                        for view in ['base','target_jump','target_drift','target_speed','neighbor_stress']:
                            z=load(where/f'{prefix}_{view}.npz');reference=data[split]
                            check(str(z['split'])==split and int(z['selected_epoch'])==expected_epoch,'prediction_split_epoch',key+'/'+prefix+'/'+view)
                            for field in ['window_id','mmsi','region','future_xy','query_s','maneuver']:
                                check(np.array_equal(z[field],reference[field]),'prediction_reference_alignment',key+'/'+prefix+'/'+view+'/'+field)
                            err=np.linalg.norm(z['pred_xy']-reference['future_xy'],axis=-1)
                            check(np.isfinite(err).all() and np.allclose(z['ADE'],err.mean(1),rtol=0,atol=1e-9) and np.allclose(z['FDE'],err[:,-1],rtol=0,atol=1e-9),'prediction_error_recalculation',key+'/'+prefix+'/'+view)
                            check(np.isclose(err.mean(),metrics[view]['ADE_m'],rtol=0,atol=1e-9) and np.isclose(err[:,-1].mean(),metrics[view]['FDE_m'],rtol=0,atol=1e-9),'result_mean_matches_predictions',key+'/'+prefix+'/'+view)
                            if view=='base':base_pred=z['pred_xy'].copy()
                            if view=='neighbor_stress' and arm in ['P0','P1']:check(np.array_equal(base_pred,z['pred_xy']),'self_only_neighbor_invariance',key+'/'+prefix)
                    early+=expected<epochs;selection_counts[spec['policy']]+=1
                    run_rows.append(dict(experiment=name,model=model,seed=seed,arm=arm,policy=spec['policy'],hidden=r['hidden'],
                                         selected_epoch=expected,budget=epochs,best_val_ADE_m=r['best_val']['ADE_m'],last_epoch_ADE_m=r['last_epoch']['ADE_m'],checkpoint_sha256=selected_sha))
        check(run_count==48,'actual_model_count',name)
        for group,values in init_groups.items():check(len(values)==1,'arm_initial_weights_equal',name+str(group))
        for group,values in param_groups.items():check(len(values)==1,'arm_parameter_counts_equal',name+str(group))
        check(len(norms)==1,'arm_normalizers_equal',name)
        for split in ['val','test']:
            summary=read(exp/f'reports/{split}/summary.json');check(summary['status']=='complete' and summary['completed_models']==48,'summary_complete',name+'/'+split)
        experiments.append(dict(name=name,models=run_count,policy=spec['policy'],hidden=spec['hidden'],selected_before_last=early,
            data_dir=str(data_dir),counts={s:len(z['window_id']) for s,z in data.items()},mmsi_counts={s:len(np.unique(z['mmsi'])) for s,z in data.items()}))
        print('Audited',name,run_count,flush=True)
    check(len(all_normalizers)==1,'all_experiment_normalizers_equal')
    small=get_data(Path(read(root/'data_small/experiment.json')['data_dir']));large=get_data(Path(read(root/'main/experiment.json')['data_dir']))
    for split in ['val','test']:
        check(set(small[split])==set(large[split]),'evaluation_cache_keys_equal',split)
        for key in small[split]:check(np.array_equal(small[split][key],large[split][key]),'shared_evaluation_cache_exact',split+'/'+key)
    order={str(w):i for i,w in enumerate(large['train']['window_id'])};small_ids=small['train']['window_id']
    check(all(str(w) in order for w in small_ids),'small_training_subset_of_large')
    take=np.array([order[str(w)] for w in small_ids])
    for key in small['train']:check(np.array_equal(small['train'][key],large['train'][key][take]),'nested_training_values_exact',key)
    a.out.mkdir(parents=True,exist_ok=True)
    result=dict(status='pass' if not issues else 'failed',completed_training_runs=len(run_rows),completed_test_runs=len(run_rows),
        experiments=experiments,selection_counts=dict(selection_counts),checks=dict(checks),check_count=sum(checks.values()),issues=issues,
        data_shared_test_windows=len(large['test']['window_id']),data_shared_val_windows=len(large['val']['window_id']),
        test_sources_previously_used=True,scope='Read-only audit; no training, checkpoint edits or score selection performed.',script_sha256=sha(__file__))
    (a.out/'independent_completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (a.out/'checkpoint_selection_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,list(run_rows[0]));wr.writeheader();wr.writerows(run_rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
