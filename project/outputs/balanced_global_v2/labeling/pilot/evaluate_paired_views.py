"""Frozen-checkpoint cross-generation and supported group-variation evaluation.

No optimizer, scaler fit, threshold selection, or test-based model selection is
performed. Whole test views are checked for matching semantics before evaluating
the exact source IDs recorded in the frozen pilot subset CSVs.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys,time
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
import pilot_train as training

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/labeling/labeled_v1')
FEATURES=('raw','plain_context','plain_affine_context','context')
BACKBONES=('tcn','gru','transformer')
VIEWS=('features_scene','features_hard_scene','paired_group_scene')
STRATA=('ge4','one_to_three','zero')


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def require(ok,message):
 if not ok:raise ValueError(message)


def read_csv(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def write_csv(path,records):
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def support_stratum(count):
 return 'ge4' if count>=4 else ('one_to_three' if count else 'zero')


def safe_metrics(y,p,threshold):
 y=np.asarray(y,dtype=int);p=np.asarray(p,float);n=len(y)
 if not n:return dict(samples=0,positive_samples=0,no_behavior_samples=0,AP=None,AUROC=None,F1=None,no_behavior_FPR=None,reason='no_samples')
 pred=p>=threshold;positive=int(y.sum());negative=n-positive
 require(np.isfinite(p).all(),'Nonfinite probability')
 # Empty and single-class strata are explicit. AP/F1 require positive cases;
 # AUROC requires both classes. Nominal FPR requires at least one y=0 case.
 return dict(samples=n,positive_samples=positive,no_behavior_samples=negative,
             AP=float(average_precision_score(y,p)) if positive else None,
             AUROC=float(roc_auc_score(y,p)) if positive and negative else None,
             F1=float(f1_score(y,pred,zero_division=0)) if positive else None,
             no_behavior_FPR=float(np.mean(pred[y==0])) if negative else None,
             false_positives=int(np.sum(pred&(y==0))),
             prevalence=float(y.mean()),single_class=not (positive and negative))


def domain_summary(y,p,region,threshold,selected=None,all_regions=None):
 if selected is None:selected=np.ones(len(y),bool)
 regions=list(all_regions) if all_regions is not None else sorted(set(region))
 per={}
 for r in regions:
  use=selected&(region==r);per[r]=safe_metrics(y[use],p[use],threshold)
 macro={};contributors={}
 for key in ['AP','AUROC','F1','no_behavior_FPR']:
  values=[v[key] for v in per.values() if v[key] is not None]
  macro[key]=float(np.mean(values)) if values else None;contributors[key]=len(values)
 return {'per_region':per,'macro':macro,'macro_contributing_domains':contributors,
         'pooled':safe_metrics(y[selected],p[selected],threshold),
         'missing_domain_policy':'Undefined/empty metrics remain null and are excluded, never replaced by zero'}


def metadata_compact(path,expected):
 found={}
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   md=json.loads(line);w=str(md['window_id'])
   require(w not in found,'Repeated metadata ID')
   require(w in expected,'Metadata not present in NPZ')
   neighbors=md['neighbor_observation_noise']
   # Context exposure can legitimately differ with changed target trajectories;
   # the underlying corruption plan itself must remain matched.
   neighbor_plan={k:neighbors.get(k) for k in ['requested','candidate_mmsi','selected_mmsi','changes','effective','no_target_fallback']}
   group=md['group_variation'];controlled=list(map(str,group.get('controlled_mmsi',[])))
   require(len(controlled)==len(set(controlled)),'Duplicate controlled group member')
   found[w]={'base_window_id':md['base_window_id'],'mmsi':str(md['target_mmsi']),'y':int(md['y']),
             'type':int(md['type']),'behavior_type':md['behavior_type'],
             'counterfactual_row':int(md['counterfactual_row']),
             'behavior_generation':{k:md['behavior_generation'].get(k) for k in ['kind','applied_params','is_sham']},
             'requested_observation_noise':md['requested_observation_noise'],
             'target_noise_plan':{k:md['target_observation_noise'].get(k) for k in ['noise','parameters','requested_for_target']},
             'neighbor_noise_plan':neighbor_plan,
             'controlled_count':len(controlled),'controlled_mmsi':controlled,
             'group_requested':bool(group.get('requested')),
             'group_factor':group.get('factor'),'group_onset_s':group.get('onset_s'),
             'view':md['view']}
 require(set(found)==set(expected),'Missing metadata samples')
 return found


def load_and_validate_views(args,subset,regions):
 caches={v:[] for v in VIEWS};proof=[];checks={};n_total=0
 common=('window_id','base_window_id','mmsi','y','type','inttype','behavior_type','noise_type','noise_scope','observation_noise','mask','time_s')
 for region in regions:
  wanted=[r for r in subset if r['region']==region and r['split']=='test']
  require(len(wanted)==80,f'{region}: expected the frozen 80 test windows')
  wanted_ids=[r['window_id'] for r in wanted];require(len(set(wanted_ids))==80,'Repeated pilot test ID')
  raw={};meta={};maps={};source_proofs={}
  for view in VIEWS:
   path=args.release/view/region/'test.npz';report_path=path.parent/'feature_report.json';report=json.loads(report_path.read_text())
   audit_path=path.parent/'test_context_audit.jsonl.gz'
   h=sha(path);ah=sha(audit_path)
   require(report['status']=='completed' and report['view']==view and report['region']==region,'Incomplete feature view')
   require(h==report['output_hashes']['test']['npz_sha256'],'View NPZ hash mismatch')
   require(ah==report['output_hashes']['test']['audit_sha256'],'View metadata hash mismatch')
   source_proofs[view]=report['source_proof']
   require(report['source_proof']['labels_sha256']==sha(args.release/'label_manifest.parquet'),'Feature labels differ from frozen manifest')
   with np.load(path,allow_pickle=False) as z:raw[view]={k:z[k] for k in z.files}
   ids=raw[view]['window_id'].astype(str);require(len(ids)==len(set(ids)),'Repeated feature window')
   maps[view]={w:i for i,w in enumerate(ids)}
   require(set(wanted_ids)<=set(ids),'Frozen subset missing in view')
   meta[view]=metadata_compact(audit_path,set(ids))
   proof.append({'view':view,'region':region,'npz_path':str(path),'npz_sha256':h,'audit_path':str(audit_path),'audit_sha256':ah,'feature_report_sha256':sha(report_path),'source_proof':report['source_proof']})
  require(source_proofs['features_scene']==source_proofs['features_hard_scene']==source_proofs['paired_group_scene'],'Matched views do not use the same original sources/code/label version')
  base=raw['features_scene'];ids=base['window_id'].astype(str);mask=training.validate_mask(base['mask']);n=len(ids)
  for view in VIEWS[1:]:
   other=raw[view];require(set(maps[view])==set(ids),'Different full test IDs between views')
   align=np.array([maps[view][w] for w in ids])
   for key in common:require(np.array_equal(base[key],other[key][align]),f'Unmatched {region}/{view}/{key}')
  hard=raw['features_hard_scene'];hard_order=np.array([maps['features_hard_scene'][w] for w in ids]);negative=base['y']==0
  for key in ['target_pos','target_sog_kn','target_cog_deg','pre_noise_pos','pre_noise_sog_kn','pre_noise_cog_deg']:
   require(np.array_equal(base[key][negative],hard[key][hard_order][negative]),f'Nominal target differs in smooth/hard {region}/{key}')
  for key in FEATURES:
   require(np.array_equal(base[key][negative],hard[key][hard_order][negative]),f'Nominal features differ in matched controls {region}/{key}')
  for w in ids:
   b=meta['features_scene'][w];h=meta['features_hard_scene'][w];g=meta['paired_group_scene'][w]
   for other in [h,g]:
    for key in ['base_window_id','mmsi','y','type','behavior_type','counterfactual_row','behavior_generation','requested_observation_noise','neighbor_noise_plan']:
     require(b[key]==other[key],f'Changed source/type/noise intervention {region}/{w}/{key}')
    require(b['target_noise_plan']==other['target_noise_plan'],f'Changed target noise plan {region}/{w}')
   require(not b['group_requested'] and not h['group_requested'] and g['group_requested'],'Group view request semantics mismatch')
  for view in VIEWS:
   d=raw[view];idx=np.array([maps[view][w] for w in wanted_ids]);part={k:d[k][idx] for k in ['window_id','base_window_id','mmsi','y','mask','time_s','behavior_type','noise_type','noise_scope','observation_noise']}
   for k in FEATURES:
    part[k]=d[k][idx].astype(np.float32);require(np.isfinite(part[k]).all(),'Nonfinite test features')
    require(np.all(part[k][~part['mask']]==0),'Test padding changed')
   part['region']=np.repeat(region,len(idx));part['type']=d['type'][idx]
   part['group_controlled_count']=np.array([meta['paired_group_scene'][w]['controlled_count'] for w in wanted_ids],dtype=np.int64)
   part['group_support_stratum']=np.array([support_stratum(n) for n in part['group_controlled_count']])
   for i,row in enumerate(wanted):
    require(int(part['y'][i])==int(row['y']) and str(part['mmsi'][i])==row['mmsi'] and str(part['base_window_id'][i])==row['base_window_id'],'Frozen pilot IDs/labels/MMSI changed')
   caches[view].append(part)
  checks[region]={'full_test_windows_verified':n,'frozen_test_windows_evaluated':len(wanted),'no_behavior_samples_target_and_features_equal':int(negative.sum()),'group_support_in_subset':{s:int(np.sum(caches['features_scene'][-1]['group_support_stratum']==s)) for s in STRATA}}
  n_total+=n;print('MATCHED_VIEWS_VERIFIED',region,n,flush=True)
 datasets={view:{k:np.concatenate([p[k] for p in parts]) for k in parts[0]} for view,parts in caches.items()}
 require(len(datasets['features_scene']['y'])==80*len(regions),'Wrong frozen test size')
 return datasets,{'status':'passed','scope':'Whole three test views checked before evaluating frozen pilot subset','full_test_windows_verified':n_total,'subset_test_windows':80*len(regions),'regions':checks,'source_files':proof}


def frozen_subsets(path):
 rows=read_csv(path);require(rows and {r['split'] for r in rows}=={'train','val','test'},'Bad pilot subset file')
 key=lambda r:(r['region'],r['split'],r['window_id'],r['base_window_id'],r['mmsi'],int(r['y']))
 keys=[key(r) for r in rows];require(len(keys)==len(set(keys)),'Repeated pilot subset item')
 return rows,sorted(keys)


def load_model(pilot,kind,feature):
 cfg=json.loads((pilot/'pilot_configuration.json').read_text())
 require(cfg['script_sha256']==sha(Path(training.__file__)),'Model definition changed after fitting')
 folder=pilot/f'{kind}__{feature}';metrics=json.loads((folder/'metrics.json').read_text());cp=folder/'checkpoint.pt'
 require(metrics['backbone']==kind and metrics['feature_group']==feature,'Misidentified checkpoint directory')
 require(sha(cp)==metrics['output_hashes']['checkpoint.pt'],'Checkpoint hash mismatch')
 state=torch.load(cp,map_location='cpu',weights_only=True)
 require(state['backbone']==kind and state['input_channels']==training.DIMENSIONS[feature],'Checkpoint architecture/input differs')
 require(float(state['threshold'])==float(metrics['threshold']),'Frozen threshold differs from source validation result')
 norm={'mean':state['normalizer_mean'].cpu().numpy(),'std':state['normalizer_std'].cpu().numpy()}
 with np.load(pilot/'training_normalizers.npz',allow_pickle=False) as z:
  require(np.array_equal(norm['mean'],z[feature+'_mean']) and np.array_equal(norm['std'],z[feature+'_std']),'Checkpoint normalizer changed')
 model=training.BackboneClassifier(kind,state['input_channels'],state['hidden'],state['dropout']);model.load_state_dict(state['state_dict']);model.eval()
 return model,norm,float(state['threshold']),cfg,{'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'normalizers_sha256':sha(pilot/'training_normalizers.npz'),'metrics_sha256':sha(folder/'metrics.json'),'pilot_configuration_sha256':sha(pilot/'pilot_configuration.json'),'threshold':float(state['threshold']),'train_normalizer_mean':norm['mean'].tolist(),'train_normalizer_std':norm['std'].tolist()}


def pilot_source_check(pilot,expected_data):
 cfg=json.loads((pilot/'pilot_configuration.json').read_text())
 require(Path(cfg['data']).resolve()==expected_data.resolve(),'Pilot was trained on a different source view')
 verified=[]
 for item in cfg['source_files']:
  p=Path(item['path']);actual=sha(p);require(actual==item['sha256'],'Fitting source file changed after checkpoint creation')
  verified.append({'path':str(p),'sha256':actual,'split':item['split'],'region':item['region']})
 return cfg,verified


def predict_one(model,norm,dataset,feature,batch):
 x=training.normalized(dataset[feature],dataset['mask'],norm)
 return training.predict(model,torch.from_numpy(x),torch.from_numpy(dataset['mask']),batch)


def cached_source_prediction_check(pilot,kind,feature,dataset,scores):
 p=pilot/f'{kind}__{feature}'/'test_predictions.csv';rows=read_csv(p)
 previous={r['window_id']:r for r in rows};require(set(previous)==set(dataset['window_id']),'Source prediction subset differs')
 errors=[]
 for i,w in enumerate(dataset['window_id']):
  row=previous[str(w)];require(int(row['y'])==int(dataset['y'][i]),'Saved source labels differ')
  errors.append(abs(float(row['score'])-scores[i]))
 require(max(errors)<3e-6,'Frozen diagonal inference failed to reproduce saved source predictions')
 return {'saved_predictions_sha256':sha(p),'replay_max_abs_error':float(max(errors))}


def prediction_rows(dataset,scores,threshold,train_view,test_view,kind,feature):
 rows=[]
 for i,p in enumerate(scores):
  rows.append({'train_view':train_view,'test_view':test_view,'backbone':kind,'feature_group':feature,'region':str(dataset['region'][i]),'window_id':str(dataset['window_id'][i]),'base_window_id':str(dataset['base_window_id'][i]),'mmsi':str(dataset['mmsi'][i]),'y':int(dataset['y'][i]),'type':int(dataset['type'][i]),'behavior_type':str(dataset['behavior_type'][i]),'observation_noise':bool(dataset['observation_noise'][i]),'noise_type':str(dataset['noise_type'][i]),'noise_scope':str(dataset['noise_scope'][i]),'group_controlled_count':int(dataset['group_controlled_count'][i]),'group_support_stratum':str(dataset['group_support_stratum'][i]),'score':float(p),'frozen_validation_threshold':threshold,'prediction':int(p>=threshold)})
 return rows


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--release',type=Path,default=ROOT);ap.add_argument('--smooth-pilot',type=Path,default=ROOT/'pilot_scene_seed0');ap.add_argument('--hard-pilot',type=Path,default=ROOT/'pilot_hard_seed0');ap.add_argument('--out',type=Path,default=ROOT/'paired_view_evaluation_seed0');ap.add_argument('--threads',type=int,default=4);args=ap.parse_args()
 if args.out.exists() and any(args.out.iterdir()):raise SystemExit('Refusing to overwrite an existing paired evaluation')
 args.out.mkdir(parents=True,exist_ok=True);(args.out/'predictions').mkdir();started=time.time()
 require(1<=args.threads<=4,'CPU threads must be1..4');torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 smooth_rows,smooth_keys=frozen_subsets(args.smooth_pilot/'pilot_subset.csv');_,hard_keys=frozen_subsets(args.hard_pilot/'pilot_subset.csv')
 require(smooth_keys==hard_keys,'Smooth/hard fitting subsets were not matched')
 cfg,smooth_sources=pilot_source_check(args.smooth_pilot,args.release/'features_scene')
 hard_cfg,hard_sources=pilot_source_check(args.hard_pilot,args.release/'features_hard_scene')
 for key in ['epochs','hidden','dropout','lr','weight_decay','clip_grad_norm','batch_size','train_per_region','val_per_region','test_per_region','seed']:
  require(cfg['configuration'][key]==hard_cfg['configuration'][key],'Smooth/hard fitting budgets differ: '+key)
 regions=cfg['configuration']['regions'];require(regions==hard_cfg['configuration']['regions'],'Pilot source regions differ')
 datasets,validation=load_and_validate_views(args,smooth_rows,regions)
 validation['subset_hashes']={'smooth':sha(args.smooth_pilot/'pilot_subset.csv'),'hard':sha(args.hard_pilot/'pilot_subset.csv')}
 (args.out/'paired_view_validation.json').write_text(json.dumps(validation,indent=2))
 matrix={'status':'completed_frozen_evaluation','input_groups':['raw','context'],'subset_windows':len(datasets['features_scene']['y']),'threshold_policy':'Each source model retains its original validation-selected threshold for every target view','normalization_policy':'Each source model retains its own TRAIN-only mean/std; no target refit','train_axis':['smooth','hard'],'test_axis':['smooth','hard'],'cells':{}}
 group={'status':'completed_frozen_evaluation','scope':'Coherent controlled group slowdown, same source IDs and unchanged behavior labels; y=0 means no injected individual behavior','support_strata':{'ge4':'>=4 effective controlled neighbor MMSI','one_to_three':'1-3 effective controlled neighbor MMSI','zero':'0 effective controlled neighbor MMSI'},'baseline_comparison':'Baseline and stress are evaluated on the identical support-stratified source IDs, using the same source checkpoint/normalizer/threshold','models':{}}
 cache={};provenance={};source_replays={};prediction_files={}
 def evaluate(train_view,kind,feature,test_view):
  key=(train_view,kind,feature,test_view)
  if key in cache:return cache[key]
  pilot=args.smooth_pilot if train_view=='smooth' else args.hard_pilot
  model,norm,threshold,pc,proof=load_model(pilot,kind,feature);dataset=datasets[test_view]
  batch=int(pc['configuration']['batch_size']);scores=predict_one(model,norm,dataset,feature,batch)
  if (train_view=='smooth' and test_view=='features_scene') or (train_view=='hard' and test_view=='features_hard_scene'):
   source_replays['/'.join(key)]=cached_source_prediction_check(pilot,kind,feature,dataset,scores)
  path=args.out/'predictions'/('__'.join(key)+'.csv');write_csv(path,prediction_rows(dataset,scores,threshold,train_view,test_view,kind,feature))
  prediction_files[path.name]=sha(path);provenance['/'.join((train_view,kind,feature))]=proof
  result={'scores':scores,'threshold':threshold,'summary':domain_summary(dataset['y'],scores,dataset['region'],threshold,all_regions=regions)}
  cache[key]=result;print('FROZEN_EVAL','/'.join(key),json.dumps(result['summary']['macro']),flush=True);return result
 for feature in ['raw','context']:
  cells={}
  for train_view in ['smooth','hard']:
   cells[train_view]={}
   for test_name,test_view in [('smooth','features_scene'),('hard','features_hard_scene')]:
    r=evaluate(train_view,'tcn',feature,test_view);cells[train_view][test_name]={'threshold':r['threshold'],**r['summary']}
  matrix['cells'][feature]=cells
 for kind in BACKBONES:
  for feature in FEATURES:
   baseline=evaluate('smooth',kind,feature,'features_scene');stress=evaluate('smooth',kind,feature,'paired_group_scene');ds=datasets['paired_group_scene'];rec={'frozen_threshold':stress['threshold'],'all_samples':{'baseline':baseline['summary'],'stress':stress['summary']},'support_strata':{}}
   for stratum in STRATA:
    use=ds['group_support_stratum']==stratum
    b=domain_summary(ds['y'],baseline['scores'],ds['region'],stress['threshold'],use,regions);s=domain_summary(ds['y'],stress['scores'],ds['region'],stress['threshold'],use,regions)
    delta={k:float(s['macro'][k]-b['macro'][k]) if s['macro'][k] is not None and b['macro'][k] is not None else None for k in ['AP','AUROC','F1','no_behavior_FPR']}
    rec['support_strata'][stratum]={'baseline':b,'stress':s,'stress_minus_baseline_macro':delta}
   group['models'][kind+'__'+feature]=rec
 (args.out/'cross_generation_matrix.json').write_text(json.dumps(matrix,indent=2))
 (args.out/'group_variation_metrics.json').write_text(json.dumps(group,indent=2))
 proof={'status':'completed','script_sha256':sha(Path(__file__)),'pilot_model_definition_sha256':sha(Path(training.__file__)),'labels_manifest_sha256':sha(args.release/'label_manifest.parquet'),'matched_view_validation_sha256':sha(args.out/'paired_view_validation.json'),'verified_fitting_source_files':{'smooth':smooth_sources,'hard':hard_sources},'checkpoint_and_normalizer_sources':provenance,'source_diagonal_prediction_replays':source_replays,'prediction_csv_sha256':prediction_files,'weights_refitted':False,'thresholds_refitted':False,'normalizers_refitted':False,'test_labels_reassigned':False,'scope':'single-seed source-pilot checkpoint evaluations, not final multi-seed performance claims','versions':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'sklearn':__import__('sklearn').__version__},'elapsed_seconds':time.time()-started}
 (args.out/'evaluation_provenance.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps({'status':'completed','out':str(args.out),'prediction_files':len(prediction_files),'elapsed_seconds':time.time()-started},indent=2),flush=True)


if __name__=='__main__':main()
