"""Reaggregate frozen predictions by independently replayed numeric group effects.

prepare runs in the pandas/Parquet environment; reaggregate runs in the sklearn
environment. There are no Torch imports, checkpoints, model inference, fitting,
normalizer fitting, or threshold optimization in either stage.
"""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,time
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/labeling/labeled_v1')
GROUPS=('ge4','one_to_three','zero')
FEATURES=('raw','plain_context','plain_affine_context','context')
BACKBONES=('tcn','gru','transformer')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok,message):
 if not ok:raise ValueError(message)


def csv_rows(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def cohort(n):return 'ge4' if n>=4 else ('one_to_three' if n else 'zero')


def bool_value(value):
 if isinstance(value,(bool,np.bool_)):return bool(value)
 if isinstance(value,(int,np.integer)) and value in (0,1):return bool(value)
 if isinstance(value,str) and value.lower() in ('true','false'):return value.lower()=='true'
 raise ValueError('Invalid boolean value')


def prepare(args):
 import pandas as pd
 args.out.mkdir(parents=True,exist_ok=True)
 require(not (args.out/'numeric_effect_inputs.npz').exists(),'Prepared numeric inputs already exist')
 require(not (args.out/'numeric_group_variation_metrics.json').exists(),'Refusing to overwrite completed numeric evaluation')
 sidecar_hash=sha(args.sidecar);effects=pd.read_parquet(args.sidecar)
 required={'window_id','base_window_id','region','y','target_numeric_changed','neighbor_numeric_changed_count','neighbor_numeric_changed_mmsi','operator_member_count'}
 require(required<=set(effects.columns),'Missing independent numeric-effect columns')
 require(not effects.window_id.duplicated().any(),'Repeated window in independent sidecar')
 require(len(effects)==1680,'Expected a full eight-domain test sidecar')
 index={str(r.window_id):r for r in effects.itertuples()}
 original=json.loads((args.source/'evaluation_provenance.json').read_text());files=original['prediction_csv_sha256']
 require(len(files)==30,'Expected 30 frozen prediction CSV files')
 all_ids=None;reference=None
 for name,expected in files.items():
  path=args.source/'predictions'/name;require(sha(path)==expected,'Original prediction CSV changed: '+name)
  rows=csv_rows(path);require(len(rows)==640,'Unexpected source prediction sample size')
  ids={(r['window_id'],r['base_window_id'],r['mmsi'],int(r['y']),r['region']) for r in rows}
  require(len(ids)==640,'Repeated frozen prediction source')
  if all_ids is None:all_ids=ids;reference=rows
  else:require(ids==all_ids,'Different model prediction source IDs')
 records=[];all_counts={'target_changed':0,'target_unchanged':0};transitions={a:{b:0 for b in GROUPS} for a in GROUPS}
 for row in reference:
  w=row['window_id'];require(w in index,'Missing selected window in numeric replay sidecar');e=index[w]
  require(str(e.base_window_id)==row['base_window_id'] and str(e.region)==row['region'] and int(e.y)==int(row['y']),'Numeric replay uses different source/region/label')
  ids=json.loads(e.neighbor_numeric_changed_mmsi) if isinstance(e.neighbor_numeric_changed_mmsi,str) else list(e.neighbor_numeric_changed_mmsi)
  ids=list(map(str,ids));count=int(e.neighbor_numeric_changed_count);operator=int(e.operator_member_count)
  require(count>=0 and count==len(set(ids))==len(ids),'Wrong numeric neighbor identity count')
  require(operator>=count and operator==int(row['group_controlled_count']),'Operator/numeric replay does not match original group plan')
  changed=bool_value(e.target_numeric_changed);all_counts['target_changed' if changed else 'target_unchanged']+=1
  old=cohort(operator);new=cohort(count);transitions[old][new]+=1
  records.append({'window_id':w,'base_window_id':row['base_window_id'],'region':row['region'],'y':int(row['y']),
                  'target_numeric_changed':changed,'neighbor_numeric_changed_count':count,'numeric_support_stratum':new,
                  'operator_member_count':operator,'operator_support_stratum':old,'neighbor_numeric_changed_mmsi':json.dumps(ids)})
 payload={k:np.array([r[k] for r in records]) for k in records[0]}
 np.savez_compressed(args.out/'numeric_effect_inputs.npz',**payload)
 pd.DataFrame(records).to_parquet(args.out/'numeric_effect_pilot_cases.parquet',index=False)
 # Preserve a copy of the independently replayed full test sidecar for review.
 shutil.copy2(args.sidecar,args.out/'group_numeric_effects.parquet')
 cross=args.source/'cross_generation_matrix.json';shutil.copy2(cross,args.out/'cross_generation_matrix_unchanged_reference.json')
 report={'status':'passed','source_prediction_files_verified':len(files),'full_test_sidecar_rows':len(effects),'pilot_evaluation_windows':640,
         'sidecar_definition':'Pre-observation-noise numeric comparisons of position/speed/course; exact comparison tolerance=0; a requested/active operator is not sufficient',
         'sidecar_sha256':sidecar_hash,'sidecar_original_path':str(args.sidecar),'source_evaluation_provenance_sha256':sha(args.source/'evaluation_provenance.json'),
         'source_group_operator_report_sha256':sha(args.source/'group_variation_metrics.json'),'unchanged_cross_matrix_sha256':sha(cross),
         'prediction_csv_sha256':files,'numeric_inputs_sha256':sha(args.out/'numeric_effect_inputs.npz'),
         'target_numeric_change_counts':all_counts,'operator_to_numeric_support_transition':transitions,
         'preparer_script_sha256':sha(Path(__file__)),'original_reports_overwritten':False,'original_predictions_changed':False}
 replay_proof=args.sidecar.with_name('group_numeric_effects_verification.json')
 if replay_proof.exists():report['sidecar_independent_replay_verification_sha256']=sha(replay_proof)
 require(sha(args.sidecar)==sidecar_hash,'Numeric sidecar changed while preparing')
 (args.out/'numeric_effect_input_validation.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'status':'prepared','target_numeric_change_counts':all_counts,'transitions':transitions},indent=2),flush=True)


def safe_metrics(y,p,threshold):
 from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
 n=len(y)
 if not n:return {'samples':0,'positive_samples':0,'no_behavior_samples':0,'AP':None,'AUROC':None,'F1':None,'no_behavior_FPR':None,'reason':'no_samples'}
 pos=int(y.sum());neg=n-pos;pred=p>=threshold
 return {'samples':n,'positive_samples':pos,'no_behavior_samples':neg,
         'AP':float(average_precision_score(y,p)) if pos else None,
         'AUROC':float(roc_auc_score(y,p)) if pos and neg else None,
         'F1':float(f1_score(y,pred,zero_division=0)) if pos else None,
         'no_behavior_FPR':float(np.mean(pred[y==0])) if neg else None,
         'false_positives':int(np.sum(pred&(y==0))),'single_class':not(pos and neg)}


def summarize(y,p,regions,threshold,use,all_regions):
 per={r:safe_metrics(y[use&(regions==r)],p[use&(regions==r)],threshold) for r in all_regions}
 macro={};n={}
 for metric in ['AP','AUROC','F1','no_behavior_FPR']:
  values=[d[metric] for d in per.values() if d[metric] is not None]
  macro[metric]=float(np.mean(values)) if values else None;n[metric]=len(values)
 return {'per_region':per,'macro':macro,'macro_contributing_domains':n,'pooled':safe_metrics(y[use],p[use],threshold)}


def reaggregate(args):
 import sklearn
 started=time.time();require(not (args.out/'numeric_group_variation_metrics.json').exists(),'Refusing to overwrite completed evaluation')
 check=json.loads((args.out/'numeric_effect_input_validation.json').read_text());require(check['status']=='passed','Numeric input preparation failed')
 require(sha(args.out/'numeric_effect_inputs.npz')==check['numeric_inputs_sha256'],'Numeric cohort array changed')
 require(sha(args.sidecar)==check['sidecar_sha256'],'Independent replay sidecar changed')
 with np.load(args.out/'numeric_effect_inputs.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
 regions=data['region'];y=data['y'].astype(int);changed=data['target_numeric_changed'].astype(bool);support=data['neighbor_numeric_changed_count'].astype(int)
 all_regions=sorted(set(regions));primary=changed&(support>=4)
 report={'status':'completed_prediction_reaggregation','version':'numeric_effect_evaluation_v2','cohort_definition':'Count neighbors whose actual pre-noise observed arrays numerically changed; primary group robustness requires target_numeric_changed=true AND changed neighbors>=4',
         'source_scope':'Existing single-seed pilot predictions; no new inference, fitting, thresholds or normalization',
         'target_unchanged_interpretation':'Diagnostic only; these samples do not establish that target and surrounding group both changed movement',
         'threshold_interpretation':'Frozen source validation-max-F1 thresholds; absolute FPR is not a matched-FAR operating-point comparison',
         'undefined_metric_policy':'Null/empty metrics are excluded from regional means, never imputed as zero',
         'sidecar_sha256':check['sidecar_sha256'],'numeric_effect_input_validation_sha256':sha(args.out/'numeric_effect_input_validation.json'),
         'source_operator_report_sha256':check['source_group_operator_report_sha256'],'cross_generation_matrix_unchanged':True,
         'unchanged_cross_generation_matrix_sha256':check['unchanged_cross_matrix_sha256'],'models':{},
         'cohort_counts':{},'operator_to_numeric_support_transition':check['operator_to_numeric_support_transition'],
         'versions':{'numpy':np.__version__,'sklearn':sklearn.__version__},'script_sha256':sha(Path(__file__))}
 masks={'all_cases':np.ones(len(y),bool),'primary_target_and_group_changed':primary,
        'target_unchanged_diagnostic':~changed}
 for name,mask in masks.items():
  report['cohort_counts'][name]={'windows':int(mask.sum()),'no_behavior_samples':int(np.sum(mask&(y==0))),
   'per_region':{r:{'windows':int(np.sum(mask&(regions==r))),'no_behavior_samples':int(np.sum(mask&(regions==r)&(y==0)))} for r in all_regions}}
 paths=check['prediction_csv_sha256'];prediction_index={}
 original_group_report=json.loads((args.source/'group_variation_metrics.json').read_text())
 for filename,expected in paths.items():
  p=args.source/'predictions'/filename;require(sha(p)==expected,'Prediction changed during reaggregation')
  parts=filename.removesuffix('.csv').split('__');require(len(parts)==4,'Unexpected prediction filename')
  prediction_index[tuple(parts)]=p
 def read_scores(kind,feature,view):
  p=prediction_index['smooth',kind,feature,view];rows=csv_rows(p);d={r['window_id']:r for r in rows}
  require(set(d)==set(data['window_id']),'Prediction/cohort source IDs differ')
  out=[];thresholds=set()
  for i,w in enumerate(data['window_id']):
   row=d[str(w)];require(int(row['y'])==y[i] and row['region']==regions[i],'Prediction labels/region changed')
   out.append(float(row['score']));thresholds.add(float(row['frozen_validation_threshold']))
  require(len(thresholds)==1,'Source threshold changed within a prediction file')
  return np.asarray(out),thresholds.pop()
 for kind in BACKBONES:
  for feature in FEATURES:
   b,threshold=read_scores(kind,feature,'features_scene');s,sth=read_scores(kind,feature,'paired_group_scene');require(threshold==sth,'Baseline/stress threshold changed')
   def paired(mask):
    base=summarize(y,b,regions,threshold,mask,all_regions);stress=summarize(y,s,regions,threshold,mask,all_regions)
    delta={m:float(stress['macro'][m]-base['macro'][m]) if stress['macro'][m] is not None and base['macro'][m] is not None else None for m in ['AP','AUROC','F1','no_behavior_FPR']}
    return {'baseline':base,'stress':stress,'stress_minus_baseline_macro':delta}
   rec={'frozen_threshold':threshold,'all_cases':paired(masks['all_cases']),
        'primary_target_and_group_changed':paired(primary),
        'target_unchanged_diagnostic':paired(~changed),
        'numeric_neighbor_support_all_target_states':{},'numeric_neighbor_support_target_changed':{}}
   for condition in ['baseline','stress']:
    old=original_group_report['models'][kind+'__'+feature]['all_samples'][condition]
    for metric in ['AP','AUROC','F1','no_behavior_FPR']:
     require(abs(rec['all_cases'][condition]['macro'][metric]-old['macro'][metric])<1e-12,'All-case scores changed despite unchanged predictions')
   for group in GROUPS:
    which=np.array([cohort(int(n))==group for n in support]);rec['numeric_neighbor_support_all_target_states'][group]=paired(which)
    rec['numeric_neighbor_support_target_changed'][group]=paired(which&changed)
   report['models'][kind+'__'+feature]=rec
   print('REAGGREGATED',kind,feature,'primary',json.dumps(rec['primary_target_and_group_changed']['stress']['macro']),flush=True)
 require(sha(args.source/'cross_generation_matrix.json')==check['unchanged_cross_matrix_sha256'],'Cross matrix changed unexpectedly')
 report.update(weights_loaded=False,model_inference_performed=False,models_refitted=False,thresholds_refitted=False,normalizers_refitted=False,
               source_prediction_files_verified=len(paths),all_case_metrics_identical_to_operator_report=True,elapsed_seconds=time.time()-started)
 (args.out/'numeric_group_variation_metrics.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'status':'completed','out':str(args.out),'cohort_counts':report['cohort_counts']},indent=2),flush=True)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['prepare','reaggregate'],required=True)
 ap.add_argument('--sidecar',type=Path,default=ROOT/'group_numeric_effects.parquet')
 ap.add_argument('--source',type=Path,default=ROOT/'paired_view_evaluation_seed0')
 ap.add_argument('--out',type=Path,default=ROOT/'numeric_effect_evaluation_v2');args=ap.parse_args()
 if args.stage=='prepare':prepare(args)
 else:reaggregate(args)


if __name__=='__main__':main()
