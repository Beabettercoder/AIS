from pathlib import Path
import json,argparse,datetime
import train_study as ts
R=Path(__file__).resolve().parent;O=R/'field_extension';O.mkdir(exist_ok=True)
ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',action='store_true');a=ap.parse_args()
def save(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
if not a.checkpoints:
    names=['models.py','train_study.py','protocol.json','data_manifest.json','code_lock.json']
    fn=['field_builder.py','field_protocol.json','field_audit.json','train_field_extension.py']+[f'extra_data/field_{s}.npz' for s in ['train','val','test']]
    manifest={'parent_data_manifest_sha256':ts.sha(R/'data_manifest.json'),'fields':{p:ts.sha(R/p) for p in fn},'source':'same individual task and splits, exposed development test; local-affine weighting extension'}
    save(O/'data_manifest.json',manifest)
    save(O/'code_lock.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_files':{p:ts.sha(R/p) for p in names},'field_files':{p:ts.sha(R/p) for p in fn},'fixed_normalizer':'parent train pooled C1/C2/C3 normalizer','arms':['F0','F1','F2'],'shared_training':'parent fixed epoch25 pure BCE, 8 encoders x3arms x3seeds=72'})
    print('FIELD_CODE_FROZEN',ts.sha(O/'code_lock.json'))
else:
    p=json.loads((R/'protocol.json').read_text());cps={}
    for arch in p['backbones']:
      for seed in p['seeds']:
       for arm in ['F0','F1','F2']:
        path=O/'runs'/'individual'/arch/f'seed_{seed}'/arm/'final.pt';rr=json.loads(path.with_name('result.json').read_text());assert ts.sha(path)==rr['checkpoint_sha256'];assert rr['code_lock_sha256']==ts.sha(O/'code_lock.json');cps[str(path.relative_to(O)).replace('\\','/')]=ts.sha(path)
    save(O/'checkpoint_lock.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(cps),'checkpoints':cps,'field_code_lock_sha256':ts.sha(O/'code_lock.json')});print('FIELD_CHECKPOINTS_FROZEN',len(cps))
