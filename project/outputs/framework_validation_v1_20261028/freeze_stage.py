from pathlib import Path
import json,hashlib,argparse,datetime
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_once(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',action='store_true');a=ap.parse_args()
if not a.checkpoints:
    m=json.loads((R/'data_manifest.json').read_text());assert m['status']=='complete_and_audited'
    names=['protocol.json','train_study.py','models.py','data_builder.py','data_manifest.json','data_schema.json','data_sources.json','model_inventory.json']
    fs={n:sha(R/n) for n in names};data={}
    for task,splits in m['tasks'].items():
        for split,meta in splits.items():assert sha(R/meta['path'])==meta['sha256'];data[meta['path']]=meta['sha256']
    save_once(R/'code_lock.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'frozen_before_training','files':fs,'data_files':data})
    print('CODE_FROZEN',sha(R/'code_lock.json'))
else:
    p=json.loads((R/'protocol.json').read_text());allcp={};missing=[]
    for task in p['tasks']:
      for arch in p['backbones']:
       for seed in p['seeds']:
        for arm in p['configurations']:
         path=R/'runs'/task/arch/f'seed_{seed}'/arm/'final.pt';rp=path.with_name('result.json')
         if not(path.exists() and rp.exists()):missing.append(str(path.relative_to(R)));continue
         rr=json.loads(rp.read_text());assert rr['code_lock_sha256']==sha(R/'code_lock.json');assert sha(path)==rr['checkpoint_sha256'];allcp[str(path.relative_to(R)).replace('\\','/')]=sha(path)
    assert not missing,missing
    save_once(R/'checkpoint_lock.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_lock_sha256':sha(R/'code_lock.json'),'checkpoints':allcp,'count':len(allcp)})
    print('CHECKPOINTS_FROZEN',len(allcp))
