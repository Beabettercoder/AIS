"""Frozen relaxed relation model study; test is read only after checkpoint lock."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys
from datetime import datetime, timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
DATA=ROOT/'relation_dual_pressure_v2_20261027'
OUT=DATA/'model_study'
PRIOR=ROOT/'relation_robust_v1_20261027'
MANIFEST=DATA/'dataset_manifest_combined.json'
sys.path.insert(0,str(PRIOR)); import run_relation as rr
sys.path.insert(0,str(ROOT/'relation_test_expanded_v1_20261027')); import report_context_common as rc
VIEWS=('base','natural','suffix')
MODES=('target','mean','median','absolute_median','absolute_report_median','mixed_report_median')
ARCH=('tcn','gru'); SEEDS=(0,1,2); MAXLEN=20

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def save(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def check_manifest():
    lock=json.loads((OUT/'code_lock.json').read_text(encoding='utf8'))
    assert sha(Path(__file__))==lock['script_sha256']
    assert sha(DATA/'protocol.json')==lock['protocol_sha256']
    assert sha(MANIFEST)==lock['dataset_manifest_sha256']
    for p,h in lock['dependency_hashes'].items(): assert sha(p)==h,p
    return json.loads(MANIFEST.read_text(encoding='utf8'))
def safe_path(x):
    p=(DATA/x).resolve(); assert p.is_relative_to(DATA.resolve()); return p
def weights_abs(*args):
    v,wa,wm,e,er=weight_original(*args)
    return v,wa,wa,e,er
weight_original=rr.weights

def prepare(split):
    m=check_manifest()
    if split=='test':
        fl=OUT/'checkpoint_lock.json'; assert fl.exists(), 'test requires frozen training checkpoints'
        cl=json.loads(fl.read_text(encoding='utf8'))
        assert cl['code_lock_sha256']==sha(OUT/'code_lock.json')
        for p,h in cl['checkpoints'].items(): assert sha(p)==h,p
    rows=[x for x in m['samples'] if x['version']=='relaxed' and x['split']==split]
    rows=sorted(rows,key=lambda x:(x['region'],x['window_id']))
    if not rows: raise RuntimeError(f'No relaxed {split} source')
    x={f'{v}__{k}':[] for v in VIEWS for k in MODES}; lens=[]; labels=[];regions=[];sources=[];ships=[];cases=[];statrows=[]
    for item_index,item in enumerate(rows):
        for view in VIEWS:
            entry=item['files'][view];p=safe_path(entry['path']);assert sha(p)==entry['sha256']
            with np.load(p,allow_pickle=False) as z:
                obs=rr.observed(z); y=np.asarray(z['y'],dtype=int)
            assert np.array_equal(y,[0,1,1,0]); assert len(obs['target_t'])==4
            for case in range(4):
                rr.weights=weight_original
                stats,f=rr.extract(obs,case); mixed_report=rc.extract(obs,case)
                rr.weights=weights_abs
                abs_stats,abs_f=rr.extract(obs,case); abs_report=rc.extract(obs,case)
                rr.weights=weight_original
                fs={'target':f['target'],'mean':f['mean'],'median':f['median'],
                    'absolute_median':abs_f['mixed_median'],'absolute_report_median':abs_report,
                    'mixed_report_median':mixed_report}
                n=len(fs['target']); assert 1<n<=MAXLEN
                for mode in MODES:
                    assert fs[mode].shape==(n,7) and np.isfinite(fs[mode]).all()
                    padded=np.zeros((MAXLEN,7),np.float32);padded[:n]=fs[mode];x[f'{view}__{mode}'].append(padded)
                statrows.append(dict(split=split,view=view,source=item['window_id'],mmsi=item['mmsi'],
                    region=item['region'],case=case,y=int(y[case]),**stats,
                    absolute_weighted_median=abs_stats['mixed_weighted_median'],
                    absolute_report_weighted_median=abs_stats['mixed_report_weighted_median']))
                if view=='base':
                    labels.append(y[case]);lens.append(n);regions.append(item['region']);sources.append(item['window_id']);ships.append(str(item['mmsi']));cases.append(case)
                else:
                    idx=4*item_index+case
                    assert lens[idx]==n and labels[idx]==y[case]
        if len(lens)%400==0:print('PREP',split,len(lens)//4,'/',len(rows),flush=True)
    out={k:np.asarray(v,dtype=np.float32) for k,v in x.items()}
    out.update(y=np.asarray(labels,dtype=np.int8),length=np.asarray(lens,dtype=np.int16),
               region=np.asarray(regions),source=np.asarray(sources),mmsi=np.asarray(ships),case=np.asarray(cases,dtype=np.int8))
    assert all(a.shape==(len(labels),MAXLEN,7) for k,a in out.items() if '__' in k)
    assert len(labels)==4*len(rows)
    for view in VIEWS:
        tar=out[f'{view}__target'];assert np.array_equal(tar[0::4],tar[1::4]);assert np.array_equal(tar[2::4],tar[3::4])
        for mode in MODES[1:]:
            z=out[f'{view}__{mode}'][...,4:]; assert np.array_equal(z[0::4],z[2::4]);assert np.array_equal(z[1::4],z[3::4])
    path=OUT/f'features_{split}.npz';assert not path.exists();np.savez_compressed(path,**out)
    csvpath=OUT/f'statistics_{split}.csv'
    with csvpath.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(statrows[0]));writer.writeheader();writer.writerows(statrows)
    save(OUT/f'features_{split}_manifest.json',{'split':split,'source_count':len(rows),'target_mmsi':len(set(ships)),
         'scenario_count':len(labels),'dataset_manifest_sha256':sha(MANIFEST),
         'feature_file_sha256':sha(path),'statistics_file_sha256':sha(csvpath),'code_lock_sha256':sha(OUT/'code_lock.json')})
    print('PREP_COMPLETE',split,len(rows),len(set(ships)),flush=True)

def metrics(y,p,region):
    x={r:float(average_precision_score(y[region==r],p[region==r])) for r in np.unique(region)}
    return {'pooled_AP':float(average_precision_score(y,p)),'pooled_AUROC':float(roc_auc_score(y,p)),
            'macro_region_AP':float(np.mean(list(x.values()))),'regions':x}
def predict(model,X,L,device):
    model.eval();p=[]
    with torch.no_grad():
        for lo in range(0,len(X),256):
            xb=torch.tensor(X[lo:lo+256],device=device);lb=torch.tensor(L[lo:lo+256],dtype=torch.long,device=device)
            p.extend(torch.sigmoid(model(xb,lb)).cpu().numpy().tolist())
    return np.array(p,dtype=np.float32)
def train(arch,device_name):
    check_manifest();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device=torch.device(device_name)
    with np.load(OUT/'features_train.npz',allow_pickle=False) as q:tr={k:q[k] for k in q.files}
    with np.load(OUT/'features_val.npz',allow_pickle=False) as q:va={k:q[k] for k in q.files}
    assert not set(tr['mmsi'])&set(va['mmsi']);dest=OUT/'runs'/arch;dest.mkdir(parents=True,exist_ok=False)
    training=dict(epochs=20,batch_size=64,lr=.001,optimizer='Adam',scaler='train-valid-time-steps',views=list(VIEWS),modes=list(MODES),seeds=list(SEEDS))
    save(dest/'worker_config.json',{'training':training,'code_lock_sha256':sha(OUT/'code_lock.json'),
         'train_features_sha256':sha(OUT/'features_train.npz'),'val_features_sha256':sha(OUT/'features_val.npz'),
         'device':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})
    valid=np.arange(MAXLEN)[None,:]<tr['length'][:,None];y=tr['y'].astype(np.float32)
    for exposure in VIEWS:
      for mode in MODES:
        source=tr[f'{exposure}__{mode}']; z=source[valid];mu=z.mean(0);sd=z.std(0);sd[sd<1e-6]=1
        normalized=((source-mu)/sd).astype(np.float32)
        trainX=torch.tensor(normalized,device=device);trainL=torch.tensor(tr['length'],dtype=torch.long,device=device);trainY=torch.tensor(y,device=device)
        evalX={v:((va[f'{v}__{mode}']-mu)/sd).astype(np.float32) for v in VIEWS}
        for seed in SEEDS:
            torch.manual_seed(seed);np.random.seed(seed)
            model=rc.Model(arch).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(seed)
            losslog=[];order_hashes=[]
            for epoch in range(20):
                model.train();batchloss=[];order=rng.permutation(len(y));order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
                for a in np.array_split(order,int(np.ceil(len(order)/64))):
                    ix=torch.tensor(a,dtype=torch.long,device=device);opt.zero_grad(set_to_none=True)
                    logits=model(trainX[ix],trainL[ix]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,trainY[ix]);loss.backward();opt.step();batchloss.append(float(loss.detach().cpu()))
                losslog.append(float(np.mean(batchloss)))
            run=dest/exposure/mode/f'seed{seed}';run.mkdir(parents=True,exist_ok=False)
            checkpoint=run/'checkpoint.pt'
            torch.save({'state_dict':model.state_dict(),'mean':mu,'std':sd,'architecture':arch,'mode':mode,'exposure':exposure,'seed':seed,'epochs':20},checkpoint)
            preds={v:predict(model,evalX[v],va['length'],device) for v in VIEWS}
            np.savez_compressed(run/'val_predictions.npz',**preds,source=va['source'],mmsi=va['mmsi'],region=va['region'],y=va['y'],case=va['case'])
            save(run/'run.json',{'architecture':arch,'mode':mode,'exposure':exposure,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),
                  'loss_per_epoch':losslog,'batch_order_sha256_per_epoch':order_hashes,'checkpoint_sha256':sha(checkpoint),
                  'val_predictions_sha256':sha(run/'val_predictions.npz'),
                  'val_metrics':{v:metrics(va['y'],preds[v],va['region']) for v in VIEWS}})
            print('TRAIN',arch,exposure,mode,seed,'val',*(f'{v}:{metrics(va["y"],preds[v],va["region"])["pooled_AP"]:.4f}' for v in VIEWS),flush=True)
    save(dest/'completed.json',{'status':'complete','count':len(VIEWS)*len(MODES)*len(SEEDS),'worker_config_sha256':sha(dest/'worker_config.json')})

def freeze():
    lock=check_manifest();checkpoints={}
    for arch in ARCH:
        assert (OUT/'runs'/arch/'completed.json').exists()
        for exposure in VIEWS:
          for mode in MODES:
            for seed in SEEDS:
                run=OUT/'runs'/arch/exposure/mode/f'seed{seed}'
                cp=run/'checkpoint.pt';r=json.loads((run/'run.json').read_text());assert sha(cp)==r['checkpoint_sha256'];assert sha(run/'val_predictions.npz')==r['val_predictions_sha256']
                checkpoints[str(cp)]=sha(cp)
    save(OUT/'checkpoint_lock.json',{'status':'all_training_checkpoints_frozen_before_test_features_and_prediction',
         'frozen_utc':datetime.now(timezone.utc).isoformat(),'code_lock_sha256':sha(OUT/'code_lock.json'),
         'source_manifest_sha256':sha(MANIFEST),'count':len(checkpoints),'checkpoints':checkpoints})
    print('FREEZE',len(checkpoints),flush=True)

def evaluate(arch,device_name):
    check_manifest();cl=json.loads((OUT/'checkpoint_lock.json').read_text());assert cl['count']==108
    torch.set_num_threads(2);device=torch.device(device_name)
    with np.load(OUT/'features_test.npz',allow_pickle=False) as q:te={k:q[k] for k in q.files}
    dest=OUT/'runs'/arch; rows=[]
    for exposure in VIEWS:
      for mode in MODES:
        for seed in SEEDS:
            run=dest/exposure/mode/f'seed{seed}';cp=run/'checkpoint.pt';assert sha(cp)==cl['checkpoints'][str(cp)]
            ck=torch.load(cp,map_location='cpu',weights_only=False);model=rc.Model(arch).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            preds={v:predict(model,((te[f'{v}__{mode}']-ck['mean'])/ck['std']).astype(np.float32),te['length'],device) for v in VIEWS}
            np.savez_compressed(run/'test_predictions.npz',**preds,source=te['source'],mmsi=te['mmsi'],region=te['region'],y=te['y'],case=te['case'])
            result={'architecture':arch,'exposure':exposure,'mode':mode,'seed':seed,'checkpoint_sha256':sha(cp),
                    'test_predictions_sha256':sha(run/'test_predictions.npz'), 'metrics':{v:metrics(te['y'],preds[v],te['region']) for v in VIEWS}}
            save(run/'test_metrics.json',result);rows.append(result)
    save(dest/'test_results.json',{'checkpoint_lock_sha256':sha(OUT/'checkpoint_lock.json'),'results':rows})
    print('TEST_DONE',arch,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','freeze','evaluate']);p.add_argument('--split',choices=['train','val','test']);p.add_argument('--arch',choices=ARCH);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':assert a.split;prepare(a.split)
    elif a.action=='train':assert a.arch;train(a.arch,a.device)
    elif a.action=='freeze':freeze()
    else:assert a.arch;evaluate(a.arch,a.device)
