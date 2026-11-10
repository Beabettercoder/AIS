"""One frozen forecasting test evaluation; no checkpoint selection or refitting."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from common import *

def persist(folder,view,pred,data,model,arm,seed):
    folder.mkdir(parents=True,exist_ok=True);error=np.linalg.norm(pred-data['future_xy'],axis=-1)
    tmp=folder/f'predictions_{view}.tmp.npz';dest=folder/f'predictions_{view}.npz'
    np.savez_compressed(tmp,pred_xy=pred,future_xy=data['future_xy'],query_s=data['query_s'],point_error=error,ADE=error.mean(1),FDE=error[:,-1],
        window_id=data['window_id'],mmsi=data['mmsi'],region=data['region'],maneuver=data['maneuver'],split='test',view=view,model=model,arm=arm,seed=seed)
    tmp.replace(dest);return T.metrics(error,data)

def evaluate_test(work,device_name='cuda:0'):
    work=Path(work);freeze=assert_frozen(work);test_manifest=read(work/'data/test_manifest.json')
    assert sha(work/'data/test.npz')==test_manifest['npz_sha256']
    data=T.load_data(work/'data/test.npz');norm=read(work/'data/normalizer.json');device=torch.device(device_name)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    identity=dict(freeze_sha256=sha(work/'checkpoint_freeze.json'),test_sha256=sha(work/'data/test.npz'),device=device_name,
        code_hashes=code_hashes(),torch_version=torch.__version__)
    marker=work/'test_evaluation_plan.json'
    if marker.exists():assert read(marker)==identity,'Do not change frozen test evaluation settings on resume'
    else:save(marker,identity)
    for name,kind in [('CV_raw','raw'),('CV_gate','gate')]:
        folder=work/'test_runs/baselines'/name;result={}
        for view in CONFIG['views']:
            pred=data[view+'__anchor_'+kind][:,None]+data[view+'__vel_'+kind].astype(float)[:,None]*data['query_s'].astype(float)[...,None]
            result[view]=persist(folder,view,pred,data,name,name,-1)
        save(folder/'result.json',dict(status='complete',model=name,arm=name,seed=-1,split='test',completed_epochs=0,expected_epochs=0,views=result))
    for row in tqdm(freeze['models'],desc='Frozen TEST models',unit='model'):
        m,seed,arm=row['model'],row['seed'],row['arm'];folder=work/f'test_runs/{m}/seed_{seed}/{arm}'
        done=folder/'result.json'
        if done.exists():
            r=read(done);assert r['checkpoint_sha256']==row['checkpoint_sha256']
            if r['status']=='complete':continue
        model=ForecastModel(m).to(device);ck=torch.load(row['checkpoint'],map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);assert ck['completed_epoch']==CONFIG['epochs'][m]
        result={};base=None
        for view in CONFIG['views']:
            pred,error=T.evaluate(model,T.prepare(data,view,arm,norm),device,arm,128)
            if view=='base':base=pred.copy()
            if view=='neighbor_stress' and arm in ['P0','P1']:np.testing.assert_array_equal(pred,base)
            result[view]=persist(folder,view,pred,data,m,arm,seed)
            save(work/'test_progress.json',dict(stage='test_evaluation',model=m,seed=seed,arm=arm,view=view))
        save(done,dict(status='complete',model=m,arm=arm,seed=seed,split='test',checkpoint_sha256=row['checkpoint_sha256'],
            completed_epochs=CONFIG['epochs'][m],expected_epochs=CONFIG['epochs'][m],views=result))
    assert_frozen(work);save(work/'test_complete.json',dict(status='complete',model_count=48,identity=identity))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();evaluate_test(a.work,a.device)
