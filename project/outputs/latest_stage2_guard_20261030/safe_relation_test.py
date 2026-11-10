"""Evaluate fixed relation models on the previously exposed test split after gate."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
HEADING = Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')
sys.path.insert(0, '/mnt/sda/WangZijia/AIS/outputs/eight_backbone_independent_audit_20261030')
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HEADING))
import train_heading as base
base.OLD = Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')
from relation8 import model_for
from guarded_scale_train import gate

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True,choices=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    p.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    if a.gpu==4: raise SystemExit('GPU 4 is reserved by an existing service.')
    gate()
    from candidate_integrity import check_candidate,check_gate_candidate
    freeze=check_candidate(('train','val'))
    base.VARIANT='aligned_embed';torch.set_num_threads(2);dev=torch.device(f'cuda:{a.gpu}')
    root=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')/'formal_runs'/a.name/f'seed{a.seed}'
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    mu=np.asarray(protocol['means']);sd=np.asarray(protocol['stds'])
    epochs=protocol['epochs']
    config=json.loads((HERE/'AUTHORIZED_STAGE2.json').read_text(encoding='utf8'))
    check_gate_candidate(config,freeze)
    if epochs != config.get('relation_epochs_by_backbone',{}).get(a.name,25): raise SystemExit('Mismatched registered epoch budget.')
    for arm in ('C0','F0','F1'):
        folder=root/arm; old=json.loads((folder/'result.json').read_text(encoding='utf8'))
        if old['last_epoch']!=epochs or sha(folder/'final.pt')!=old['weights_sha256'] or sha(folder/'val_scores.npz')!=old['prediction_sha256']: raise SystemExit('Stale or wrong trained checkpoint.')
        if (folder/'test_scores.npz').exists() or (folder/'test_result.json').exists(): raise FileExistsError(f'Refusing overwrite: {folder}')
    check_candidate(('test',))
    data=base.read('test')
    for arm in ('C0','F0','F1'):
        folder=root/arm; model=model_for(a.name).to(dev)
        try: state=torch.load(folder/'final.pt',map_location=dev,weights_only=True)
        except TypeError: state=torch.load(folder/'final.pt',map_location=dev)
        model.load_state_dict(state,strict=True);model.eval()
        with np.load(folder/'val_scores.npz',allow_pickle=False) as z:
            neg=z['base_score'][z['y']==0]
        thresholds={}
        for budget in (.01,.05):
            ordered=np.sort(neg);idx=min(len(ordered)-1,max(0,int(np.ceil((1-budget)*len(ordered)))-1))
            thresholds[str(budget)]=float(ordered[idx])
        results={};saved={}
        with torch.no_grad():
            for view in base.VIEWS:
                x,m,t=base.prep(data,view,arm,mu,sd,dev);pred=[]
                for i in range(0,len(x),256): pred.extend(torch.sigmoid(model(x[i:i+256],m[i:i+256],t[i:i+256])['logits']).cpu().numpy())
                pred=np.asarray(pred); y=data['y'];res=base.metrics(y,pred,data['region'])
                for budget,thr in thresholds.items():
                    positive=pred>thr;label=y>0
                    res[f'FAR_{budget}']={'threshold_from_base_val':thr,'actual_FAR':float(positive[~label].mean()),'recall':float(positive[label].mean())}
                results[view]=res;saved[view+'_score']=pred
        output=folder/'test_scores.npz'
        if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
        np.savez_compressed(output,y=data['y'],mmsi=data['mmsi'],region=data['region'],source=data['source'],case=data['case'],**saved)
        report={'name':a.name,'seed':a.seed,'arm':arm,'checkpoint':f'fixed epoch{epochs}',
                'caveat':'previously exposed development test; background can be shared; not a fresh blind test',
                'test_data_sha256':sha(base.OLD/'test_relation_field.npz'), 'checkpoint_sha256':sha(folder/'final.pt'),
                'prediction_sha256':sha(output),'views':results}
        (folder/'test_result.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()

