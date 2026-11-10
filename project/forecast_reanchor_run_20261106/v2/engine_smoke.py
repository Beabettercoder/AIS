"""Synthetic CPU-only tests. No real test references or full experiments are read."""
from pathlib import Path
import argparse,json,tempfile
import numpy as np
import torch
import engine as E

def dataset(n,h,k,offset,ids):
    t=np.arange(h)*60.;q=np.arange(1,k+1)*60.;x=np.zeros((n,h,9),'f4')
    x[:,:,0]=5*t;x[:,:,2]=5;x[:,:,4]=5;x[:,:,5]=1;x[:,:,7]=t;x[:,1:,8]=60
    anchor=np.tile([5*t[-1],0.],(n,1)).astype('f8');v=np.tile([5.,0.],(n,1)).astype('f4')
    truth=anchor[:,None]+v[:,None]*q[None,:,None];truth[:,:,1]+=offset
    z=dict(future_xy=truth,query_s=np.tile(q,(n,1)).astype('f4'),time_s=np.tile(t,(n,1)).astype('f4'),
        window_id=np.array([f'synthetic_{ids+i}' for i in range(n)]),mmsi=np.arange(ids,ids+n),region=np.array(['synthetic']*n),maneuver=np.zeros(n,bool))
    for view in E.VIEWS:
        for name,a in dict(raw=x,gated=x,f0=np.zeros((n,h,8),'f4'),f1=np.zeros((n,h,8),'f4'),f0_valid=np.zeros((n,h),bool),
            f1_valid=np.zeros((n,h),bool),anchor_raw=anchor,anchor_gate=anchor,vel_raw=v,vel_gate=v).items():z[view+'__'+name]=a.copy()
    return z

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='forecast_v2_engine_') as tmp:
        tmp=Path(tmp);data=tmp/'data';data.mkdir()
        tr=dataset(8,24,4,20,1000);va=dataset(4,24,4,15,2000)
        for name,z in [('train',tr),('val',va),('test',dataset(4,24,4,10,3000))]:np.savez_compressed(data/f'{name}.npz',**z)
        norm=E.legacy.fit_normalizer(tr);E.save(tmp/'norm.json',norm)
        for m in E.MODEL_NAMES:
            for hidden in [64,128,256]:
                E.legacy.seed_all(0);net=E.ForecastModel(m,history_steps=24,hidden=hidden)
                tensors=E.prepare(va,'base','P3',norm);pred,err=E.infer(net,tensors,torch.device('cpu'),'P3',4)
                assert pred.shape==(4,4,2)
                np.testing.assert_allclose(err,15.,atol=0,rtol=0)
        checks.append('4 encoders x hidden64/128/256: dynamic H24/K4 and zero-initialized CV predictions exact')
        for policy in ['cosine_best','constant_last']:
            result=E.train(data,tmp/policy,'tcn',0,'P3',64,4,3,.002,policy,tmp/'norm.json','cpu')
            ck=torch.load(tmp/policy/'last_checkpoint.pt',weights_only=False)
            lr=[x['lr'] for x in ck['history']]
            assert (lr[0]>lr[-1]) if policy=='cosine_best' else len(set(lr))==1
            expected=min(ck['history'],key=lambda x:x['val_ADE_m'])['epoch'] if policy=='cosine_best' else 3
            assert result['selected_epoch']==expected
            assert len(list((tmp/policy/'weights').glob('epoch_*.pt')))==3
            assert result['best_val']['ADE_m']<=result['last_epoch']['ADE_m']
            assert (tmp/policy/'last_epoch_predictions_base.npz').exists()
        checks.append('Cosine scheduling and constant control; earliest minimum-val vs final selection; per-epoch weights and dual metrics')
        interrupted=tmp/'interrupted';original_save=E.checkpoint;triggered=False
        def stop_after_last(path,payload):
            nonlocal triggered
            original_save(path,payload)
            if Path(path).name=='last_checkpoint.pt' and payload['completed_epoch']==1 and not triggered:
                triggered=True;raise KeyboardInterrupt('synthetic interruption after complete epoch')
        E.checkpoint=stop_after_last
        try:
            try:E.train(data,interrupted,'tcn',0,'P3',64,4,3,.002,'cosine_best',tmp/'norm.json','cpu')
            except KeyboardInterrupt:pass
        finally:E.checkpoint=original_save
        resumed=E.train(data,interrupted,'tcn',0,'P3',64,4,3,.002,'cosine_best',tmp/'norm.json','cpu')
        aa=torch.load(tmp/'cosine_best/last_checkpoint.pt',weights_only=False);bb=torch.load(interrupted/'last_checkpoint.pt',weights_only=False)
        assert aa['history']==bb['history'] and aa['scheduler']==bb['scheduler']
        for key in aa['model']:torch.testing.assert_close(aa['model'][key],bb['model'][key],rtol=0,atol=0)
        for key in aa['best_model']:torch.testing.assert_close(aa['best_model'][key],bb['best_model'][key],rtol=0,atol=0)
        checks.append('Interrupted complete-epoch resume gives bit-identical last/best model, LR scheduler and validation history')
        try:E.evaluate(data,tmp/'refused',interrupted,device='cpu')
        except RuntimeError as e:assert 'freeze' in str(e).lower()
        else:raise AssertionError('Test accepted without freeze')
        E.save(tmp/'freeze.json',{'models':[{'selected_checkpoint_sha256':resumed['selected_checkpoint_sha256'],'model':'tcn','seed':0,'arm':'P3'}]})
        tested=E.evaluate(data,tmp/'test_results',interrupted,device='cpu',freeze_path=tmp/'freeze.json')
        assert tested['selected_epoch']==resumed['selected_epoch'] and tested['split']=='test'
        checks.append('Test blocked before freeze and uses only validation-selected checkpoint after freeze')
        try:E.train(data,interrupted,'tcn',0,'P3',128,4,3,.002,'cosine_best',tmp/'norm.json','cpu')
        except RuntimeError:pass
        else:raise AssertionError('Capacity changed on resume without rejection')
        checks.append('Resume refuses changed capacity/settings')
    E.save(a.out/'engine_smoke_report.json',dict(status='pass',synthetic_only=True,formal_training_started=False,checks=checks,engine_sha256=E.sha(E.__file__)))
    print('PASS',json.dumps(checks,indent=2))

if __name__=='__main__':main()
