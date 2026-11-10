import os,sys,json,random,argparse,importlib,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from tqdm import tqdm
import data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=['tft_gcn','simad','dmrad'])
    p.add_argument('--arm',required=True,choices=['raw','gate','ordinary','reliable'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=25)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--mode',choices=['train','test'],default='train')
    p.add_argument('--out-root',default=None,help='Optional root directory for results')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    if a.method=='tft_gcn' and dev.type!='cuda': raise RuntimeError('TFT-GCN 原仓库图层初始化需要 CUDA')
    tr=data.load('train');va=data.load('val')
    if set(tr['mmsi']) & set(va['mmsi']): raise RuntimeError('训练与验证目标船重叠')
    norm=data.fit(tr); train_ix=np.flatnonzero(tr['y']==0)
    if a.smoke: train_ix=train_ix[:64]
    xt=data.features(tr,norm,a.arm)[train_ix];xv=data.features(va,norm,a.arm)
    if a.smoke: xv=xv[:64];va={k:v[:64] for k,v in va.items()}
    mod=importlib.import_module('adapters.'+a.method)
    net=mod.Adapter(seq_len=xt.shape[1],channels=xt.shape[2],gpu=a.gpu).to(dev)
    out=(Path(a.out_root) if a.out_root else Path(__file__).parent/'runs'/('smoke' if a.smoke else 'formal'))/a.method/a.arm/f'seed{a.seed}'
    out.mkdir(parents=True,exist_ok=True)
    def evaluate(x,y):
        net.eval();pred=[]; losses=[]
        with torch.no_grad():
            for start in range(0,len(x),a.batch_size):
                v=torch.as_tensor(x[start:start+a.batch_size],device=dev)
                pred.extend(net.score(v).detach().cpu().numpy().tolist())
        pred=np.array(pred); normal=pred[y==0]
        return pred,dict(normal_score_mean=float(normal.mean()) if len(normal) else None,
            ap=float(average_precision_score(y,pred)) if len(np.unique(y))==2 else None,
            auroc=float(roc_auc_score(y,pred)) if len(np.unique(y))==2 else None)
    if a.mode=='test':
        ck=torch.load(out/'best.pt',map_location=dev,weights_only=False);net.load_state_dict(ck['model']);test=data.load('test')
        if set(test['mmsi']) & (set(tr['mmsi'])|set(va['mmsi'])): raise RuntimeError('测试目标船身份重叠')
        x=data.features(test,norm,a.arm); pred,metrics=evaluate(x,test['y'])
        np.savez_compressed(out/'test_scores.npz',score=pred,y=test['y'],mmsi=test['mmsi'],region=test['region'],window_id=test['window_id'])
        (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8'); print('测试结果',metrics,flush=True);return
    if (out/'best.pt').exists() and not a.smoke: raise FileExistsError(f'已有结果，拒绝覆盖: {out}')
    spec={'method':a.method,'arm':a.arm,'seed':a.seed,'epochs':a.epochs,'train_full_window':len(train_ix),'val_full_window':len(xv),'input_channels':xt.shape[-1], 'train':'label 0 only; controlled labels', 'checkpoint':'minimum validation normal anomaly score; no test selection','data':'20 valid AIS steps only','normalization':'training normal data only; paired field statistics','evaluation':'window mean of native timestamp scores','test':'previously used dataset, not a new blind test'}
    (out/'protocol.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
    opt=torch.optim.Adam(net.parameters(),lr=a.lr);best=float('inf');history=[]
    for epoch in range(a.epochs):
        if hasattr(net,'begin_epoch'):net.begin_epoch(epoch,a.epochs)
        net.train();order=np.random.default_rng(a.seed+epoch*10000).permutation(len(xt));values=[]
        bar=tqdm(range(0,len(order),a.batch_size),desc=f'{a.method}/{a.arm} seed={a.seed} 轮次 {epoch+1}/{a.epochs}',file=sys.stdout,dynamic_ncols=True)
        for i in bar:
            ids=order[i:i+a.batch_size];v=torch.as_tensor(xt[ids],device=dev);opt.zero_grad(set_to_none=True)
            loss=net.loss(v,indices=ids,epoch=epoch)
            if not torch.isfinite(loss):raise RuntimeError('非有限损失')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();values.append(float(loss.detach()));bar.set_postfix(loss=f'{values[-1]:.5f}')
        pred,m=evaluate(xv,va['y']); row=dict(epoch=epoch+1,train_loss=float(np.mean(values)),**m);history.append(row)
        checkpoint={'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'protocol':spec,'metrics':m,'norm':norm}
        torch.save(checkpoint,out/'last.pt')
        if m['normal_score_mean'] is not None and m['normal_score_mean']<best:
            best=m['normal_score_mean'];torch.save(checkpoint,out/'best.pt')
            np.savez_compressed(out/'best_val_scores.npz',score=pred,y=va['y'],mmsi=va['mmsi'],region=va['region'],window_id=va['window_id'])
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print('验证',json.dumps(row,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
