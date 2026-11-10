"""One-batch reproducibility/forward checks, never benchmark fitting."""
from pathlib import Path
import argparse,json,os
import numpy as np
import torch
import train_historical as tr
import historical_model as hm

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device(args.device)
    data,proof,_=tr.load_data(args.data);stats=tr.fit_normalizer(data['train']['features']['kglf_compat9'],data['train']['mask'])
    t=tr.to_tensors(data['train'],tr.normalized(data['train']['features']['kglf_compat9'],data['train']['mask'],stats,False))
    clean=np.flatnonzero(~data['train']['observation_noise']);cnt=np.bincount(data['train']['label_class'][clean][data['train']['label_class'][clean]>0]-1,minlength=5)
    cw=torch.tensor(1./np.maximum(cnt,1),dtype=torch.float32,device=device)
    ii=clean[:128];b=tr.batch(t,ii,device);full=clean[data['train']['mask'][clean].all(1)][:32];bf=tr.batch(t,full,device)
    reports=[]
    for kind in ['tcn','gru']:
        tr.seed_all(3);ref=hm.legacy.Net(kind,9,'ours').to(device)
        tr.seed_all(3);model=hm.HistoricalModel(kind,'full35_compat').to(device)
        ref.eval();model.eval()
        with torch.no_grad():
            x=ref(bf['x'],bf['nb_seq'],bf['nb_xy'],bf['nb_mask'],bf['nb_w'],bf['rf'][:,-1])
            y=model(bf['x'],bf['mask'],bf['nb_seq'],bf['nb_xy'],bf['nb_mask'],bf['nb_w'],bf['rf'])
        errors=[float((a-bb).abs().max()) for a,bb in zip(x,y)]
        if any(v!=0 for v in errors):raise AssertionError('Historical GPU forward not exact')
        hashes=[];losses=[]
        for repeat in [0,1]:
            tr.seed_all(9);m=hm.HistoricalModel(kind,'full35_compat').to(device);m.train()
            opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
            loss,logs=hm.training_loss(m,b,cw);loss.backward();opt.step()
            if not torch.isfinite(loss) or any(not torch.isfinite(p).all() for p in m.parameters()):raise FloatingPointError('Nonfinite GPU step')
            hashes.append(tr.state_sha(m.state_dict()));losses.append(float(loss.detach()))
        if hashes[0]!=hashes[1] or losses[0]!=losses[1]:raise AssertionError('GPU one-step replay is not deterministic')
        reports.append(dict(backbone=kind,full20_source_forward_max_abs_errors=errors,one_step_losses=losses,one_step_state_sha256=hashes[0],deterministic_repeats_equal=True))
    result=dict(status='passed',scope='Source forward parity and duplicate one-batch updates; no experiment run/checkpoint selection',device=str(device),device_name=torch.cuda.get_device_name(device),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),tf32=False,dtype='float32',sources=proof,clean_training_windows=len(clean),reports=reports,code_sha256={p.name:tr.sha(p) for p in [Path(__file__),tr.HERE/'train_historical.py',tr.HERE/'historical_model.py']})
    args.out.parent.mkdir(parents=True,exist_ok=True);tr.jwrite(args.out,result)
    print(json.dumps(dict(status='passed',clean_training_windows=len(clean),reports=reports)))
if __name__=='__main__':main()
