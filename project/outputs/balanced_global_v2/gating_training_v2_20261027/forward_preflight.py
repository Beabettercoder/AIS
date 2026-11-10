from pathlib import Path
import json,sys,numpy as np,torch
here=Path(__file__).parent
sys.path.insert(0,str(here/'historical'))
from historical_model import ARMS,HistoricalModel,training_loss
from train_historical import load_data,to_tensors,batch,fit_normalizer,normalized
data,_,_=load_data(here/'data'); torch.set_num_threads(2);device=torch.device('cuda:0')
out=[]
for kind in ('tcn','gru'):
  for arm,cfg in ARMS.items():
    d=data['train']; st=fit_normalizer(d['features'][cfg['key']][:64],d['mask'][:64]);x=normalized(d['features'][cfg['key']][:64],d['mask'][:64],st,False)
    dd={k:(v[:64] if isinstance(v,np.ndarray) else v) for k,v in d.items()}; tensors=to_tensors(dd,x);b=batch(tensors,slice(0,32),device)
    m=HistoricalModel(kind,arm).to(device);p,pc,h,c=m(b['x'],b['mask'],b['nb_seq'],b['nb_xy'],b['nb_mask'],b['nb_w'],b['rf'])
    loss,_=training_loss(m,b,torch.ones(5,device=device));loss.backward()
    assert p.shape==(32,) and torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None)
    out.append({'backbone':kind,'arm':arm,'parameters':sum(x.numel() for x in m.parameters()),'loss':float(loss.detach())})
(here/'forward_preflight.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
