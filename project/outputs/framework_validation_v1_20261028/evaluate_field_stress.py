from pathlib import Path
import argparse,json
import numpy as np
import torch
import train_study as ts
from models import make_model
R=Path(__file__).resolve().parent;O=R/'field_extension'
def run(arch,device):
    ts.verify_lock();cl=json.loads((O/'checkpoint_lock.json').read_text());assert cl['count']==72
    data=ts.load_npz(R/'extra_data'/'field_stress_test.npz');base=ts.load_npz(R/'data'/'individual'/'test.npz')
    assert np.array_equal(data['window_id'],base['window_id']) and np.array_equal(data['y'],base['y'])
    views=sorted({k.rsplit('__',1)[0] for k in data if k.endswith('__F0')});assert len(views)==15
    for seed in [0,1,2]:
      for arm in ['F0','F1','F2']:
        cp=O/'runs'/'individual'/arch/f'seed_{seed}'/arm/'final.pt';key=str(cp.relative_to(O)).replace('\\','/');assert ts.sha(cp)==cl['checkpoints'][key]
        out=R/'field_stress_results'/arch/f'seed_{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():continue
        saved=torch.load(cp,map_location='cpu',weights_only=False);model=make_model(arch,12,max_len=20,hidden=64).to(device);model.load_state_dict(saved['model']);results={}
        for view in views:
            td=ts.tensor_data(data,view+'__'+arm,saved['normalization'],device);p=ts.predict(model,td);ts.save_predictions(out/(view+'.npz'),data,p)
            m=ts.metrics(data['y'],p,data['region'].astype(str));m['operating']=ts.operating(data['y'],p,saved['thresholds']);m['predictions_sha256']=ts.sha(out/(view+'.npz'));results[view]=m
        ts.write_json(out/'result.json',{'backbone':arch,'seed':seed,'arm':arm,'checkpoint_sha256':ts.sha(cp),'data_sha256':ts.sha(R/'extra_data'/'field_stress_test.npz'),'protocol_sha256':ts.sha(R/'field_stress_protocol.json'),'views':results})
        print('FIELD_STRESS_DONE',arch,seed,arm,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    run(a.backbone,torch.device(a.device))
