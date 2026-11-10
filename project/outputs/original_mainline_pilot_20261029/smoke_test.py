import sys
from pathlib import Path
import numpy as np
import torch
root=Path('/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028')
sys.path.insert(0,str(root))
from models import make_model
from pilot_features import build_field_inputs
from pilot_adapter import FieldResidualAdapter
with np.load(root/'data/individual/train.npz',allow_pickle=False) as z: core={k:z[k] for k in z.files}
with np.load(root/'extra_data/field_train.npz',allow_pickle=False) as z: field={k:z[k] for k in z.files}
r=build_field_inputs(core,field)
mask=torch.from_numpy(core['mask'][:4]);times=torch.from_numpy(core['time_s'][:4]);x=torch.from_numpy(r['self'][:4]);f=torch.from_numpy(r['field'][:4]);rho=torch.from_numpy(r['rho'][:4]);torch.set_num_threads(1)
for name in ('tcn','gru','transformer','itransformer'):
    torch.manual_seed(42); net=make_model(name,7); net.eval(); adapter=FieldResidualAdapter(net);adapter.eval()
    with torch.no_grad():
        a=net(x,mask,times)['logits'];b=adapter(x,mask,times,f,rho)['logits'];adapter.scale.fill_(1.0);c=adapter(x,mask,times,f,torch.zeros_like(rho))['logits']
    assert torch.equal(a,b) and torch.equal(a,c),name
    print(name,'initial_and_zero_support_exact',True)
print('rows',len(r['self']),'valid_steps',int(r['valid'].sum()),'rho_unique',np.unique(r['rho']).tolist())
