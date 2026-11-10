from pathlib import Path
import json,sys
import numpy as np,torch
R=Path('/mnt/sda/WangZijia/AIS/outputs');S=R/'formal_20261101/single';D=R/'formal_20261101/cases/v6';O=R/'formal_20261101/independent_audit/case_external_20261101'
sys.path.insert(0,str(S));from eight_backbone_adapter import Model8
fr=json.loads((S/'frozen_checkpoint_thresholds.json').read_text());n=json.loads((S/'frozen_train_protocol.json').read_text())['normalizer_arrays'];sm=np.asarray(n['sm']);ss=np.asarray(n['ss']);fm=np.asarray(n['fm']);fs=np.asarray(n['fs']);device='cuda:2';torch.set_num_threads(2)
names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];results={};maximum=0.;nactual=0
for subset in ('nonoverlap','rolling'):
 z=np.load(D/('realcase_inputs.npz' if subset=='nonoverlap' else 'rolling_inputs.npz'),allow_pickle=False)
 for case in ('MATSON','SARA'):
  mask=z[case+'__mask'];x=(z[case+'__self'].astype(float)-sm)/ss;x[~mask]=0;field=(z[case+'__field'].astype(float)-fm)/fs;field[~z[case+'__field_valid']]=0
  rho=z[case+'__rho'];tm=z[case+'__time_s'];empty=(rho*mask).sum(1)==0;nactual+=int(empty.sum())*24;results[subset+'/'+case]={'zero_support_windows':int(empty.sum()),'windows':len(empty),'max_abs_diff_actual_zero_support':0.,'max_abs_diff_forced_zero_support':0.}
  X=tuple(torch.as_tensor(a,device=device) for a in (x.astype(np.float32),mask,tm.astype(np.float32),field.astype(np.float32),rho.astype(np.float32)));xt,mt,tt,ft,rt=X
  randomfield=torch.arange(ft.numel(),device=device,dtype=torch.float32).reshape(ft.shape).remainder(17).sub(8)
  for n in names:
   for seed in (0,1,2):
    model=Model8(n).to(device);model.load_state_dict(torch.load(fr['arms'][f'{n}/{seed}/f1_bundle']['checkpoint'],map_location=device,weights_only=True));model.eval()
    with torch.inference_mode():
     forced=model(xt,mt,tt,randomfield,torch.zeros_like(rt),True)[0];without=model(xt,mt,tt,ft,rt,False)[0]
     df=float(torch.max(torch.abs(forced-without)).cpu());results[subset+'/'+case]['max_abs_diff_forced_zero_support']=max(results[subset+'/'+case]['max_abs_diff_forced_zero_support'],df)
     if empty.any():
      actual=model(xt,mt,tt,ft,rt,True)[0];ix=torch.as_tensor(empty,device=device);da=float(torch.max(torch.abs(actual[ix]-without[ix])).cpu());results[subset+'/'+case]['max_abs_diff_actual_zero_support']=max(results[subset+'/'+case]['max_abs_diff_actual_zero_support'],da)
    del model
  assert results[subset+'/'+case]['max_abs_diff_forced_zero_support']==0
  assert results[subset+'/'+case]['max_abs_diff_actual_zero_support']==0
  print(subset,case,json.dumps(results[subset+'/'+case]),flush=True)
out={'scope':'All 24 frozen F1 models; no-field branch vs field-enabled with rho forced zero and arbitrary finite field, plus all actual zero-support windows. False branch uses same trained model weights.','actual_zero_support_model_windows':nactual,'results':results}
p=O/'case_fallback_independent.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf8')
