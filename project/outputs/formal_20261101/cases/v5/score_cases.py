"""Frozen 8-backbone x 3-seed F1 single-vessel event replay. No result-based selection."""
from pathlib import Path
import numpy as np,json,torch,hashlib,sys,datetime
S=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v5');sys.path.insert(0,str(S))
from eight_backbone_adapter import Model8
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8<<20),b''):h.update(x)
 return h.hexdigest()
models=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];seeds=[0,1,2];cases=['MATSON','SARA']
protocol=json.loads((D/'case_protocol_prescore.json').read_text());assert protocol['architectures']==models and protocol['seeds']==seeds
frozen=S/'frozen_checkpoint_thresholds.json'; assert frozen.is_file(),'Wait for full 96-arm training + threshold freeze'
fr=json.loads(frozen.read_text());pt=S/'frozen_train_protocol.json';assert sha(pt)==fr['train_protocol_sha256'];norm=json.loads(pt.read_text())['normalizer_arrays'];sm=np.asarray(norm['sm']);ss=np.asarray(norm['ss']);fm=np.asarray(norm['fm']);fs=np.asarray(norm['fs']); assert (ss>0).all() and (fs>0).all()
lock={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_protocol_sha256':sha(D/'case_protocol_prescore.json'),'case_inputs_sha256':sha(D/'realcase_inputs.npz'),'case_audit_sha256':sha(D/'realcase_input_audit.json'),'training_protocol_sha256':sha(pt),'frozen_checkpoint_thresholds_sha256':sha(frozen),'models':[],'selection':'All 8 requested architectures x seeds 0,1,2, F1; no selection by case scores','threshold':'Each own pre-frozen single-vessel val-negative 1% and 5% thresholds; descriptive, not event-specific error guarantee'}
for n in models:
 for seed in seeds:
  a=fr['arms'][f'{n}/{seed}/f1_bundle'];p=Path(a['checkpoint']);assert p.is_file() and sha(p)==a['checkpoint_sha256'];lock['models'].append({'name':n,'seed':seed,'checkpoint':str(p),'checkpoint_sha256':a['checkpoint_sha256'],'validation_sha256':a['validation_sha256'],'thresholds':a['thresholds']})
assert len(lock['models'])==24
f=D/'case_models_prescore_lock.json';assert not f.exists();f.write_text(json.dumps(lock,indent=2))
with np.load(D/'realcase_inputs.npz',allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
inputs={}
for name in cases:
 mask=data[name+'__mask']; x=(data[name+'__self']-sm)/ss;x[~mask]=0; field=(data[name+'__field']-fm)/fs;field[~data[name+'__field_valid']]=0
 assert np.isfinite(x).all() and np.isfinite(field).all() and np.all(data[name+'__rho'][~data[name+'__field_valid']]==0)
 inputs[name]=tuple(torch.tensor(a) for a in (x.astype(np.float32),mask,data[name+'__time_s'].astype(np.float32),field.astype(np.float32),data[name+'__rho'].astype(np.float32)))
result={name:np.zeros((len(models),len(seeds),len(data[name+'__self'])),np.float64) for name in cases};torch.set_num_threads(4)
for i,n in enumerate(models):
 for seed in seeds:
  a=fr['arms'][f'{n}/{seed}/f1_bundle'];p=Path(a['checkpoint']);assert sha(p)==a['checkpoint_sha256'];model=Model8(n);model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));model.eval()
  with torch.no_grad():
   for name in cases:
    x,mask,tm,field,rho=inputs[name];logit,_,_,_=model(x,mask,tm,field,rho,True);v=torch.sigmoid(logit).numpy();assert np.isfinite(v).all();result[name][i,seed]=v
  print('SCORED',n,seed,flush=True)
out=D/'case_scores.npz';assert not out.exists();np.savez_compressed(out,**result,models=np.asarray(models),seeds=np.asarray(seeds));audit=json.loads((D/'realcase_input_audit.json').read_text());summary={}
for name in cases:
 rows=audit['cases'][name];scores=result[name];assert scores.shape[-1]==len(rows)
 out_rows=[]
 for j,r in enumerate(rows):
  per={}
  for i,n in enumerate(models):
   p=scores[i,:,j]; t=[fr['arms'][f'{n}/{seed}/f1_bundle']['thresholds'] for seed in seeds]
   per[n]={'scores':p.tolist(),'mean':float(p.mean()),'min':float(p.min()),'max':float(p.max()),'threshold_1pct':[q['0.01']['threshold'] for q in t],'threshold_5pct':[q['0.05']['threshold'] for q in t],'over_1pct':[bool(p[k]>t[k]['0.01']['threshold']) for k in range(3)],'over_5pct':[bool(p[k]>t[k]['0.05']['threshold']) for k in range(3)]}
  out_rows.append({'index':j,'start_utc':r['first_utc'],'end_utc':r['end_utc'],'n':r['count'],'median_sog_kn':r['median_sog_kn'],'max_sog_kn':r['max_sog_kn'],'context_valid_steps':r['context_valid_steps'],'field_fraction':r['context_valid_steps']/r['count'],'training_foundation_ok':r['target_training_foundation_ok'],'target_gap_policy_ok':r['target_gap_policy_ok'],'models':per})
 summary[name]=out_rows
sf=D/'case_scores.json';assert not sf.exists();sf.write_text(json.dumps({'scope':'Event-level OOD illustrative responses, no pointwise labels, no real recall/FAR, no calibration guarantee','scores_npz_sha256':sha(out),'prescore_lock_sha256':sha(f),'cases':summary},indent=2));print('SAVED',out,'sha',sha(out),flush=True)
