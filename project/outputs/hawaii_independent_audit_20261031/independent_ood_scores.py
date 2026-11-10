from pathlib import Path
import sys, json, hashlib
import numpy as np, torch
R=Path('/mnt/sda/WangZijia/AIS/outputs'); D=R/'hawaii_event_feasibility_20261031';O=R/'hawaii_independent_audit_20261031';sys.path.insert(0,str(R/'heading_alignment_20261030'));import train_heading
m=json.loads((D/'frozen_ood_models_thresholds_v2.json').read_text())['models'];rows=json.loads((D/'ood_windows_v2.json').read_text())['all_windows'];
with np.load(D/'ood_inputs_v2.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
with np.load(D/'ood_scores.npz',allow_pickle=False) as f:prior=f['score']
assert prior.shape==(40,len(rows)) and len(rows)==358 and set(data).isdisjoint({'y','label','incident_num'})
selected=np.array([i for i,r in enumerate(rows) if r['name']=='MATSON NAVIGATOR']);assert len(selected)==5 and all(rows[i]['n_anchor_members']==0 and rows[i]['field_supported_steps']==0 for i in selected)
assert np.array_equal(data['F0'][selected],data['F1'][selected]) and np.array_equal(data['C0'][selected],data['F0'][selected])
dev=torch.device('cuda:5');torch.set_num_threads(2); errors=[]; mismatches=[];summaries=[]
for i,modelref in enumerate(m):
 model=train_heading.get_model(modelref['name'],40,modelref['name']=='itransformer').to(dev);model.load_state_dict(torch.load(modelref['model'],map_location=dev,weights_only=True));model.eval();arm={'C0':'C0','F0_heading':'F0','F1_heading':'F1'}[modelref['arm']];p=[]
 with torch.no_grad():
  for start in range(0,len(rows),128):
   idx=slice(start,start+128);mask=data['mask'][idx];t=data['time_s'][idx].copy();t-=t[:,:1];t[~mask]=0;x=torch.tensor(data[arm][idx],device=dev);out=model(x,torch.tensor(mask,device=dev),torch.tensor(t,dtype=torch.float32,device=dev))['logits'];p.extend(torch.sigmoid(out).cpu().numpy())
 p=np.asarray(p,dtype=np.float64);err=float(np.max(np.abs(p-prior[i])));errors.append(err)
 if not np.array_equal(p,prior[i]):mismatches.append(i)
 for name in sorted(set(r['name'] for r in rows)):
  idx=[j for j,r in enumerate(rows) if r['name']==name];v=p[idx];summaries.append({'model':i,'event':name,'windows':len(idx),'threshold_exceedances':{k:int(np.count_nonzero(v>v2['threshold'])) for k,v2 in modelref['thresholds'].items()},'median':float(np.median(v))})
 assert np.all((p>=0)&(p<=1))
saved=json.loads((D/'ood_descriptive_scores.json').read_text())['by_event_model'];assert len(summaries)==len(saved)==160
for own,other in zip(summaries,saved):
 assert own['event']==other['event_name'] and own['windows']==other['window_count'] and own['threshold_exceedances']==other['validation_threshold_exceedance_count'] and own['median']==other['score_median']
out={'windows':len(rows),'models':len(m),'model_window_pairs':prior.size,'device':'cuda:5 matching original','max_score_difference':max(errors),'bitwise_mismatch_model_indices':mismatches,'event_model_summary_rows':len(summaries),'same_data_candidate_matson':'all five windows use exactly identical C0,F0,F1 input and no local context; independent model weights can produce different scores','threshold':'independently rechecked from preexisting development validation negatives; not real-world FAR','label_safety':'ood_inputs.npz contains no label or incident class; event weak association not used as input','scope':'out-of-training-domain diagnostic; no event recall, precision, or actual FAR calculated'}
assert not mismatches
(O/'independent_ood_scores.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
