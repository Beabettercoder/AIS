"""Independent full-ledger audit against frozen sources and counterfactual NPZs."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

BASE=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
OUT=BASE/'labeling/labeled_v1';CF=BASE/'labeling/counterfactuals_v1';SOURCE=BASE/'release_20261026'
KINDS=['turn_back','deviation','stall','sinuous','speed_drop'];Q={'train':(980,182,196,392),'val':(210,39,42,84),'test':(210,39,42,84)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def velocity(s,c):
 a=np.deg2rad(c);return s[:,None]*(1852/3600)*np.column_stack((np.sin(a),np.cos(a)))

started=time.time();g=pd.read_parquet(OUT/'label_manifest.parquet');rep=json.loads((OUT/'selection_report.json').read_text());registry=json.loads((SOURCE/'source_registry.json').read_text())['regions']
groups=pd.read_parquet(SOURCE/'global_all_mmsi_split.parquet');group_map=dict(zip(groups.mmsi,groups.split))
assert len(g)==11200 and not g.window_id.duplicated().any() and np.array_equal(g.window_id,g.base_window_id)
assert g.labels_frozen.all() and g.counterfactual_accepted.all() and g.y.isin([0,1]).all()
assert int(g.y.sum())==4480 and sha(OUT/'label_manifest.parquet')==rep['label_manifest_sha256']
assert g.groupby('mmsi').split.nunique().max()==1
assert set(g.loc[g.y==1,'latent_kind'].value_counts())=={896}
audit={'status':'passed','windows':len(g),'positives':int(g.y.sum()),'negatives':int((g.y==0).sum()),'global_unique_mmsi':int(g.mmsi.nunique()),'regions':{},'source_observations_checked':0,'counterfactual_innovation_max_abs_difference_mps':0.,'input_hashes':{'label_manifest':sha(OUT/'label_manifest.parquet'),'selection_report':sha(OUT/'selection_report.json'),'verifier_script':sha(Path(__file__))},'legacy_noise_field_note':'requested_noise_type is an inherited proposal. Final application uses noise_kind=noise_type=observation_noise_type; the feature builder has been explicitly told this contract.'}
split_ids={s:set() for s in Q};split_obs={s:set() for s in Q}
for region,d in g.groupby('region',sort=True):
 assert len(d)==1400 and d.mmsi.nunique()==260 and set(d.loc[d.y==1,'latent_kind'].value_counts())=={112}
 cp=CF/region/'counterfactual_plan.parquet';cz=CF/region/'counterfactual_observations.npz';parents=pd.read_parquet(cp).set_index('window_id');z=np.load(cz,allow_pickle=False)
 assert sha(cp)==d.counterfactual_plan_sha256.iloc[0] and sha(cz)==d.counterfactual_observations_sha256.iloc[0]
 meta=registry[region];p=Path(meta['canonical_file']);assert sha(p)==meta['sha256']
 raw=pd.read_parquet(p,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True);a={k:raw[k].to_numpy() for k in raw.columns}
 b=meta['core_bbox'];lon0=(b[0]+b[2])/2;lat0=(b[1]+b[3])/2
 summary={}
 for split,h in d.groupby('split',sort=True):
  nw,nv,nk,ny=Q[split];assert len(h)==nw and h.mmsi.nunique()==nv and h.y.sum()==ny
  assert h.groupby('mmsi').size().between(1,20).all()
  assert all((h.latent_kind==kind).sum()==nk for kind in KINDS)
  poscounts=h.loc[h.y==1,'latent_kind'].value_counts();assert poscounts.max()-poscounts.min()<=1
  noise={}
  for y,sub in h.groupby('y'):
   noisy=sub[sub.observation_noise];assert abs(len(noisy)-len(sub)/3)<=.5+1e-8
   assert set(noisy.noise_kind)<=set(['teleport','speed_spike','drift']) and set(noisy.noise_scope)<=set(['target_only','neighbors_only','both'])
   assert noisy.noise_kind.value_counts().max()-noisy.noise_kind.value_counts().min()<=1
   assert noisy.noise_scope.value_counts().max()-noisy.noise_scope.value_counts().min()<=1
   assert (sub.loc[~sub.observation_noise,'noise_kind']=='none').all() and (sub.loc[~sub.observation_noise,'noise_scope']=='none').all()
   noise[str(int(y))]={'windows':len(sub),'noisy':len(noisy),'fraction':len(noisy)/len(sub)}
  assert (h.noise_kind==h.noise_type).all() and (h.noise_kind==h.observation_noise_type).all()
  summary[split]={'windows':len(h),'vessels':int(h.mmsi.nunique()),'positive':int(h.y.sum()),'negative':int((h.y==0).sum()),'positive_types':poscounts.to_dict(),'noise':noise}
 for row in d.itertuples():
  assert group_map[row.mmsi]==row.split
  parent=parents.loc[row.window_id];assert parent.counterfactual_accepted
  for key in ['mmsi','split','latent_kind','onset_utc_s','behavior_seed','noise_seed','group_seed','counterfactual_row','source_file','source_sha256','timestamps','row_indices']:
   assert getattr(row,key)==parent[key],(row.window_id,key)
  assert parent.applied_params==row.counterfactual_applied_params
  if row.y==0:assert row.type==0 and row.behavior_type=='none' and row.applied_params=='{}' and row.is_sham
  else:assert row.type==KINDS.index(row.latent_kind)+1 and row.behavior_type==row.latent_kind and row.applied_params==parent.applied_params and not row.is_sham
  ii=np.array(json.loads(row.row_indices),dtype=np.int64);tt=a['timestamp'][ii];n=len(ii);j=int(row.counterfactual_row)
  assert z['window_id'][j]==row.window_id and z['mask'][j].sum()==n and np.array_equal(z['t'][j,:n],tt)
  assert np.array_equal(tt,json.loads(row.timestamps)) and np.all(a['mmsi'][ii]==row.mmsi)
  xy=np.column_stack(((a['lon'][ii]-lon0)*111195*np.cos(np.deg2rad(lat0)),(a['lat'][ii]-lat0)*111195))
  assert np.array_equal(z['original_pos'][j,:n],xy)
  assert np.array_equal(z['original_sog'][j,:n],a['sog'][ii]) and np.array_equal(z['original_cog'][j,:n],a['cog'][ii])
  dt=np.diff(tt);oldv=velocity(z['original_sog'][j,:n],z['original_cog'][j,:n]);newv=velocity(z['smooth_sog'][j,:n],z['smooth_cog'][j,:n])
  olde=np.diff(xy,axis=0)/dt[:,None]-(oldv[:-1]+oldv[1:])/2
  newe=np.diff(z['smooth_pos'][j,:n],axis=0)/dt[:,None]-(newv[:-1]+newv[1:])/2
  err=float(np.abs(newe-olde).max());assert err<1e-8,(row.window_id,err)
  audit['counterfactual_innovation_max_abs_difference_mps']=max(audit['counterfactual_innovation_max_abs_difference_mps'],err)
  assert row.generator_sha256=='d05df588da58120e78493a4703b6f8d9109d993d344315a51f1bcba9ad996882'
  split_ids[row.split].add(int(row.mmsi));split_obs[row.split].update((int(row.mmsi),int(t)) for t in tt)
  audit['source_observations_checked']+=n
 audit['regions'][region]=summary;print('VERIFIED_LEDGER',region,len(d),flush=True)
audit['cross_split_intersections']={}
for x,y in [('train','val'),('train','test'),('val','test')]:
 values={'target_mmsi':len(split_ids[x]&split_ids[y]),'raw_target_observations':len(split_obs[x]&split_obs[y])};assert not any(values.values());audit['cross_split_intersections'][x+'__'+y]=values
audit['elapsed_seconds']=time.time()-started
(OUT/'independent_label_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='regions'},indent=2))
