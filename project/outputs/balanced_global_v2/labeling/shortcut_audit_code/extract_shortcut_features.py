"""Pre-noise nuisance features, with independent all-sample invariants."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/labeling')
KN=1852/3600
EPS=1e-6
PREFIX_NAMES=['prefix_n_obs','prefix_duration_s','onset_from_first_report_s',
 'prefix_sog_mean_kn','prefix_sog_std_kn','prefix_sog_terminal_kn',
 'prefix_abs_course_change_mean_deg','prefix_abs_course_change_std_deg',
 'prefix_abs_course_change_max_deg','prefix_net_course_change_deg',
 'prefix_innovation_mean_mps','prefix_innovation_std_mps',
 'prefix_innovation_p95_mps','prefix_innovation_max_mps','prefix_innovation_nearzero_fraction']
FULL_NAMES=['source_innovation_mean_mps','source_innovation_std_mps',
 'source_innovation_p95_mps','source_innovation_max_mps','source_innovation_nearzero_fraction']


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def velocity(s,c):
 r=np.asarray(c)*np.pi/180
 return np.asarray(s)[:,None]*KN*np.column_stack([np.sin(r),np.cos(r)])


def innovation(t,p,s,c):
 v=velocity(s,c)
 return np.diff(p,axis=0)/np.diff(t)[:,None]-(v[:-1]+v[1:])*.5


def summary(x):
 if not len(x):raise ValueError('No valid innovation pairs')
 return [float(np.mean(x)),float(np.std(x)),float(np.quantile(x,.95)),float(np.max(x)),float(np.mean(x<=EPS))]


def require(ok,message):
 if not ok:raise ValueError(message)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--labels',type=Path,default=ROOT/'labeled_v1/label_manifest.parquet');ap.add_argument('--out',type=Path,default=ROOT/'labeled_v1/shortcut_audit');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 started=time.time();label_hash=sha(args.labels);labels=pd.read_parquet(args.labels)
 require(len(labels)==11200 and not labels.sample_id.duplicated().any(),'Unexpected or repeated label samples')
 require(labels.counterfactual_accepted.all() and labels.labels_frozen.all(),'Unaccepted/unfrozen label input')
 require(np.array_equal(labels.is_sham.to_numpy(bool),labels.y.to_numpy()==0),'Sham/binary labels disagree')
 independent=ROOT/'counterfactuals_v1/independent_counterfactual_audit.json';previous=json.loads(independent.read_text())
 require(previous['status']=='passed','Counterfactual source audit not passed')
 prefix=[];full=[];records=[];source_hashes={};max_error=0.;prefix_checked=0;prefix_reports=0;innovation_pairs=0;regions={}
 for region,g in labels.groupby('region',sort=True):
  paths=set(g.counterfactual_observations_file);require(len(paths)==1,'Multiple untracked regional source arrays')
  path=Path(paths.pop());actual=sha(path);require(set(g.counterfactual_observations_sha256)=={actual},'Changed counterfactual arrays')
  previous_hash=previous['regions'][region]['input_hashes']['counterfactual_observations.npz'];require(actual==previous_hash,'No matching original-source audit chain')
  source_hashes[str(path)]=actual
  with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
  rp=0;mp=0.
  for row in g.itertuples():
   j=int(row.counterfactual_row);require(arrays['window_id'][j]==row.window_id,'Counterfactual row index/identity mismatch')
   n=int(row.n_obs);require(np.array_equal(arrays['mask'][j],np.arange(20)<n),'Incorrect source mask')
   t=arrays['t'][j,:n];p=arrays['original_pos'][j,:n];s=arrays['original_sog'][j,:n];c=arrays['original_cog'][j,:n]
   sp=arrays['smooth_pos'][j,:n];ss=arrays['smooth_sog'][j,:n];sc=arrays['smooth_cog'][j,:n]
   onset=float(row.onset_utc_s);pre=t<onset;count=int(pre.sum())
   require(count>=3 and count<n,'Invalid prefix')
   # Both latent counterfactual and actually applied y-dependent observations
   # must preserve all pre-onset original fields exactly.
   for before,after in [(p,sp),(s,ss),(c,sc)]:require(np.array_equal(before[pre],after[pre]),'Injected maneuver changed source prefix')
   actual_p,actual_s,actual_c=(sp,ss,sc) if row.y else (p,s,c)
   for before,after in [(p,actual_p),(s,actual_s),(c,actual_c)]:require(np.array_equal(before[pre],after[pre]),'Applied label altered prefix')
   if not row.y:
    require(np.array_equal(actual_p,p) and np.array_equal(actual_s,s) and np.array_equal(actual_c,c),'Sham source altered')
   orig_innov=innovation(t,p,s,c);smooth_innov=innovation(t,sp,ss,sc)
   err=float(np.linalg.norm(orig_innov-smooth_innov,axis=1).max());require(err<1e-7,'Smooth innovations differ from original')
   max_error=max(max_error,err);mp=max(mp,err);innovation_pairs+=len(orig_innov)
   actual_innov=innovation(t,actual_p,actual_s,actual_c);require(np.allclose(actual_innov,orig_innov,atol=1e-7,rtol=0),'Applied-label innovation changed')
   tt=t[pre];sg=s[pre];course=c[pre];dc=(np.diff(course)+180)%360-180
   pinv=innovation(tt,p[pre],sg,course);valid=np.diff(tt)<=300
   require(valid.any(),'No shared-gate-valid prefix pair')
   ps=summary(np.linalg.norm(pinv[valid],axis=1))
   features=[count,float(tt[-1]-tt[0]),float(onset-t[0]),float(np.mean(sg)),float(np.std(sg)),float(sg[-1]),float(np.mean(np.abs(dc))),float(np.std(np.abs(dc))),float(np.max(np.abs(dc))),float(np.sum(dc))]+ps
   whole_valid=np.diff(t)<=300;whole=summary(np.linalg.norm(orig_innov[whole_valid],axis=1))
   prefix.append(features);full.append(whole)
   records.append({'sample_id':str(row.sample_id),'window_id':str(row.window_id),'region':region,'split':row.split,'mmsi':int(row.mmsi),'y':int(row.y),'onset_utc_s':onset,'latent_kind':row.latent_kind})
   prefix_checked+=1;prefix_reports+=count;rp+=count
  regions[region]={'windows':len(g),'prefix_reports_checked':rp,'maximum_vector_innovation_error_mps':mp,'class_counts':g.groupby('split').y.agg(['count','sum']).to_dict(orient='index')}
  print('EXTRACTED_AND_VERIFIED',region,len(g),flush=True)
 Xp=np.asarray(prefix,dtype=np.float64);Xf=np.asarray(full,dtype=np.float64)
 require(Xp.shape==(len(labels),len(PREFIX_NAMES)) and Xf.shape==(len(labels),len(FULL_NAMES)),'Feature shape mismatch')
 require(np.isfinite(Xp).all() and np.isfinite(Xf).all(),'Nonfinite nuisance features')
 payload={'prefix_X':Xp,'innovation_X':Xf,'prefix_feature_names':np.array(PREFIX_NAMES),'innovation_feature_names':np.array(FULL_NAMES)}
 for key in records[0]:payload[key]=np.array([r[key] for r in records])
 output=args.out/'pre_noise_nuisance_features.npz';np.savez_compressed(output,**payload)
 pd.DataFrame(records).to_parquet(args.out/'audit_sample_index.parquet',index=False)
 report={'status':'passed','scope':'pre-noise only; original and latent smooth counterfactuals; no post-noise feature access','windows':len(labels),'prefix_windows_checked':prefix_checked,'prefix_reports_checked':prefix_reports,'full_window_vector_innovation_pairs_checked':innovation_pairs,'maximum_vector_innovation_error_mps':max_error,'all_pre_onset_source_states_equal':True,'all_smooth_vector_innovations_preserved':True,'all_applied_pre_noise_innovations_preserved':True,'normal_pre_noise_observations_are_original':True,'features':{'prefix':PREFIX_NAMES,'full_source_innovation':FULL_NAMES},'innovation_definition':'position difference / actual dt minus endpoint-average reported velocity, m/s; only 0<dt<=300 pairs used as model features','nearzero_threshold_mps':EPS,'regions':regions,'input_hashes':{'label_manifest':label_hash,'independent_counterfactual_audit':sha(independent),'counterfactual_array_files':source_hashes,'extractor_script':sha(Path(__file__))},'output_npz_sha256':sha(output),'elapsed_seconds':time.time()-started}
 require(sha(args.labels)==label_hash,'Label manifest changed during extraction')
 (args.out/'pre_noise_invariant_audit.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({'status':'passed','windows':len(labels),'max_innovation_error':max_error,'output':str(output)},indent=2),flush=True)


if __name__=='__main__':main()
