"""Reconstruct numerical group effects before observation noise, without relabeling."""
from pathlib import Path
import gzip,hashlib,json,time
import numpy as np
import pandas as pd
from audit_math_v2 import Background,slowdown,close

BASE=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');LAB=BASE/'labeling/labeled_v1';SOURCE=BASE/'release_20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def delta(p0,s0,c0,p1,s1,c1):
 changed=np.any(p0!=p1,axis=1)|(s0!=s1)|(c0!=c1)
 return changed,{'max_position_change_m':float(np.linalg.norm(p1-p0,axis=1).max()),'max_speed_change_kn':float(np.abs(s1-s0).max()),'max_course_change_deg':float(np.abs((c1-c0+180)%360-180).max())}

started=time.time();ledger=pd.read_parquet(LAB/'label_manifest.parquet');ledger=ledger[ledger.split=='test'];registry=json.loads((SOURCE/'source_registry.json').read_text())['regions'];rows=[];proof={}
for region,table in ledger.groupby('region',sort=True):
 dest=LAB/'paired_group_scene'/region;fr=json.loads((dest/'feature_report.json').read_text());cfg=fr['context_config'];meta=registry[region];src=Path(meta['canonical_file']);assert sha(src)==meta['sha256']==fr['source_proof']['canonical_sha256']
 d=pd.read_parquet(src,columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True);b=meta['core_bbox'];lon0=(b[0]+b[2])/2;lat0=(b[1]+b[3])/2
 bg=Background({'mmsi':d.mmsi.to_numpy().astype(str),'time_s':d.timestamp.to_numpy(np.float64),'x_m':(d.lon.to_numpy()-lon0)*111195*np.cos(np.deg2rad(lat0)),'y_m':(d.lat.to_numpy()-lat0)*111195,'sog_kn':d.sog.to_numpy(np.float64),'cog_deg':d.cog.to_numpy(np.float64)},cfg)
 cfpath=BASE/'labeling/counterfactuals_v1'/region/'counterfactual_observations.npz';assert sha(cfpath)==table.counterfactual_observations_sha256.iloc[0]
 cf=np.load(cfpath,allow_pickle=False);npzpath=dest/'test.npz';assert sha(npzpath)==fr['output_hashes']['test']['npz_sha256'];z=np.load(npzpath,allow_pickle=False)
 byid={r.window_id:r for r in table.itertuples()};assert set(z['window_id'])==set(byid)
 auditfile=dest/'test_context_audit.jsonl.gz';assert sha(auditfile)==fr['output_hashes']['test']['audit_sha256']
 with gzip.open(auditfile,'rt',encoding='utf-8') as f:
  for j,line in enumerate(f):
   md=json.loads(line);wid=md['window_id'];r=byid[wid];assert z['window_id'][j]==wid and md['y']==r.y
   k=int(r.counterfactual_row);n=int(r.n_obs);t=cf['t'][k,:n]
   original_p=cf['original_pos'][k,:n];original_s=cf['original_sog'][k,:n];original_c=cf['original_cog'][k,:n]
   prefix='smooth' if r.y else 'original';p=cf[prefix+'_pos'][k,:n];s=cf[prefix+'_sog'][k,:n];c=cf[prefix+'_cog'][k,:n]
   gm=md['group_variation'];assert gm['requested'] and gm['onset_s']==r.onset_utc_s
   p2,s2,c2=slowdown(t,p,s,c,gm['onset_s'],gm['factor'],gm['duration_s'])
   close(z['pre_noise_pos'][j,:n],p2,'group pre-noise target position',atol=1e-8,rtol=1e-12)
   close(z['pre_noise_sog_kn'][j,:n],s2,'group pre-noise target speed',atol=1e-10,rtol=1e-12)
   close(z['pre_noise_cog_deg'][j,:n],c2,'group pre-noise target course',atol=1e-10,rtol=1e-12)
   changed_target,td=delta(p,s,c,p2,s2,c2)
   ref=max(0,int(np.searchsorted(t,r.onset_utc_s,side='right'))-1)
   candidates=bg.original_candidates(t[ref],original_p[ref],original_s[ref],original_c[ref],str(r.mmsi));members={str(e['mmsi']):e for e in gm['members']}
   assert set(members)=={e['mmsi'] for e in candidates}
   local,lo=bg.snapshot(float(t[-1]));numeric=[];operator=[];details=[];changed_pairs=set()
   for ent in candidates:
    m=ent['mmsi'];good=local['mmsi']==m
    for key in ['time_s','x_m','y_m','sog_kn','cog_deg']:good&=np.isfinite(local[key])
    good&=(local['sog_kn']>=0)&(local['sog_kn']<102.2)&(local['cog_deg']>=0)&(local['cog_deg']<360)
    ii=np.flatnonzero(good)
    if len(ii)<2:assert not members[m]['effective'];continue
    tt=local['time_s'][ii];pp=np.column_stack((local['x_m'][ii],local['y_m'][ii]));ss=local['sog_kn'][ii];cc=local['cog_deg'][ii]
    pp2,ss2,cc2=slowdown(tt,pp,ss,cc,gm['onset_s'],gm['factor'],gm['duration_s']);ch,dd=delta(pp,ss,cc,pp2,ss2,cc2)
    active=bool(np.any(tt>gm['onset_s']));assert members[m]['effective']==active
    if active:operator.append(m)
    if ch.any():numeric.append(m);changed_pairs.update((m,float(when)) for when in tt[ch])
    details.append({'mmsi':m,'operator_active':active,'numeric_changed':bool(ch.any()),**dd})
   assert set(operator)==set(gm['controlled_mmsi'])
   read=set();accepted_read=set()
   for step in md['steps']:
    for nb in step['neighbors']:
     if (str(nb['mmsi']),float(nb['time_s'])) in changed_pairs:
      read.add(str(nb['mmsi']))
      if nb['accepted']:accepted_read.add(str(nb['mmsi']))
   target_changed=bool(changed_target.any())
   rows.append({'window_id':wid,'base_window_id':str(r.base_window_id),'region':region,'split':'test','y':int(r.y),'latent_kind':str(r.latent_kind),'target_numeric_changed':target_changed,'neighbor_numeric_changed_count':len(numeric),'neighbor_numeric_changed_mmsi':json.dumps(sorted(numeric)),'neighbor_numeric_changed_read_count':len(read),'neighbor_numeric_changed_read_mmsi':json.dumps(sorted(read)),'neighbor_numeric_changed_accepted_count':len(accepted_read),'neighbor_numeric_changed_accepted_mmsi':json.dumps(sorted(accepted_read)),'target_max_position_change_m':td['max_position_change_m'],'target_max_speed_change_kn':td['max_speed_change_kn'],'target_max_course_change_deg':td['max_course_change_deg'],'operator_member_count':len(operator),'operator_only_no_numeric_change_member_count':len(set(operator)-set(numeric)),'member_numeric_effect_details':json.dumps(details,separators=(',',':')),'group_factor':gm['factor'],'group_onset_s':gm['onset_s'],'group_duration_s':gm['duration_s'],'main_group_support':target_changed and len(numeric)>=4,'comparison_tolerance':0.0,'effect_stage':'after common group operator, before observation corruption; compared to original smooth/sham state'})
 proof[region]={'source_sha256':sha(src),'counterfactual_observations_sha256':sha(cfpath),'group_view_npz_sha256':sha(npzpath),'context_audit_sha256':sha(auditfile)}
 print('GROUP_NUMERIC_REPLAY',region,len(table),flush=True)
out=pd.DataFrame(rows).sort_values(['region','window_id']).reset_index(drop=True)
assert len(out)==1680 and not out.window_id.duplicated().any() and set(out.window_id)==set(ledger.window_id)
path=LAB/'group_numeric_effects.parquet';out.to_parquet(path,index=False)
report={'status':'passed','windows':len(out),'labels_sha256':sha(LAB/'label_manifest.parquet'),'sidecar_sha256':sha(path),'replay_script_sha256':sha(Path(__file__)),'independent_math_module_sha256':sha(Path(__file__).with_name('audit_math_v2.py')),'effect_stage':'Shared group slowdown before observation corruption; target baseline is the same source smooth/sham sample','comparison_tolerance':0.0,'id_list_format':'JSON string containing string MMSI identifiers','numeric_target_changed_windows':int(out.target_numeric_changed.sum()),'no_numeric_target_change_windows':int((~out.target_numeric_changed).sum()),'operator_active_neighbor_memberships':int(out.operator_member_count.sum()),'numerically_changed_neighbor_memberships':int(out.neighbor_numeric_changed_count.sum()),'operator_only_neighbor_memberships':int(out.operator_only_no_numeric_change_member_count.sum()),'main_group_support_windows':int(out.main_group_support.sum()),'main_group_support_definition':'target_numeric_changed and neighbor_numeric_changed_count>=4; additionally expose actual-read and gate-accepted counts for sensitivity analysis','regions':{r:{'windows':len(g),'target_numeric_changed':int(g.target_numeric_changed.sum()),'main_group_support':int(g.main_group_support.sum()),'numeric_neighbors':int(g.neighbor_numeric_changed_count.sum())} for r,g in out.groupby('region')},'source_hashes':proof,'elapsed_seconds':time.time()-started}
(LAB/'group_numeric_effects_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','regions']},indent=2),flush=True)
