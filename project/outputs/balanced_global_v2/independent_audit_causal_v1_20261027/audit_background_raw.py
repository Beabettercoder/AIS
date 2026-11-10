"""Sample raw parquet replay without invoking shared Source or overlay code.

Scope is windows with no effective imposed neighbor corruption; altered-neighbor
replay is separately covered by the creator's shared-Source audit, not here.
"""
from pathlib import Path
import gzip,json
import numpy as np
import pandas as pd

root=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2');fld=root/'robust_background_v1_20261027';out=root/'independent_audit_causal_v1_20261027';base=root/'labeling/labeled_v1/features_scene'
regions=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai'];registry=json.loads((root/'release_20261026/source_registry.json').read_text())['regions'];results={}
for region in regions:
 md=registry[region];src=pd.read_parquet(md['canonical_file'],columns=['mmsi','timestamp','lon','lat','sog','cog']).reset_index(drop=True)
 src['id']=src['mmsi'].astype(str);src['stable_order']=np.arange(len(src));src=src.sort_values(['id','timestamp','stable_order'],kind='stable').drop_duplicates(['id','timestamp'],keep='last').reset_index(drop=True)
 ship={s:part.reset_index(drop=True) for s,part in src.groupby('id',sort=False)}
 c=md['core_bbox'];lon=(c[0]+c[2])/2;lat=(c[1]+c[3])/2;scale=111195.;knot=1852/3600
 stats={'verified_steps':0,'verified_members':0,'max_speed_vector_error':0.,'max_weight_error':0.,'max_field_error':0.,'max_position_distance_error_m':0.,'zero_weight_fallbacks':0,'splits':{}}
 for split in ('train','val'):
  with np.load(base/region/(split+'.npz'),allow_pickle=False) as z, np.load(fld/region/(split+'.npz'),allow_pickle=False) as f:
   indexes={str(w):i for i,w in enumerate(z['window_id'])};
   input_md={}
   with gzip.open(base/region/(split+'_context_audit.jsonl.gz'),'rt',encoding='utf8') as h:
    for line in h:
     m=json.loads(line)
     if not any(ch.get('effective') for ch in m['neighbor_observation_noise']['changes']):input_md[str(m['window_id'])]=m
   checked=0
   with gzip.open(fld/region/(split+'_member_provenance.jsonl.gz'),'rt',encoding='utf8') as h:
    for line in h:
     p=json.loads(line);wid=str(p['window_id'])
     if wid not in input_md:continue
     i=indexes[wid];j=int(p['step']);now=float(p['time_s']);target=str(p['target_mmsi']);assert f['common_valid'][i,j] and z['context_valid'][i,j] and abs(now-z['time_s'][i,j])<1e-9
     source_entries={str(e['mmsi']):e for e in input_md[wid]['steps'][j]['neighbors'] if e['plain_accepted']}
     assert set(source_entries)=={str(e['mmsi']) for e in p['members']}
     velocities=[];weights=[]
     for member in p['members']:
      mid=str(member['mmsi']);stamp=float(member['time_s']);a=ship[mid];sel=np.flatnonzero(a['timestamp'].to_numpy(float)==stamp);assert len(sel)==1;ix=int(sel[0]);o=a.iloc[ix]
      assert mid!=target and 0<=now-stamp<=300 and float(a.loc[a['timestamp']<=now,'timestamp'].max())==stamp
      angle=np.deg2rad(float(o['cog']));v=np.array([float(o['sog'])*knot*np.sin(angle),float(o['sog'])*knot*np.cos(angle)])
      stats['max_speed_vector_error']=max(stats['max_speed_vector_error'],float(np.max(np.abs(v-np.array(member['velocity_mps'])))))
      position=np.array([(float(o['lon'])-lon)*scale*np.cos(np.deg2rad(lat)),(float(o['lat'])-lat)*scale]);dist=np.linalg.norm(position-z['target_pos'][i,j]);ent=source_entries[mid]
      stats['max_position_distance_error_m']=max(stats['max_position_distance_error_m'],abs(float(dist)-float(ent['distance_m'])))
      valid=False;weight=0.
      if ix>0:
       prev=a.iloc[ix-1];dt=stamp-float(prev['timestamp']);s0=float(prev['sog']);c0=float(prev['cog']);x0=float(prev['lon']);y0=float(prev['lat'])
       good0=np.isfinite([s0,c0,x0,y0]).all() and 0<=s0<102.2 and 0<=c0<360
       good1=np.isfinite([o['sog'],o['cog'],o['lon'],o['lat']]).all() and 0<=o['sog']<102.2 and 0<=o['cog']<360
       valid=bool(good0 and good1 and 0<dt<=300)
       if valid:
        prevpos=np.array([(x0-lon)*scale*np.cos(np.deg2rad(lat)),(y0-lat)*scale]);v0=s0*knot*np.array([np.sin(np.deg2rad(c0)),np.cos(np.deg2rad(c0))]);e=np.linalg.norm((position-prevpos)/dt-(v+v0)/2);weight=float(np.exp(-max(0.,e-2.5)**2/8))
      assert valid==bool(member['kinematic_valid']);stats['max_weight_error']=max(stats['max_weight_error'],abs(weight-float(member['w_abs'])))
      velocities.append(v);weights.append(weight);stats['verified_members']+=1
     v=np.asarray(velocities);w=np.asarray(weights);pred={'mean':np.mean(v,axis=0),'median':np.median(v,axis=0)}
     if w.sum()<=1e-12:pred['weighted_median']=np.median(v,axis=0);stats['zero_weight_fallbacks']+=1
     else:
      pred['weighted_median']=np.array([v[np.argsort(v[:,d],kind='stable')[np.searchsorted(np.cumsum(w[np.argsort(v[:,d],kind='stable')]),w.sum()/2,side='left')],d] for d in range(2)])
     a=np.deg2rad(float(z['target_cog_deg'][i,j]));tv=float(z['target_sog_kn'][i,j])*knot*np.array([np.sin(a),np.cos(a)])
     for name in pred:
      stats['max_field_error']=max(stats['max_field_error'],float(np.max(np.abs(pred[name]-f['background_'+name+'_mps'][i,j]))),float(np.max(np.abs(tv-pred[name]-f['residual_'+name+'_mps'][i,j]))))
     checked+=1;stats['verified_steps']+=1
     if checked>=40:break
   stats['splits'][split]={'verified_steps':checked,'scope':'neighbor windows without effective controlled neighbor corruption'}
 assert stats['verified_steps']>0
 results[region]=stats
 assert stats['max_weight_error']<1e-8 and stats['max_speed_vector_error']<1e-8 and stats['max_field_error']<1e-4 and stats['max_position_distance_error_m']<1e-6
 print(region,json.dumps(stats),flush=True)
result={'status':'passed','scope':'raw parquet direct replay, up to 40 common support steps per region and split, with no effective imposed neighbor corruption; modified-neighbor Source replay not independently checked', 'regions':results}
(out/'audit_background_raw.json').write_text(json.dumps(result,indent=2),encoding='utf8')
