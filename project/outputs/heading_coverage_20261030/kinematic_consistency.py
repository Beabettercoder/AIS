from pathlib import Path
import numpy as np,json
ROOT=Path('/mnt/sda/WangZijia/AIS'); OLD=ROOT/'outputs/balanced_global_v2/relation_dual_pressure_v2_20261027';OUT=ROOT/'outputs/heading_coverage_20261030'
rows=[r for r in json.loads((OLD/'dataset_manifest_combined.json').read_text())['samples'] if r.get('version')=='relaxed' and r.get('split') in ('train','val')]
def vel(s,c):
 a=np.deg2rad(c);return s[...,None]*0.514444444444*np.stack([np.sin(a),np.cos(a)],axis=-1)
def inn(t,p,s,c):
 dt=np.diff(t);return np.diff(p,axis=0)/dt[:,None]-(vel(s,c)[1:]+vel(s,c)[:-1])/2
mx=0.;n=0;errors=[];angle_by_speed={'low':[],'moving':[]};nhead=0; mismatches=0;pressure_count=0;balanced={'target':0,'neighbor':0}
def signature(items):
 import hashlib
 h=hashlib.sha256()
 for x in items:
  x=np.asarray(x);h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
 return h.hexdigest()
heads={}
for split in ('train','val'):
 with np.load(OUT/f'heading_{split}.npz',allow_pickle=False) as z:heads[split]={k:z[k] for k in z.files}
for split in ('train','val'):
 z=heads[split];lookup={str(w):i for i,w in enumerate(z['window_id'])};assert len(lookup)==len(z['window_id'])
 for r in (x for x in rows if x['split']==split):
  i=lookup[r['window_id']]; k=int(z['target_length'][i]); lo,hi=z['sample_neighbor_offsets'][i:i+2]; th=z['target_heading'][i,:k];tv=z['target_valid'][i,:k];nh=z['neighbor_heading'][lo:hi];nv=z['neighbor_valid'][lo:hi]
  with np.load(OLD/r['files']['base']['path'],allow_pickle=False) as q:
   assert np.array_equal(q['target_t'][0],z['target_t'][i,:k]);assert np.array_equal(q['neighbor_t'][0],z['neighbor_t'][lo:hi]);assert np.array_equal(q['neighbor_mmsi'].astype(str),z['neighbor_mmsi'][lo:hi]);nhead+=int(tv.sum()+nv.sum())
   for case in range(4):
    t=q['target_t'][case]; s=q['target_sog_kn'][case]; c=q['target_cog_deg'][case];p=q['pre_noise_target_pos'][case];base=inn(q['source_target_t'],q['source_target_pos'],q['source_target_sog_kn'],q['source_target_cog_deg']);e=float(np.max(np.abs(inn(t,p,s,c)-base)));mx=max(mx,e);errors.append(e);n+=1
    for left,right in zip(q['member_offsets'][:-1],q['member_offsets'][1:]):
     sl=slice(left,right);base=inn(q['source_neighbor_t'][sl],q['source_neighbor_pos'][sl],q['source_neighbor_sog_kn'][sl],q['source_neighbor_cog_deg'][sl]);e=float(np.max(np.abs(inn(q['neighbor_t'][case,sl],q['pre_noise_neighbor_pos'][case,sl],q['neighbor_sog_kn'][case,sl],q['neighbor_cog_deg'][case,sl])-base)));mx=max(mx,e);errors.append(e);n+=1
   # Angle comparisons apply to original source only and are not fed to model.
   for heading,valid,s,c in ((th,tv,q['source_target_sog_kn'],q['source_target_cog_deg']),(nh,nv,q['source_neighbor_sog_kn'],q['source_neighbor_cog_deg'])):
    good=valid & np.isfinite(c) & (c>=0)&(c<360);angle=np.abs((heading-c+180)%360-180)
    angle_by_speed['low'].extend(angle[good&(s<2)].tolist());angle_by_speed['moving'].extend(angle[good&(s>=5)].tolist())
  for view in ('base','natural','suffix'):
   with np.load(OLD/r['files'][view]['path'],allow_pickle=False) as q:
    y=q['y'];assert np.array_equal(y,np.array([0,1,1,0]))
    for side in ('target','neighbor'):
     hd,vm=(th,tv) if side=='target' else (nh,nv)
     hashes=[signature([q[side+'_t'][c],q[side+'_pos'][c],q[side+'_sog_kn'][c],q[side+'_cog_deg'][c],hd,vm]) for c in range(4)]
     assert len(set(hashes))==2
     for h in set(hashes):assert sorted(y[np.array(hashes)==h].tolist())==[0,1]
     balanced[side]+=1
  # Additional views share MMSI/time heading, so coordinate identity carries no explicit case-label input.
  pressure_count+=1
stats={k:dict(n=len(x),median=float(np.median(x)) if x else None,p90=float(np.percentile(x,90)) if x else None,share_over45=float(np.mean(np.array(x)>45)) if x else None) for k,x in angle_by_speed.items()}
report={'sources':len(rows),'quartet_trajectories_checked':n,'max_abs_pre_noise_velocity_innovation_change_mps':mx,'n_abs_error_gt_1e-7':int(np.sum(np.array(errors)>1e-7)),'heading_valid_report_occurrences':nhead,'source_heading_minus_cog_abs_deg_by_source_sog':stats,'heading_invariant_across_four_cells_and_views_by_key':True,'target_and_neighbor_only_with_heading_balanced_per_source_view':balanced,'warning':'No independent truth or labeled real maneuvers; heading/cog angle is not a contamination label; pre-noise kinematic consistency alone does not certify unchanged heading physically consistent with injected behavior.'}
(OUT/'kinematic_consistency.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
