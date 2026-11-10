"""Independent replay of controller conditions from observed train/val inputs."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
LAB=ROOT/'labeling/labeled_v1/features_scene'
BRIDGE=ROOT/'gating_training_v2_20261027/data'
RUN=ROOT/'factorial_ablation_v1_20261027'
OUT=ROOT/'factorial_independent_audit_20261027'
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
SPLITS=('train','val')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

all_parts={s:[] for s in SPLITS};splits_ids={};max_e=0.;max_w=0.;max_speed=0.
for split in SPLITS:
    split_ids={k:set() for k in ('mmsi','window_id','base_window_id')}
    for region in REGIONS:
        with np.load(LAB/region/f'{split}.npz',allow_pickle=False) as z:
            s={k:z[k] for k in ('mask','time_s','time_interval_valid','position_increment_valid','kinematic_valid','target_pos','target_sog_kn','target_cog_deg','e_kin_mps','self_gate_weight','mmsi','window_id','base_window_id')}
        with np.load(BRIDGE/region/f'{split}.npz',allow_pickle=False) as z:
            assert np.array_equal(s['window_id'],z['window_id'])
            raw=z['raw_legacy7'].copy();bg=z['ep'].copy();mw=z['w_self'].copy()
        for k in split_ids: split_ids[k].update(s[k].astype(str))
        mask=s['mask'];n,T=mask.shape
        speed=s['target_sog_kn']*(1852/3600)
        angle=np.deg2rad(s['target_cog_deg'])
        rep=np.stack((speed*np.sin(angle),speed*np.cos(angle)),axis=-1)
        dt=np.diff(s['time_s'],axis=1)
        ev=np.zeros((n,T),float)
        vpos=np.divide(np.diff(s['target_pos'],axis=1),dt[...,None],out=np.zeros((n,T-1,2),float),where=(dt[...,None]>0))
        ev[:,1:]=np.linalg.norm(vpos-(rep[:,1:]+rep[:,:-1])/2,axis=-1)
        kin=mask&s['kinematic_valid'];
        if kin.any():
            max_e=max(max_e,float(np.max(np.abs(ev[kin]-s['e_kin_mps'][kin]))))
            expected=np.exp(-np.maximum(0,ev[kin]-2.5)**2/8)
            max_w=max(max_w,float(np.max(np.abs(expected-s['self_gate_weight'][kin]))))
        raw_speed=np.linalg.norm(raw[...,2:4],axis=-1)
        max_speed=max(max_speed,float(np.max(np.abs(raw_speed[mask]-speed[mask]))))
        assert np.allclose(mw[mask],s['self_gate_weight'][mask],atol=0,rtol=0)
        acceleration=np.zeros((n,T),float)
        ok=mask[:,1:]&mask[:,:-1]&s['time_interval_valid'][:,1:]&np.isfinite(dt)&(dt>0)&np.isfinite(np.diff(raw_speed,axis=1))
        # Deliberately use the original float32 speed norm, since it defines the frozen experimental precision.
        acceleration[:,1:]=np.divide(np.abs(np.diff(raw_speed,axis=1)),dt,out=np.zeros((n,T-1)),where=ok)
        x=np.stack((raw_speed,acceleration,s['e_kin_mps'],s['self_gate_weight']),axis=-1)
        valid=np.zeros((n,T,4),bool);valid[...,0]=mask&np.isfinite(raw_speed);valid[:,1:,1]=ok
        valid[...,2]=kin&np.isfinite(s['e_kin_mps']);valid[...,3]=kin&np.isfinite(s['self_gate_weight'])
        xv=np.where(valid,x,0).astype(np.float32)
        all_parts[split].append(dict(ids=s['window_id'],selfx=xv,self_valid=valid,bg=bg,mask=mask))
    splits_ids[split]=split_ids
assert all(not (splits_ids['train'][k]&splits_ids['val'][k]) for k in splits_ids['train'])

reported=json.loads((RUN/'condition_stats.json').read_text(encoding='utf8'))
stats={}
for kind in ('self','background'):
    a=np.concatenate([p['selfx' if kind=='self' else 'bg'] for p in all_parts['train']]);
    v=np.concatenate([p['self_valid'] if kind=='self' else np.broadcast_to(p['mask'][...,None],p['bg'].shape) for p in all_parts['train']])
    c=[]
    for j in range(4):
        x=a[...,j][v[...,j]].astype(np.float64);c.append((float(x.mean()),float(max(x.std(),1e-6)),len(x)))
        assert abs(c[-1][0]-reported[kind][j]['mean'])<1e-7
        assert abs(c[-1][1]-reported[kind][j]['std'])<1e-7
        assert c[-1][2]==reported[kind][j]['eligible_count']
    stats[kind]=c

proof={};maxdiff=0.
for split in SPLITS:
    ids=np.concatenate([p['ids'] for p in all_parts[split]])
    with np.load(RUN/f'{split}_conditions.npz',allow_pickle=False) as z:
        assert np.array_equal(ids,z['window_id'])
        for kind in ('self','background'):
            a=np.concatenate([p['selfx' if kind=='self' else 'bg'] for p in all_parts[split]])
            v=np.concatenate([p['self_valid'] if kind=='self' else np.broadcast_to(p['mask'][...,None],p['bg'].shape) for p in all_parts[split]])
            assert np.array_equal(v,z[kind+'_valid'])
            mu=np.array([x[0] for x in stats[kind]],np.float32);sd=np.array([x[1] for x in stats[kind]],np.float32)
            normalized=np.clip((a-mu)/sd,-5,5).astype(np.float32);normalized[~v]=0
            diff=float(np.max(np.abs(normalized-z[kind+'_ep'])));maxdiff=max(maxdiff,diff)
            assert diff<1e-5
    proof[split]={'windows':len(ids),'file_sha256':sha(RUN/f'{split}_conditions.npz'),'unique_mmsi':len(splits_ids[split]['mmsi'])}
assert max_e<0.001 and max_w<1e-5 and max_speed<1e-5
out={'status':'passed','scope':'observed target state, training-only scalar stats, train/validation alignment; independent AIS vector reconstruction',
     'split_proof':proof,'target_mmsi_and_ids_intersection':0,'max_independent_kinematic_residual_error_mps':max_e,
     'max_independent_gate_error':max_w,'max_reported_speed_bridge_error_mps':max_speed,
     'max_normalized_controller_ep_error':maxdiff,'stats':stats,'protocol_sha256':sha(RUN/'protocol.json'),
     'limitations':'background ep originates from previously audited bridge; this run checks normalization rather than independent neighbor reconstruction; eligibility flags are reused',
     'script_sha256':sha(Path(__file__))}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'audit_condition.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(out,indent=2,ensure_ascii=False))
