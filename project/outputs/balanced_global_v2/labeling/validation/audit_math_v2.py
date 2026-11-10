"""Independent numerical replay. Does not import builder or generator code."""
import hashlib
import numpy as np

KNOT=1852/3600
COLS=('mmsi','time_s','x_m','y_m','sog_kn','cog_deg')

def close(a,b,name,atol=3e-5,rtol=3e-5):
 if not np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True):
  raise AssertionError(f'{name}: numerical mismatch, max={np.nanmax(np.abs(np.asarray(a)-np.asarray(b)))}')
def stable(seed,tag):return int.from_bytes(hashlib.sha256(f'{int(seed)}:{tag}'.encode()).digest()[:8],'little')
def velocity(s,c):
 a=np.deg2rad(c);return np.asarray(s)[...,None]*KNOT*np.stack((np.sin(a),np.cos(a)),axis=-1)
def innovation(t,p,s,c):
 v=velocity(s,c);return np.diff(p,axis=0)/np.diff(t)[:,None]-(v[:-1]+v[1:])/2
def apply_delta(t,p,old_s,old_c,new_s,new_c):
 dv=velocity(new_s,new_c)-velocity(old_s,old_c);off=np.zeros_like(p)
 off[1:]=np.cumsum((dv[:-1]+dv[1:])/2*np.diff(t)[:,None],axis=0)
 return p+off
def ramp(t,onset,duration,hard=False):
 if hard:return (t>=onset).astype(float)
 u=np.clip((t-onset)/duration,0,1);return u*u*(3-2*u)
def behavior(t,p,s,c,kind,onset,params,hard=False):
 if kind=='none':return p.copy(),s.copy(),c.copy()
 q=params;r=ramp(t,onset,q['duration_s'],hard);speed=s.copy();delta=np.zeros(len(t))
 if kind=='turn_back':delta=q['sign']*q['delta_deg']*r
 elif kind=='deviation':delta=q['sign']*q['delta_deg']*(r-ramp(t,onset+q['duration_s']+q['hold_s'],q['duration_s'],hard))
 elif kind=='stall':speed=s*(1-r)+q['target_kn']*r
 elif kind=='speed_drop':speed=s*(1-(1-q['factor'])*r)
 elif kind=='sinuous':
  end=onset+q['active_duration_s'];down=end if hard else end-q['ramp_down_s']
  delta=q['sign']*q['amplitude_deg']*r*(1-ramp(t,down,q['ramp_down_s'],hard))*np.sin(2*np.pi*(t-onset)/q['period_s']+q['phase_rad'])
 else:raise ValueError(kind)
 course=(c+delta)%360
 return apply_delta(t,p,s,c,speed,course),speed,course
def slowdown(t,p,s,c,onset,factor,duration):
 speed=s*(1-(1-factor)*ramp(t,onset,duration));return apply_delta(t,p,s,c,speed,c),speed,c.copy()
def noise(t,p,s,c,kind,params):
 p=p.copy();s=s.copy();c=c.copy()
 if kind=='none':return p,s,c
 if kind=='teleport':
  p[int(params['report_index'])]+=params['offset_m']*np.array([np.sin(np.deg2rad(params['angle_deg'])),np.cos(np.deg2rad(params['angle_deg']))])
 elif kind=='speed_spike':
  i=int(params['report_index']);j=min(len(t),i+int(params['reports']));s[i:j]=np.minimum(s[i:j]*params['factor'],102.1)
 elif kind=='drift':
  u=(t-params['onset_s'])/params['duration_s'];env=np.where((u>=0)&(u<=1),.5-.5*np.cos(2*np.pi*np.clip(u,0,1)),0.)
  p+=params['offset_m']*env[:,None]*np.array([np.sin(np.deg2rad(params['angle_deg'])),np.cos(np.deg2rad(params['angle_deg']))])
 else:raise ValueError(kind)
 return p,s,c

def target_kin(t,p,s,c,cfg):
 n=len(t);dt=np.r_[0,np.diff(t)];dp=np.vstack([np.zeros(2),np.diff(p,axis=0)]);v=velocity(s,c)
 valid=np.r_[False,(np.diff(t)>0)&(np.diff(t)<=cfg['max_kinematic_gap_seconds'])]
 error=np.zeros(n);error[valid]=np.linalg.norm(dp[valid]/dt[valid,None]-(v[np.flatnonzero(valid)-1]+v[valid])/2,axis=1)
 w=np.zeros(n);w[valid]=np.exp(-np.maximum(0,error[valid]-cfg['gate_deadzone_mps'])**2/cfg['gate_denominator_mps2'])
 gate=np.where(valid,w,1.);R=cfg['radius_nm']*1852;vs=cfg['velocity_scale_mps']
 raw=np.column_stack((dp/R,v/vs,dt/cfg['window_seconds'],np.sin(np.deg2rad(c)),np.cos(np.deg2rad(c))))
 gated=raw.copy();gated[:,[0,1,2,3,5,6]]*=gate[:,None]
 return dict(raw=raw,gated=gated,e_kin_mps=error,self_reliability=w,self_gate_weight=gate,kinematic_valid=valid,time_interval_valid=np.arange(n)>0,position_increment_valid=np.arange(n)>0,velocity=v)

def ordered(d):
 a={k:np.asarray(d[k],dtype=str if k=='mmsi' else float).copy() for k in COLS}
 order=np.lexsort((np.arange(len(a['time_s'])),a['time_s'],a['mmsi']))
 for k in a:a[k]=a[k][order]
 keep=np.r_[(a['mmsi'][:-1]!=a['mmsi'][1:])|(a['time_s'][:-1]!=a['time_s'][1:]),True]
 for k in a:a[k]=a[k][keep]
 order=np.argsort(a['time_s'],kind='stable')
 return {k:v[order] for k,v in a.items()}
def fingerprint(d):
 h=hashlib.sha256()
 for k in COLS:h.update(k.encode());h.update(d[k].tobytes())
 return h.hexdigest()
def bg_kin(d,cfg):
 n=len(d['time_s']);order=np.lexsort((d['time_s'],d['mmsi']));v=velocity(np.nan_to_num(d['sog_kn']),np.nan_to_num(d['cog_deg']))
 obs=np.isfinite(d['x_m'])&np.isfinite(d['y_m'])&np.isfinite(d['sog_kn'])&np.isfinite(d['cog_deg'])&(d['sog_kn']>=0)&(d['sog_kn']<102.2)&(d['cog_deg']>=0)&(d['cog_deg']<360)
 valid=np.zeros(n,bool);error=np.zeros(n);w=np.zeros(n)
 j=order[1:];prev=order[:-1];dt=d['time_s'][j]-d['time_s'][prev]
 good=obs[j]&obs[prev]&(d['mmsi'][j]==d['mmsi'][prev])&(dt>0)&(dt<=cfg['max_kinematic_gap_seconds'])
 j=j[good];prev=prev[good];dt=d['time_s'][j]-d['time_s'][prev];valid[j]=True
 delta=np.column_stack((d['x_m'][j]-d['x_m'][prev],d['y_m'][j]-d['y_m'][prev]))
 error[j]=np.linalg.norm(delta/dt[:,None]-(v[j]+v[prev])/2,axis=1)
 w[j]=np.exp(-np.maximum(0,error[j]-cfg['gate_deadzone_mps'])**2/cfg['gate_denominator_mps2'])
 return dict(observed=obs,kinematic_valid=valid,e_kin_mps=error,reliability=w,velocity=v)

class Background:
 def __init__(self,data,cfg):
  self.data=ordered(data);self.cfg=cfg;self.kin=bg_kin(self.data,cfg);self.fingerprint=fingerprint(self.data);self.latest_cache={}
 def latest(self,t):
  if t not in self.latest_cache:
   ts=self.data['time_s'];lo=np.searchsorted(ts,t-self.cfg['freshness_seconds']);hi=np.searchsorted(ts,t,side='right');ii=np.arange(lo,hi)
   if len(ii):
    _,r=np.unique(self.data['mmsi'][ii][::-1],return_index=True);ii=ii[len(ii)-1-r]
   self.latest_cache[t]=ii
  return self.latest_cache[t]
 def snapshot(self,end):
  t=self.data['time_s'];lo=np.searchsorted(t,end-self.cfg['window_seconds']-self.cfg['freshness_seconds']-self.cfg['max_kinematic_gap_seconds']);hi=np.searchsorted(t,end,side='right')
  return {k:v[lo:hi].copy() for k,v in self.data.items()},int(lo)
 def states(self,ii,override=None):
  out={k:self.data[k][ii].copy() for k in COLS};out.update({k:v[ii].copy() for k,v in self.kin.items()})
  if override is not None and len(override['rows']):
   jj=np.searchsorted(override['rows'],ii);ok=jj<len(override['rows']);ok[ok]&=override['rows'][jj[ok]]==ii[ok]
   for k in out:out[k][ok]=override[k][jj[ok]]
  return out
 def original_candidates(self,t,p,s,c,identity):
  ii=self.latest(float(t));ii=ii[self.data['mmsi'][ii]!=identity];st=self.states(ii)
  dist=np.hypot(st['x_m']-p[0],st['y_m']-p[1]);ang=np.abs((st['cog_deg']-c+180)%360-180)
  ok=np.isfinite(st['x_m'])&np.isfinite(st['y_m'])&np.isfinite(st['sog_kn'])&np.isfinite(st['cog_deg'])&(st['sog_kn']>=self.cfg['min_heading_speed_kn'])&(st['sog_kn']<102.2)&(st['cog_deg']>=0)&(st['cog_deg']<360)&(dist<=self.cfg['radius_nm']*1852)&(ang<=self.cfg['heading_limit_deg'])
  return sorted([{'mmsi':str(m),'time_s':float(tt),'distance_m':float(dd)} for m,tt,dd in zip(st['mmsi'][ok],st['time_s'][ok],dist[ok])],key=lambda x:x['mmsi'])

def finite_rows(local,identity):
 good=local['mmsi']==identity
 for k in ['time_s','x_m','y_m','sog_kn','cog_deg']:good&=np.isfinite(local[k])
 good&=(local['sog_kn']>=0)&(local['sog_kn']<102.2)&(local['cog_deg']>=0)&(local['cog_deg']<360)
 return np.flatnonzero(good)

def noise_spec(t,row):
 rng=np.random.default_rng(stable(row.noise_seed,'query_and_target_noise'));ix=int(rng.integers(max(1,len(t)//2),len(t)));q=float(t[ix]);kind=row.noise_kind;p={}
 if kind=='teleport':p={'report_index':ix,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
 elif kind=='speed_spike':p={'report_index':ix,'reports':max(1,int(np.sum((t>=q)&(t<q+120)))),'factor':float(rng.uniform(3,5))}
 elif kind=='drift':p={'onset_s':q-150,'duration_s':300.,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
 return {'requested':bool(row.observation_noise),'kind':kind,'scope':row.noise_scope,'query_step':ix,'query_time_s':q,'seed':int(row.noise_seed),'target_parameters':p,'target_requested':bool(row.observation_noise) and row.noise_scope in ['target_only','both'],'neighbors_requested':bool(row.observation_noise) and row.noise_scope in ['neighbors_only','both']}

def replay_background(base,t,p,s,c,row,spec,group,md):
 """Replay actual changes, recalculate the snapshot hash, retain sparse overlay."""
 gm=md['group_variation'];nm=md['neighbor_observation_noise'];identity=str(row.mmsi)
 if not group and not spec['neighbors_requested']:
  assert not gm['requested'] and not nm['requested']
  assert md['background_fingerprint']==base.fingerprint
  return None,{'fingerprint_recomputed':False,'changed_background':False,'changed_noise_pairs':set()}
 local,lo=base.snapshot(float(t[-1]));changed_ids=set();numeric_changed_ids=set();changed_noise_pairs=set();group_expected=[];operator_only_members=[]
 if group:
  factor=float(np.random.default_rng(stable(row.group_seed,'shared_group_slowdown')).uniform(.65,.85));onset=float(row.onset_utc_s);duration=300.
  close([gm['factor'],gm['onset_s'],gm['duration_s']],[factor,onset,duration],'group parameters',atol=1e-10,rtol=0)
  ref=max(0,int(np.searchsorted(t,onset,side='right'))-1);assert gm['reference_source_step']==ref
  candidates=base.original_candidates(t[ref],p[ref],s[ref],c[ref],identity)
  members={str(x['mmsi']):x for x in gm['members']};assert set(members)=={x['mmsi'] for x in candidates}
  for ent in candidates:
   m=ent['mmsi'];ii=finite_rows(local,m);record=members[m]
   if len(ii)<2:assert not record['effective'];continue
   tt=local['time_s'][ii];pp=np.column_stack((local['x_m'][ii],local['y_m'][ii]));ss=local['sog_kn'][ii];cc=local['cog_deg'][ii]
   pp2,ss2,cc2=slowdown(tt,pp,ss,cc,onset,factor,duration)
   effective=not(np.array_equal(pp,pp2) and np.array_equal(ss,ss2) and np.array_equal(cc,cc2))
   operator_active=bool(np.any(ramp(tt,onset,duration)>0))
   # Frozen builder v1 called an activated multiplier "effective", including
   # stationary post-onset reports whose numerical observations do not change.
   # Audit the actual values and expose this distinction; never silently count
   # a zero numerical change as an observed group perturbation.
   assert record['effective']==operator_active
   close([record['factor'],record['onset_s'],record['duration_s']],[factor,onset,duration],'group member shared control',atol=1e-10,rtol=0)
   assert record['affected_time_s']==tt[tt>=onset].tolist()
   local['x_m'][ii]=pp2[:,0];local['y_m'][ii]=pp2[:,1];local['sog_kn'][ii]=ss2;local['cog_deg'][ii]=cc2
   if operator_active:changed_ids.add(m);group_expected.append(m)
   if effective:numeric_changed_ids.add(m)
   if operator_active and not effective:operator_only_members.append(m)
  assert set(gm['controlled_mmsi'])==set(group_expected)
 if spec['neighbors_requested']:
  qi=spec['query_step'];candidates=base.original_candidates(t[qi],p[qi],s[qi],c[qi],identity)
  assert nm['candidate_mmsi']==[e['mmsi'] for e in candidates]
  count=min(len(candidates),max(1,int(np.ceil(.15*len(candidates))))) if candidates else 0
  chosen=sorted(np.random.default_rng(stable(spec['seed'],'neighbor_selection')).permutation(len(candidates))[:count].tolist())
  entries=[candidates[j] for j in chosen];assert nm['selected_mmsi']==[e['mmsi'] for e in entries]
  changes={str(x['mmsi']):x for x in nm['changes']};assert set(changes)==set(nm['selected_mmsi'])
  for ent in entries:
   m=ent['mmsi'];ii=finite_rows(local,m);record=changes[m]
   if len(ii)<2:assert not record['effective'];continue
   tt=local['time_s'][ii];pp=np.column_stack((local['x_m'][ii],local['y_m'][ii]));ss=local['sog_kn'][ii].copy();cc=local['cog_deg'][ii].copy();anchor=int(np.searchsorted(tt,ent['time_s']))
   assert anchor<len(ii) and tt[anchor]==ent['time_s']
   rng=np.random.default_rng(stable(spec['seed'],'neighbor:'+m));params={}
   if spec['kind']=='teleport':params={'report_index':anchor,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
   elif spec['kind']=='speed_spike':params={'report_index':anchor,'reports':max(1,int(np.sum((tt>=ent['time_s'])&(tt<ent['time_s']+120)))),'factor':float(rng.uniform(3,5))}
   else:params={'onset_s':ent['time_s']-150,'duration_s':300.,'offset_m':float(rng.uniform(500,1500)),'angle_deg':float(rng.uniform(0,360))}
   for k,v in params.items():close(record['parameters'][k],v,'neighbor noise parameters',atol=1e-10,rtol=0)
   pp2,ss2,cc2=noise(tt,pp,ss,cc,spec['kind'],params);changed=np.any(pp2!=pp,axis=1)|(ss2!=ss)|(cc2!=cc);effective=bool(changed.any())
   assert record['effective']==effective and record['affected_time_s']==tt[changed].tolist()
   assert record['original_latest_query_report_s']==ent['time_s'] and record['target_query_time_s']==spec['query_time_s']
   local['x_m'][ii]=pp2[:,0];local['y_m'][ii]=pp2[:,1];local['sog_kn'][ii]=ss2;local['cog_deg'][ii]=cc2
   if effective:changed_ids.add(m);numeric_changed_ids.add(m);changed_noise_pairs.update((m,float(when)) for when in tt[changed])
  assert nm['effective']==bool(changed_noise_pairs)
 assert gm['source_background_fingerprint']==base.fingerprint
 if not changed_ids:
  assert md['background_fingerprint']==base.fingerprint
  return None,{'fingerprint_recomputed':False,'changed_background':False,'changed_noise_pairs':changed_noise_pairs,'numeric_changed_ids':[],'group_operator_only_members':operator_only_members}
 # BackgroundIndex receives an already time/identity-ordered, immutable-ID
 # snapshot. Updates never alter identities or times, so its final ordering is
 # exactly this order; dtype, including Unicode width, is deliberately retained.
 newhash=fingerprint(local);assert newhash!=base.fingerprint
 assert md['background_fingerprint']==newhash==gm['augmented_background_fingerprint']
 kk=bg_kin(local,base.cfg);keep=np.isin(local['mmsi'],np.array(sorted(changed_ids)));rows=lo+np.flatnonzero(keep)
 overlay={'rows':rows};overlay.update({k:v[keep] for k,v in local.items()});overlay.update({k:v[keep] for k,v in kk.items()})
 return overlay,{'fingerprint_recomputed':True,'changed_background':bool(numeric_changed_ids),'changed_noise_pairs':changed_noise_pairs,'numeric_changed_ids':sorted(numeric_changed_ids),'group_operator_only_members':operator_only_members}

def affine(pos,vel,weights,cfg):
 if len(pos)<cfg['min_neighbors']:return False,np.zeros(2),0.,0.
 w=weights/weights.sum();center=pos-np.sum(w[:,None]*pos,axis=0)
 if np.linalg.eigvalsh(center.T@(w[:,None]*center))[0]<cfg['min_geometry_spread_m']**2:return False,np.zeros(2),0.,0.
 R=cfg['radius_nm']*1852;X=np.column_stack((np.ones(len(pos)),pos/R));gram=X.T@(w[:,None]*X)
 cond=np.linalg.cond(gram)
 if not np.isfinite(cond) or cond>cfg['max_design_condition']:return False,np.zeros(2),0.,0.
 theta=np.linalg.solve(gram+cfg['ridge']*np.diag([0.,1.,1.]),X.T@(w[:,None]*vel))
 return True,theta[0],float((theta[1,0]+theta[2,1])/R),float((theta[1,1]-theta[2,0])/R)

def verify_step(base,overlay,step,t,p,s,c,v,gate,w,error,kinvalid,boundary,raw,gated,cfg,arrays,j,k):
 """Check the complete actual candidate pool, weights and both field arms."""
 ii=base.latest(float(t));ii=ii[base.data['mmsi'][ii]!=step['target_identity']]
 st=base.states(ii,overlay);dist=np.hypot(st['x_m']-p[0],st['y_m']-p[1]);spatial=np.isfinite(dist)&(dist<=cfg['radius_nm']*1852)
 if s<cfg['min_heading_speed_kn']:spatial[:]=False
 for name in st:st[name]=st[name][spatial]
 dist=dist[spatial];expected_ids=st['mmsi'].tolist();actual=step['neighbors'];seen=[str(x['mmsi']) for x in actual]
 assert len(set(seen))==len(seen) and set(seen)==set(expected_ids),'Candidate pool differs from causal latest-source spatial pool'
 lookup={m:i for i,m in enumerate(expected_ids)};order=np.array([lookup[m] for m in seen],dtype=int)
 for name in st:st[name]=st[name][order]
 dist=dist[order]
 close([x['time_s'] for x in actual],st['time_s'],'latest causal neighbor times',atol=0,rtol=0)
 close([x['age_s'] for x in actual],t-st['time_s'],'neighbor ages',atol=1e-9,rtol=0)
 assert np.all(st['time_s']<=t) and np.all(t-st['time_s']<=cfg['freshness_seconds'])
 hd=np.abs((st['cog_deg']-c+180)%360-180);dw=np.exp(-.5*(dist/(cfg['distance_sigma_nm']*1852))**2)
 plain=st['observed']&(st['sog_kn']>=cfg['min_heading_speed_kn'])&(hd<=cfg['heading_limit_deg'])
 accepted=plain&st['kinematic_valid']&(st['reliability']>=cfg['min_neighbor_reliability'])&(st['reliability']>0)&(dw>np.finfo(float).tiny)
 close([x['distance_m'] for x in actual],dist,'neighbor distances',atol=1e-6,rtol=1e-7)
 close([np.nan if x['heading_difference_deg'] is None else x['heading_difference_deg'] for x in actual],hd,'heading distance',atol=1e-8,rtol=1e-7)
 close([x['reliability'] for x in actual],st['reliability'],'neighbor reliability',atol=1e-8,rtol=1e-7)
 close([x['e_kin_mps'] for x in actual],st['e_kin_mps'],'neighbor physical innovation',atol=1e-8,rtol=1e-7)
 close([x['distance_weight'] for x in actual],dw,'distance Gaussian',atol=1e-10,rtol=1e-8)
 close([x['weight'] for x in actual],np.where(accepted,dw*st['reliability'],0),'joint neighbor weight',atol=1e-9,rtol=1e-7)
 assert np.array_equal([x['kinematic_valid'] for x in actual],st['kinematic_valid'])
 assert np.array_equal([x['accepted'] for x in actual],accepted) and np.array_equal([x['plain_accepted'] for x in actual],plain)
 nc=int(accepted.sum());np_=int(plain.sum());assert arrays['valid_neighbor_count'][j,k]==nc and arrays['plain_neighbor_count'][j,k]==np_
 rel=np.column_stack((st['x_m']-p[0],st['y_m']-p[1]));vv=st['velocity'];weights=dw[accepted]*st['reliability'][accepted]
 valid,field,div,curl=affine(rel[accepted],vv[accepted],weights,cfg) if nc else (False,np.zeros(2),0.,0.)
 bv,bfield,bdiv,bcurl=affine(rel[plain],vv[plain],np.ones(np_),cfg) if np_ else (False,np.zeros(2),0.,0.)
 mean=np.sum(vv[accepted]*(weights/weights.sum())[:,None],axis=0) if nc else np.zeros(2)
 pm=vv[plain].mean(axis=0) if np_ else np.zeros(2);mv=bool(nc);pmv=bool(np_);pv=np_>=cfg['min_neighbors']
 residual=v-field if valid else np.zeros(2);pr=v-pm if pv else np.zeros(2)
 if not boundary:valid=bv=pv=mv=pmv=False;field=np.zeros(2);bfield=np.zeros(2);mean=np.zeros(2);pm=np.zeros(2);residual=np.zeros(2);pr=np.zeros(2);div=curl=bdiv=bcurl=0.
 R=cfg['radius_nm']*1852;vs=cfg['velocity_scale_mps'];br=v-bfield if bv else np.zeros(2)
 C=np.r_[gated,residual/vs*gate,error/vs,w,kinvalid,valid,min(1,nc/cfg['min_neighbors']),div*R/vs,curl*R/vs]
 B=np.r_[raw,pr/vs,0.,0.,0.,pv,min(1,np_/cfg['min_neighbors']),0.,0.]
 BA=np.r_[raw,br/vs,0.,0.,0.,bv,min(1,np_/cfg['min_neighbors']),bdiv*R/vs,bcurl*R/vs]
 for name,val in [('background_mean_mps',mean),('background_field_mps',field),('residual_mps',residual),('divergence_per_s',div),('curl_per_s',curl),('context',C),('plain_context',B),('plain_affine_context',BA)]:close(arrays[name][j,k],val,name,atol=6e-5,rtol=5e-5)
 for name,val in [('context_valid',valid),('background_mean_valid',mv),('plain_context_valid',pv),('plain_background_mean_valid',pmv),('plain_affine_valid',bv)]:assert arrays[name][j,k]==val,name
 close(arrays['plain_background_mean_mps'][j,k],pm,'plain mean')
 return set(seen),{(str(x['mmsi']),float(x['time_s'])) for x in actual},{str(x['mmsi']) for x in actual if x['accepted']},{str(x['mmsi']) for x in actual if x['plain_accepted']}
