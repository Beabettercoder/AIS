from pathlib import Path
import numpy as np, json, sys, hashlib, importlib.util
BASE=Path('/mnt/sda/WangZijia/AIS/outputs'); CAND=BASE/'delayed_observation_20261030'; OLD=BASE/'relation_repair_20261030'; RAW=BASE/'balanced_global_v2/relation_dual_pressure_v2_20261027'; OUT=BASE/'identifiability_audit_20261030'
manifest=json.loads((RAW/'dataset_manifest_combined.json').read_text())
VIEWS=('base','natural','suffix','combined_l100_m600'); ident=('source','case','y','length','mmsi','region','raw7','causal7','self_w','time_s')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out={'scope':'independent train/validation delayed-reweight audit; no test labels or training; first 16 validation sources in region/source sort, 4 views x 4 cases x 5 valid time indices','splits':{},'reconstruction':{},'checks':{}}
sets={}
for split in ('train','val'):
    path=CAND/(split+'_relation_field.npz')
    if not path.exists():continue
    with np.load(path,allow_pickle=False) as aa, np.load(OLD/(split+'_relation_field.npz'),allow_pickle=False) as bb:
        for k in ident:
            if not np.array_equal(aa[k],bb[k]):raise AssertionError((split,'mismatch',k))
        for view in VIEWS:
            if not np.array_equal(aa[view+'__count'],bb[view+'__count']):raise AssertionError((split,view,'member count'))
            z=aa[view+'__field']; assert np.array_equal(z[:,0,:,7],z[:,1,:,7])
        sets[split]=set(map(str,aa['mmsi']));out['splits'][split]={'cases':len(aa['y']),'sources':len(set(map(str,aa['source']))),'mmsis':len(sets[split]),'candidate_sha256':sha(path),'original_sha256':sha(OLD/(split+'_relation_field.npz'))}
if len(sets)==2:out['checks']['target_mmsi_train_val_overlap']=len(sets['train']&sets['val']);assert not sets['train']&sets['val']
rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']=='val'],key=lambda r:(r['region'],r['window_id']))[:16]
K=1852/3600; R=8*1852.; SIG=4*1852.; ridge=.001
def wabs(t,p,s,c):
    u=np.deg2rad(c);v=np.column_stack((s*np.sin(u),s*np.cos(u)))*K;dt=np.diff(t)
    valid=np.r_[False,(dt>0)&(dt<=300)&np.isfinite(p[:-1]).all(1)&np.isfinite(p[1:]).all(1)&np.isfinite(s[:-1])&np.isfinite(s[1:])&np.isfinite(c[:-1])&np.isfinite(c[1:])]
    w=np.zeros(len(t));i=np.flatnonzero(valid);posv=(p[i]-p[i-1])/dt[i-1,None];err=np.linalg.norm(posv-(v[i]+v[i-1])/2,axis=1);w[i]=np.exp(-np.maximum(0,err-2.5)**2/8)
    factors=np.ones(len(t),float)
    for j in range(3,len(t)-1):
        before=t[j]-t[j-1];after=t[j+1]-t[j]
        if not (0<before<=300 and 0<after<=300):continue
        if not (np.isfinite(p[j-1:j+2]).all() and np.isfinite(v[j-1:j+2]).all() and np.isfinite(s[j-3:j]).all()):continue
        if np.median(s[j-3:j])>=2:continue
        e1=np.linalg.norm((p[j]-p[j-1])/before-v[j]);e2=np.linalg.norm((p[j+1]-p[j])/after-v[j]);factors[j]=.7+.3*np.exp(-.5*max(0.,2*e1-e2-.5)**2)
    return w,v,factors
def independent_fit(p,v,w):
    w=w/w.sum(); X=np.c_[np.ones(len(p)),p/R]; G=X.T@(w[:,None]*X)
    if len(p)<4 or np.linalg.cond(G)>1e8 or not np.isfinite(np.linalg.cond(G)):return None
    A=np.vstack((np.sqrt(w)[:,None]*X,np.sqrt(ridge)*np.array([[0.,1.,0.],[0.,0.,1.]])));Y=np.vstack((np.sqrt(w)[:,None]*v,np.zeros((2,2))))
    return np.linalg.lstsq(A,Y,rcond=None)[0]
count=0; maxerr=0; nofield=0; max_age=0; target_leak=0
with np.load(CAND/'val_relation_field.npz',allow_pickle=False) as z:
 a=z['base__field'];mask=np.arange(20)[None,:]<z['length'][:,None];out['field_view_shifts']={}
 for view in ('natural','suffix','combined_l100_m600'):
    b=z[view+'__field'];common=mask&(a[:,0,:,7]>0)&(b[:,0,:,7]>0);d0=np.linalg.norm(b[:,0,:,:2]-a[:,0,:,:2],axis=-1)[common];d1=np.linalg.norm(b[:,1,:,:2]-a[:,1,:,:2],axis=-1)[common];affected=(d0>1e-6)|(d1>1e-6)
    out['field_view_shifts'][view]={'sample_steps':int(common.sum()),'f0_mean_mps':float(d0.mean()),'f1_mean_mps':float(d1.mean()),'f0_changed_fraction':float((d0>1e-6).mean()),'f1_changed_fraction':float((d1>1e-6).mean()),'affected_steps':int(affected.sum()),'fraction_f1_smaller_among_affected':float(np.mean(d1[affected]<d0[affected]-1e-6))}
 for ri,row in enumerate(rows):
    data={}
    for view in ('base','natural','suffix'):
        file=RAW/row['files'][view]['path']; assert sha(file)==row['files'][view]['sha256']
        with np.load(file,allow_pickle=False) as d:data[view]={k:d[k].copy() for k in ('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets','cohort_mmsi','y')}
    for k in data['base']:
        if k!='neighbor_sog_kn':assert np.array_equal(data['base'][k],data['suffix'][k])
    combined={k:v.copy() for k,v in data['base'].items()};combined['neighbor_sog_kn']=data['suffix']['neighbor_sog_kn'].copy()
    for mem in row['stress']['members']:
        lo=int(combined['member_offsets'][mem['member_index']]);ix=lo+int(mem['indices']['suffix']['report_index']); digest=hashlib.sha256(('framework-stress-position-v1:'+row['window_id']+':'+mem['mmsi']).encode()).digest(); theta=2*np.pi*int.from_bytes(digest[:8],'big')/2**64
        combined['neighbor_pos'][:,ix:ix+2]+=600*np.array([np.cos(theta),np.sin(theta)])
    data['combined_l100_m600']=combined
    for c in range(4):
      sample=4*ri+c;assert str(z['source'][sample])==row['window_id']; assert z['y'][sample]==data['base']['y'][c]
      for view in VIEWS:
        d=data[view];t=d['target_t'][c];p=d['target_pos'][c];s=d['target_sog_kn'][c];ang=np.deg2rad(d['target_cog_deg'][c]);tv=np.column_stack((s*np.sin(ang),s*np.cos(ang)))*K
        for j in sorted(set([0,len(t)//4,len(t)//2,3*len(t)//4,len(t)-1])):
          records=[]
          for member,lo,hi in zip(d['cohort_mmsi'],d['member_offsets'][:-1],d['member_offsets'][1:]):
            target_leak+=str(member)==str(row['mmsi']);lo=int(lo);hi=int(hi);nt=d['neighbor_t'][c,lo:hi];q=int(np.searchsorted(nt,t[j],side='right')-1)
            if q<0 or not (0<=t[j]-nt[q]<=300):continue
            max_age=max(max_age,float(t[j]-nt[q]));pos=d['neighbor_pos'][c,lo:hi];sp=d['neighbor_sog_kn'][c,lo:hi];co=d['neighbor_cog_deg'][c,lo:hi];w,v,factors=wabs(nt,pos,sp,co)
            phase=t[0] if j<len(t)//2 else t[len(t)//2]; selected=np.flatnonzero((nt>=phase)&(nt<=t[j])); selected=selected if len(selected) else np.array([q]); ww=w[selected].copy();confirmed=selected<q; assert all(nt[u+1]<=t[j] for u in selected[confirmed]);ww[confirmed]*=factors[selected[confirmed]]; v0=v[selected].mean(0); v1=np.average(v[selected],axis=0,weights=ww) if ww.sum()>1e-12 else v[q]; records.append((pos[q]-p[j],v0,v1,ww.mean()))
          saved=z[view+'__field'][sample,:,j]; mode=int(saved[0,7]);assert mode==int(saved[1,7])
          assert len(records)==int(z[view+'__count'][sample,j])
          if not records:
            assert mode==0;nofield+=1;continue
          p2=np.array([r[0] for r in records]);v0=np.array([r[1] for r in records]);v1=np.array([r[2] for r in records]);w=np.array([r[3] for r in records]);dist=np.exp(-.5*np.sum(p2**2,axis=1)/SIG**2);ws=(dist,dist*w)
          if min(x.sum() for x in ws)<=1e-12:assert mode==0;nofield+=1;continue
          fit=[independent_fit(p2,v0,ws[0]),independent_fit(p2,v1,ws[1])];aff=all(x is not None for x in fit);assert mode==(2 if aff else 1)
          rho=min(1,len(records)/4)*w.mean()
          for i in range(2):
            if aff:
              coef=fit[i];bg=coef[0];grad=coef[1:].T/R
            else:bg=np.average((v0,v1)[i],axis=0,weights=ws[i]);grad=np.zeros((2,2))
            ref=np.array([*bg,*(tv[j]-bg),grad[0,0]+grad[1,1],grad[1,0]-grad[0,1],rho,float(mode)])
            err=float(np.max(np.abs(saved[i]-ref)));maxerr=max(maxerr,err);assert err<1e-4,(view,sample,j,i,err)
          count+=1
assert target_leak==0
out['reconstruction']={'valid_points':count,'unavailable_points':nofield,'max_absolute_error':maxerr,'max_report_age_s':max_age,'target_cohort_overlap_count':target_leak,'solving':'independent augmented weighted least squares; common affine decision or zero-order'}
# Feature-label independence / normalization: import training functions, reverse labels, and compare all four views/arms.
s=importlib.util.spec_from_file_location('latest_training',CAND/'fixed_normalizer/train_fixed.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=m.read('val');rev={k:(v.copy() if hasattr(v,'copy') else v) for k,v in d.items()};rev['y']=1-rev['y']
for v in VIEWS:
 for arm in ('C0','F0','F1'):
    x,valid=m.features(d,v,arm);xx,_=m.features(rev,v,arm);assert np.array_equal(x,xx)
    if arm=='C0':assert (x[:,:,8:]==0).all()
out['checks']['features_unchanged_after_label_flip']=True;out['checks']['c0_context_all_zero']=True
if (CAND/'train_relation_field.npz').exists():
 tr=m.read('train');mu,sd=m.fit(tr);rev2={k:(v.copy() if hasattr(v,'copy') else v) for k,v in tr.items()};rev2['y']=1-rev2['y'];mu2,sd2=m.fit(rev2);assert np.array_equal(mu,mu2) and np.array_equal(sd,sd2);out['checks']['train_only_normalizer_label_invariant']=True
(OUT/'delayed_field_independent_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print(json.dumps(out,indent=2))

