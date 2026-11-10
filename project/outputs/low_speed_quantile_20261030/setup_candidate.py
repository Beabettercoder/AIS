from pathlib import Path
import hashlib,json
base=Path('/mnt/sda/WangZijia/AIS/outputs'); out=base/'low_speed_history_floor_20261030'; out.mkdir(exist_ok=True)
old=base/'neighbor_gate_safe_20261030'; s=(old/'builder.py').read_text()
s=s.replace("OUT=BASE/'neighbor_gate_safe_20261030'", "OUT=BASE/'low_speed_history_floor_20261030'")
start=s.index('def safe_neighbor_weights(');end=s.index('\ndef calc(',start)
func='''def safe_neighbor_weights(t,p,speed,course):
    # Causal soft uncertainty score; same original w_abs, only low-speed history.
    w,valid=core.absolute_weights(t,p,speed,course)
    ang=np.deg2rad(course); vv=speed[:,None]*0.514444444444*np.column_stack((np.sin(ang),np.cos(ang)))
    ii=np.flatnonzero(valid & (np.arange(len(t))>=3))
    if len(ii)==0:return w,valid
    hist=np.column_stack((speed[ii-3],speed[ii-2],speed[ii-1])); ok=np.isfinite(hist).all(1);ii=ii[ok];hist=hist[ok]
    if len(ii)==0:return w,valid
    h=np.median(hist,axis=1); vp=(p[ii]-p[ii-1])/(t[ii]-t[ii-1])[:,None];cur=np.linalg.norm(vp-vv[ii],axis=1)
    ratio=np.abs(speed[ii]-h)/(h+0.5); active=(h<2.0)&(cur>=0.5)
    factor=np.where(active,0.7+0.3*np.exp(-0.5*np.maximum(0,ratio-1)**2),1.0)
    w[ii]*=factor
    return w,valid
'''
s=s[:start]+func+s[end:]
s=s.replace('paired cumulative aggregation and causal minimum-of-residuals neighbor gate','exploratory negative control: paired cumulative aggregation and bounded low-speed historical uncertainty factor')
s=s.replace('min(1,n/4)*mean safe neighbor weights','min(1,n/4)*mean original times bounded historical uncertainty weights')
assert "OUT=BASE/'low_speed_history_floor_20261030'" in s and 'factor=np.where' in s
(out/'builder.py').write_text(s,encoding='utf8')
t=(old/'fixed_normalizer/train_fixed.py').read_text()
t=t.replace("OUT=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030/fixed_normalizer')", "OUT=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_history_floor_20261030/fixed_normalizer')")
t=t.replace("OLD=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030')", "OLD=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_history_floor_20261030')")
t=t.replace('weighted distance*safe report reliability','weighted distance*bounded history uncertainty report reliability')
assert "outputs/low_speed_history_floor_20261030/fixed_normalizer" in t
f=out/'fixed_normalizer';f.mkdir(exist_ok=True);(f/'train_fixed.py').write_text(t,encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proto={'scope':'exploratory negative control selected after training-only diagnostic; not assumed physically reliable or production-ready; no test use', 'method':'retain w_abs, multiply by 0.7+0.3*exp(-0.5*max(0,ratio-1)^2) when median prior3 SOG<2kn and norm(v_pos-v_current)>=0.5m/s; ratio=abs(SOG_t-prior3median)/(prior3median+0.5kn); earlier than 3 reports or invalid data use original w','rationale':'cap extra attenuation at 30% while preserving old gate; prior training-only unmodified high-motion proxy attenuation is notably larger than controlled slow speed alteration, risk explicitly retained','input':'same original phase-cumulative source members and synthetic views; no corruption label or future reports in weights','risk_pretraining':'400 training sources: changed current base-speed<2kn mean extra attenuation 5.6%, unmodified historylow and speed change>=1kn mean 19.7%; source/neighbor repeated, unverified natural truth. Do not launch full experiment solely on positive CI.','normalizer':'original cumulative train base pooled F0/F1, fixed; old F0 and C0 reused after exact identity audit','models':{'tcn':{'seeds':[0,1,2],'epochs':25},'itransformer':{'seeds':[0,1,2],'epochs':75},'gru':{'seeds':[0],'epochs':25},'transformer':{'seeds':[0],'epochs':25}},'model_arms':'candidate F1 only; existing paired baseline F0/F1/C0 retained','bootstrap':'target MMSI clusters, 2000 paired draws across fixed models, per seed, no multiplicity correction, does not include seed or shared-background uncertainty','formal_gate':'four model pass existing three pressure criteria AND no serious low speed real maneuver proxy attenuation; training diagnostic currently identifies serious risk so no automated formal launch solely from result','builder_sha256':sha(out/'builder.py'),'train_sha256':sha(f/'train_fixed.py'),'diagnostic_sha256':sha(base/'low_speed_quantile_20261030/history_innovation_diagnostic.py')}
(f/'frozen_protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf8');print(json.dumps(proto,indent=2))
