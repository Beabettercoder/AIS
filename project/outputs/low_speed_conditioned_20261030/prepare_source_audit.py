"""Derive independent field verifier from prior verified verifier, retaining output isolation."""
from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs')
src=(P/'gate_step_diagnostic_20261030/audit_safe_source.py').read_text()
src=src.replace("CAND=BASE/'neighbor_gate_safe_20261030'", "CAND=BASE/'low_speed_history_floor_20261030'")
src=src.replace("OUT=BASE/'gate_step_diagnostic_20261030'", "OUT=BASE/'low_speed_conditioned_20261030'")
src=src.replace("independent development-set causal min-residual gate audit", "independent development-set low-speed historical bounded gate audit")
src=src.replace("(OUT/'safe_gate_independent_audit.json')", "(OUT/'history_gate_source_independent_audit.json')")
src=src.replace("CAND/'train.py'", "CAND/'fixed_normalizer/train_fixed.py'")
old="w=np.zeros(len(t));i=np.flatnonzero(valid);posv=(p[i]-p[i-1])/dt[i-1,None];err=np.minimum(np.linalg.norm(posv-(v[i]+v[i-1])/2,axis=1),np.linalg.norm(posv-v[i],axis=1));w[i]=np.exp(-np.maximum(0,err-2.5)**2/8)"
new="w=np.zeros(len(t));i=np.flatnonzero(valid);posv=(p[i]-p[i-1])/dt[i-1,None];err=np.linalg.norm(posv-(v[i]+v[i-1])/2,axis=1);w[i]=np.exp(-np.maximum(0,err-2.5)**2/8);j=i[i>=3];hist=np.column_stack((s[j-3],s[j-2],s[j-1]));good=np.isfinite(hist).all(1);j=j[good];hist=hist[good];med=np.median(hist,axis=1);posj=(p[j]-p[j-1])/dt[j-1,None];ecur=np.linalg.norm(posj-v[j],axis=1);ratio=np.abs(s[j]-med)/(med+.5);gate=(med<2)&(ecur>=.5);w[j]*=np.where(gate,.7+.3*np.exp(-.5*np.maximum(0,ratio-1)**2),1.)"
assert old in src
src=src.replace(old,new)
(P/'low_speed_conditioned_20261030/independent_field_audit.py').write_text(src)
