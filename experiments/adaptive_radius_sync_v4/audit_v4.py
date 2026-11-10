from pathlib import Path
import json,numpy as np
from build_relation_v2 import REGIONS,OLD,FROZEN,sha
import train_relation_v2 as rel
root=Path('/home/ps/AIS_adaptive_sync_20261110')
sets={};hashes={};summary={}
for split in ('train','val','test'):
 d,h=rel.data(root,split);sets[split]=d;hashes.update(h)
 rp=FROZEN/'test_relation_field.npz' if split=='test' else OLD/f'{split}_relation_field.npz'
 with np.load(rp,allow_pickle=False) as ref:
  for region in REGIONS:
   ix=d['region'].astype(str)==region; ir=ref['region'].astype(str)==region
   for k in ('source','case','y','length','mmsi','region','raw7','causal7','self_w','time_s'):
    if not np.array_equal(d[k][ix],ref[k][ir]):raise ValueError('reference mismatch '+split+'/'+region+'/'+k)
 if len(d['y'])%4 or not np.all(d['case'].reshape(-1,4)==np.arange(4)):raise ValueError('not quartets '+split)
 if not np.all(d['sparse_source'].reshape(-1,4)==d['sparse_source'].reshape(-1,4)[:,:1]):raise ValueError('sparse split inconsistent')
 mask=d['mask'];count=d['base__loww_count'];ambient=d['base__ambient_count']
 for region in REGIONS:
  path=root/'relation'/f'{split}_{region}.json';m=json.loads(path.read_text())
  pp=path.with_suffix('.proof.json'); proof=json.loads(pp.read_text())
  if sha(pp)!=m['proof_sha256'] or len(proof)!=m['source_count']:raise ValueError('proof mismatch '+split+'/'+region)
  for entry in proof:
   if entry['innovation_max_error_mps']>1e-7 or entry['selected']!=len(entry['records']) or entry['selected']>entry['eligible']:raise ValueError('invalid intervention proof')
   for x in entry['records']:
    if not 3<=x['factor']<=5:raise ValueError('invalid stress')
  for key,file in [('protocol_sha256','protocol.json'),('operator_sha256','radius_core.py'),('builder_sha256','build_relation_sync.py')]:
   if m[key]!=sha(root/file):raise ValueError('construction source changed '+key+'/'+split+'/'+region)
 for view in ('base','natural','suffix','combined_l100_m600'):
  for arm in ('tight_plain','wide_plain','adaptive_plain_nofallback','adaptive_plain_fallback','adaptive_qualified_fallback','adaptive_credible_fallback'):
   x=d[view+'__'+arm+'__field'];n=d[view+'__'+arm+'__count'];mode=x[:,:,7]
   if not np.isfinite(x).all():raise ValueError('nonfinite field')
   if np.any((mode==2)&(n<6)):raise ValueError('six-member affine threshold violated')
   if 'plain' in arm and not np.allclose(x[:,:,6][mode>0],np.minimum(n[mode>0]/4,1),atol=1e-7):raise ValueError('plain support depends on credibility')
 summary[split]={'sources':len(d['y'])//4,'samples':len(d['y']),'target_ships':len(np.unique(d['mmsi'])),'valid_steps':int(mask.sum()),'ambient_observed_steps':int(((ambient>0)&mask).sum()),'low_credibility_candidate_steps':int(((count>0)&mask).sum()),'unknown_coverage_steps':int((~d['base__coverage_known']&mask).sum())}
for a,b in [('train','val'),('train','test'),('val','test')]:
 if set(sets[a]['mmsi'].astype(str))&set(sets[b]['mmsi'].astype(str)):raise ValueError('MMSI overlap '+a+'/'+b)
out={'status':'passed','input_sha256':hashes,'summaries':summary,'reference':'quartet identities and target observations identical to prior data; ambient requery and synchronized group slowdown','ordinary':'no neighbor reliability filter or weights; still subject to historically screened quartet source selection'}
(root/'reports/pretrain_audit.json').write_text(json.dumps(out,indent=2));print('PRETRAIN_AUDIT_PASSED',json.dumps(summary),flush=True)
