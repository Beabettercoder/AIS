"""Combine previously frozen baseline arms with new factorial validation runs."""
from pathlib import Path
import json,statistics
ROOT=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
NEW=ROOT/'factorial_ablation_v1_20261027';OLD=ROOT/'causal_gate_main_v1_20261027'
prior=['raw','old','comp','causal','causal_common','causal_common_bundle']
new=json.loads((NEW/'protocol.json').read_text())['arms'];arms=prior+new
def load(arch,seed,arm):
 path=(OLD if arm in prior else NEW)/'runs'/arch/f'seed_{seed}'/arm/'result.json'
 z=json.loads(path.read_text());assert z['status']=='completed_validation_only';return z
summary={'scope':'fixed epoch25 validation, reused old and newly trained matched same source and seed; development data previously explored', 'metrics':{},'paired':{},'interactions':{}}
for arch in ('tcn','gru'):
 summary['metrics'][arch]={}
 for arm in arms:
  rr=[load(arch,s,arm) for s in (0,1,2)];a=[r['final_validation']['macro']['AP'] for r in rr]
  pooled=[r['final_validation']['pooled']['AP'] for r in rr];auc=[r['final_validation']['pooled']['AUROC'] for r in rr]
  summary['metrics'][arch][arm]={'macro_AP':a,'mean_macro_AP':statistics.mean(a),'sd_macro_AP':statistics.stdev(a),'mean_pooled_AP':statistics.mean(pooled),'mean_pooled_AUROC':statistics.mean(auc)}
 comparisons=[('causal','raw'),('raw_common','raw'),('causal_common','causal'),('raw_bundle','raw'),('causal_bundle','causal'),('raw_common_bundle','raw_common'),('causal_common_bundle','causal_common'),('raw_common_bundle','raw_bundle'),('causal_common_bundle','causal_bundle'),('causal_bundle','causal_common_bundle'),('old_bundle','old'),('comp_bundle','comp'),
   ('raw_bundle_self','raw_bundle_bgz'),('causal_bundle_self','causal_bundle_bgz'),('causal_common_bundle_self','causal_common_bundle_bgz'),
   ('raw_bundle_bgz','raw_bundle'),('causal_bundle_bgz','causal_bundle'),('causal_common_bundle_bgz','causal_common_bundle')]
 summary['paired'][arch]={}
 for a,b in comparisons:
  diff=[x-y for x,y in zip(summary['metrics'][arch][a]['macro_AP'],summary['metrics'][arch][b]['macro_AP'])]
  summary['paired'][arch][a+' minus '+b]={'per_seed_pp':[100*x for x in diff],'mean_pp':100*statistics.mean(diff)}
 interactions=[('gate_by_bundle_support_off','causal_bundle','causal','raw_bundle','raw'),('gate_by_bundle_support_on','causal_common_bundle','causal_common','raw_common_bundle','raw_common'),
               ('support_by_bundle_raw','raw_common_bundle','raw_common','raw_bundle','raw'),('support_by_bundle_causal','causal_common_bundle','causal_common','causal_bundle','causal')]
 summary['interactions'][arch]={}
 for name,a,b,c,d in interactions:
  vals=[100*(summary['metrics'][arch][a]['macro_AP'][s]-summary['metrics'][arch][b]['macro_AP'][s]-summary['metrics'][arch][c]['macro_AP'][s]+summary['metrics'][arch][d]['macro_AP'][s]) for s in (0,1,2)]
  summary['interactions'][arch][name]={'mean_pp':statistics.mean(vals),'seed_pp':vals,'formula':f'({a} − {b}) − ({c} − {d})'}
dest=NEW/'validation_summary_v2.json';assert not dest.exists();dest.write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');print('VALIDATION_SUMMARY_V2_DONE',flush=True)
