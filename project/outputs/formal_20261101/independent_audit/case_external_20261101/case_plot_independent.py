from pathlib import Path
import csv,json,hashlib
import numpy as np
D=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/cases/v6')
O=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/independent_audit/case_external_20261101')
S=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single')
names=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];fr=json.loads((S/'frozen_checkpoint_thresholds.json').read_text());results={}
def sh(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for group in ('nonoverlap','rolling'):
 scored=D/('case_scores.npz' if group=='nonoverlap' else 'rolling_scores.npz'); meta=D/('realcase_input_audit.json' if group=='nonoverlap' else 'rolling_input_audit.json'); csvfile=D/('case_scores_all_backbones_seeds_TNR_gapped.csv' if group=='nonoverlap' else 'rolling_scores_all_backbones_seeds_TNR_gapped.csv')
 z=np.load(scored,allow_pickle=False);a=json.loads(meta.read_text())['cases'];rows=list(csv.DictReader(csvfile.open(newline='',encoding='utf8'))); expected=sum(len(a[k]) for k in ('MATSON','SARA'))*24;assert len(rows)==expected
 seen=set()
 for r in rows:
  case=r['case'];n=r['backbone'];seed=int(r['seed']);j=int(r['window_index']);i=names.index(n);source=a[case][j];key=(case,j,n,seed);assert key not in seen;seen.add(key)
  assert r['start_utc']==source['first_utc'] and r['end_utc']==source['end_utc'] and int(r['message_count'])==source['count']
  assert abs(float(r['median_sog_kn'])-source['median_sog_kn'])<1e-10 and int(r['valid_field_steps'])==source['context_valid_steps']
  assert abs(float(r['score'])-z[case][i,seed,j])<1e-12
  item=fr['arms'][f'{n}/{seed}/f1_bundle']['thresholds']
  for short,long in (('1pct','0.01'),('5pct','0.05')):
   threshold=item[long]['threshold']; assert abs(float(r['val_threshold_'+short])-threshold)<1e-12
   assert (r['exceeds_val_'+short]=='True')==(float(r['score'])>threshold)
 results[group]={'csv_rows':len(rows),'unique_model_windows':len(seen),'all_match_npz_source_and_threshold':True,'score_sha256':sh(scored),'csv_sha256':sh(csvfile)}
 # Independently verify referenced plotting logic selects the intended score JSON,
 # all eight names, min/max shading, truthful timestamp text, and gaps.
 script=D/('render_seedbands_gapped.py' if group=='nonoverlap' else 'render_rolling_gapped.py');text=script.read_text()
 assert ('case_scores.json' if group=='nonoverlap' else 'rolling_scores.json') in text
 for s in ("scores.min(1)","scores.max(1)","scores.mean(1)",'SARA: exact contact time unknown','not calibrated','not confidence intervals','font.family'):
  assert s in text,(group,s)
 assert ('2100' if group=='nonoverlap' else '900') in text
 for ext in ('png','pdf','svg'):
  figure=D/(('Fig8_single_vessel_event_replay_seedbands_TNR_gapped.' if group=='nonoverlap' else 'Fig8_single_vessel_event_replay_rolling_TNR_gapped.')+ext);assert figure.is_file();results[group][ext+'_sha256']=sh(figure)
out=O/'case_plot_independent.json';assert not out.exists();out.write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps(results,indent=2))
