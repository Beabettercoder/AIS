from pathlib import Path
import json
root=Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030/runs')
for name in ('tcn','gru','transformer','itransformer'):
 print(name)
 for arm in ('C0','F0','F1'):
  d=json.loads((root/name/arm/'result.json').read_text())
  print(arm,'loss',round(d['loss_history'][-1],4),'AP',*[round(d['views'][v]['pooled_AP'],4) for v in ('base','natural','suffix','combined_l100_m600')])
