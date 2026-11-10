from pathlib import Path
import json
root=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/runs')
for variant in sorted(root.iterdir()):
 p=variant/'itransformer/F1/result.json'
 if p.exists():
  d=json.loads(p.read_text());h=d['loss_history']; print(variant.name,'base AP',d['views']['base']['pooled_AP'],'last train bce',h[-1],'epoch20',h[19])
