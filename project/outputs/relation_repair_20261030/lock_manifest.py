from pathlib import Path
import json,hashlib
out=Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030')
keys=['builder.py','train.py','summary.py','train_relation_field.npz','val_relation_field.npz','train_relation_field.json','val_relation_field.json','metadata_correction.md','support_audit.json','development_summary.json','关系任务修复单种子报告.md']
files=[out/k for k in keys]+list((out/'runs').rglob('*.json'))+list((out/'runs').rglob('*.npz'))+list((out/'runs').rglob('*.pt'))
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
value={'scope':'frozen after completion; history and view-specific model results retained; no unseen test','count':len(files),'sha256':{str(x.relative_to(out)):h(x) for x in sorted(files)}}
(out/'full_manifest.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print('locked',len(files),'files')
