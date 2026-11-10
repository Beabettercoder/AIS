"""Build candidate relation test only after stage-1 gate, never overwrite."""
import subprocess,sys,json
from pathlib import Path
sys.path.insert(0,'/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030')
from guarded_scale_train import gate
gate()
from candidate_integrity import check_candidate,check_gate_candidate
freeze=check_candidate(('train','val'))
approval=json.loads(Path('/mnt/sda/WangZijia/AIS/outputs/eight_backbone_preparation_20261030/AUTHORIZED_STAGE2.json').read_text(encoding='utf8'))
check_gate_candidate(approval,freeze)
source=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030')
target=source/'test_relation_field.npz'
for artifact in (target,target.with_suffix('.json')):
 if artifact.exists(): raise FileExistsError(f'Refusing overwrite: {artifact}')
subprocess.run([sys.executable,str(source/'builder.py'),'--split','test'],check=True)
