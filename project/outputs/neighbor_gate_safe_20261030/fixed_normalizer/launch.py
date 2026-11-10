import hashlib,json,subprocess
from pathlib import Path
out=Path('/mnt/sda/WangZijia/AIS/outputs/neighbor_gate_safe_20261030/fixed_normalizer'); base=out.parent
protocol=json.loads((out/'frozen_protocol.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={'train_fixed_sha256':out/'train_fixed.py','candidate_train_sha256':base/'train_relation_field.npz','candidate_val_sha256':base/'val_relation_field.npz','candidate_builder_sha256':base/'builder.py','reference_train_sha256':Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030/train_relation_field.npz')}
assert all(sha(v)==protocol['files'][k] for k,v in checks.items())
assert protocol['frozen_before_fixed_normalizer_control_training'] and not (out/'runs').exists()
workers={1:[('tcn',0),('gru',0)],2:[('tcn',1),('transformer',0)],3:[('tcn',2)],5:[('itransformer',0)],6:[('itransformer',1)],7:[('itransformer',2)]}
records=[]
for gpu,jobs in workers.items():
 cmds=[f'/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python train_fixed.py --name {name} --gpu {gpu} --seed {seed} --variant aligned_embed --arms F1' for name,seed in jobs]
 f=(out/f'gpu{gpu}.log').open('ab'); proc=subprocess.Popen(['/bin/bash','-c',' && '.join(cmds)],cwd=out,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);records.append({'gpu':gpu,'jobs':jobs,'pid':proc.pid})
(out/'launch.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(records))
