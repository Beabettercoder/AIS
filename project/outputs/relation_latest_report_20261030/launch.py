import hashlib,json,subprocess,sys
from pathlib import Path
out=Path('/mnt/sda/WangZijia/AIS/outputs/relation_latest_report_20261030')
protocol=json.loads((out/'frozen_protocol.json').read_text())
for key,file in [('builder','builder.py'),('train','train.py')]:
    assert hashlib.sha256((out/file).read_bytes()).hexdigest()==protocol['sha256'][key]
assert (out/'frozen_addendum.json').exists()
assert all((out/(x+'_relation_field.npz')).exists() for x in ('train','val'))
assert not (out/'runs').exists(), 'Refusing to overwrite existing run outputs'
workers={1:[('tcn',0),('gru',0)],2:[('tcn',1),('transformer',0)],3:[('tcn',2)],5:[('itransformer',0)],6:[('itransformer',1)],7:[('itransformer',2)]}
records=[]
for gpu,jobs in workers.items():
    commands=[]
    for name,seed in jobs:
        commands.append(f'/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python train.py --name {name} --gpu {gpu} --seed {seed} --variant aligned_embed --arms C0,F0,F1')
    log=(out/f'gpu{gpu}.log').open('ab')
    p=subprocess.Popen(['/bin/bash','-c',' && '.join(commands)],cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    records.append({'gpu':gpu,'jobs':jobs,'pid':p.pid})
(out/'launch.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records))
