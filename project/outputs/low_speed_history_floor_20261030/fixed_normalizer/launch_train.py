from pathlib import Path
import hashlib,json,subprocess
out=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_history_floor_20261030/fixed_normalizer'); base=out.parent
protocol=json.loads((out/'frozen_protocol.json').read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for key,p in [('builder_sha256',base/'builder.py'),('train_sha256',out/'train_fixed.py')]:assert sha(p)==protocol[key]
for split in ('train','val'):
 p=base/(split+'_relation_field.npz');meta=json.loads(p.with_suffix('.json').read_text());assert meta['builder_sha256']==protocol['builder_sha256'] and meta['file_sha256']==sha(p)
assert not (out/'runs').exists(); protocol['train_data_sha256']=sha(base/'train_relation_field.npz');protocol['val_data_sha256']=sha(base/'val_relation_field.npz'); (out/'input_hashes.json').write_text(json.dumps({'train':protocol['train_data_sha256'],'val':protocol['val_data_sha256']},indent=2))
workers={1:[('tcn',0),('gru',0)],2:[('tcn',1),('transformer',0)],3:[('tcn',2)],5:[('itransformer',0)],6:[('itransformer',1)],7:[('itransformer',2)]}
records=[]
for gpu,jobs in workers.items():
 p=subprocess.Popen(['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python',str(out/'worker.py'),str(gpu),json.dumps(jobs)],cwd=out,stdout=(out/f'gpu{gpu}.log').open('wb'),stderr=subprocess.STDOUT,start_new_session=True); records.append({'gpu':gpu,'jobs':jobs,'pid':p.pid})
(out/'launch.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
