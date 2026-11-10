from pathlib import Path
import subprocess,sys,json
out=Path('/mnt/sda/WangZijia/AIS/outputs/delayed_observation_20261030/fixed_normalizer'); gpu=sys.argv[1]
for name,seed in json.loads(sys.argv[2]):
 cmd=['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python',str(out/'train_fixed.py'),'--name',name,'--gpu',gpu,'--seed',str(seed),'--variant','aligned_embed','--arms','F1']
 print('RUN',name,seed,flush=True);subprocess.run(cmd,cwd=out,check=True)
print('ALL_DONE',flush=True)

