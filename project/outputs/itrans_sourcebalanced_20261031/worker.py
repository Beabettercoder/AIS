import subprocess,sys
from pathlib import Path
r=Path('/mnt/sda/WangZijia/AIS/outputs/itrans_sourcebalanced_20261031')
gpu=sys.argv[1]
for seed in sys.argv[2:]:
 subprocess.run([sys.executable,str(r/'train.py'),'--name','itransformer','--seed',seed,'--gpu',gpu],check=True)
