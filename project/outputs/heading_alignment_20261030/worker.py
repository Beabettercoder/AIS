import subprocess,sys
from pathlib import Path
name,gpu=sys.argv[1:];out=Path(__file__).parent
def run(variant,arms):
 subprocess.run([sys.executable,str(out/'train_heading.py'),'--name',name,'--gpu',gpu,'--variant',variant,'--arms',arms],check=True)
run('aligned_embed','C0,F0,F1')
if name=='itransformer':
 for v in ['aligned_noembed','old_embed','old_noembed']:run(v,'F1')
print('WORKER COMPLETE',name,flush=True)
