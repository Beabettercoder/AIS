from pathlib import Path
import sys, json, importlib.util
import torch, numpy as np
R=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2')
print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'packages':{x:bool(importlib.util.find_spec(x)) for x in ['mamba_ssm','torchcde','torchdiffeq','einops','pandas','pyarrow','scipy','sklearn']}},indent=2))
for rel in ['gating_training_v2_20261027/data/houston/train.npz','robust_background_v1_20261027/houston/train.npz','labeling/labeled_v1/features_scene/houston/train.npz','relation_dual_pressure_v2_20261027/model_study/features_train.npz']:
 p=R/rel
 with np.load(p,allow_pickle=False) as z:print(rel,json.dumps({k:[list(z[k].shape),str(z[k].dtype)] for k in z.files}))
