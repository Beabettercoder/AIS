"""Construct frozen relation test views only after the user-defined pilot gate."""
import subprocess
import sys
from pathlib import Path
from guarded_scale_train import gate

gate()
source = Path('/mnt/sda/WangZijia/AIS/outputs/relation_repair_20261030')
target = source / 'test_relation_field.npz'
if target.exists():
    raise FileExistsError(f'Refusing to overwrite {target}; verify its provenance before reuse.')
subprocess.run([sys.executable, str(source/'builder.py'), '--split', 'test'], check=True)
