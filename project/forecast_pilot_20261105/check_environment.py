"""Read-only dependency check; does not install software or start experiments."""
import importlib, shutil, sys
print('Python:',sys.executable,sys.version.split()[0],flush=True)
missing=[]
for name in ['numpy','pandas','pyarrow','torch','tqdm']:
    try:
        module=importlib.import_module(name)
        print(f'{name}: {getattr(module,"__version__","available")}',flush=True)
    except ImportError:
        missing.append(name);print(f'{name}: MISSING',flush=True)
if missing:
    raise SystemExit('Missing dependencies: '+', '.join(missing))
import torch
print('CUDA available:',torch.cuda.is_available(),flush=True)
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}',flush=True)
print('Environment check passed. CPU smoke is supported; use an idle GPU for full training.',flush=True)
