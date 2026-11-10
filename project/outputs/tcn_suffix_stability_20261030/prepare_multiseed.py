from pathlib import Path

base = Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030/train_heading.py')
out = Path('/mnt/sda/WangZijia/AIS/outputs/tcn_suffix_stability_20261030/train_multiseed.py')
s = base.read_text(encoding='utf8')
old = "OUT=Path('/mnt/sda/WangZijia/AIS/outputs/heading_alignment_20261030')"
new = "OUT=Path('/mnt/sda/WangZijia/AIS/outputs/tcn_suffix_stability_20261030')"
assert old in s
s = s.replace(old,new)
old = 'def seed():random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)'
new = 'SEED=0\ndef seed():random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)'
assert old in s
s = s.replace(old,new)
old = 'def run(name,gpu,variant,arms):\n    global VARIANT; VARIANT=variant'
new = 'def run(name,gpu,variant,arms,runseed):\n    global VARIANT, SEED; VARIANT=variant; SEED=runseed'
assert old in s
s = s.replace(old,new)
old = "dest=OUT/'runs'/variant/name;"
new = "dest=OUT/'runs'/variant/('seed'+str(SEED))/name;"
assert old in s
s=s.replace(old,new)
s=s.replace("'seed':0", "'seed':SEED")
old = "p.add_argument('--arms',default='C0,F0,F1');a=p.parse_args();run(a.name,a.gpu,a.variant,a.arms)"
new = "p.add_argument('--arms',default='C0,F0,F1');p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.name,a.gpu,a.variant,a.arms,a.seed)"
assert old in s
s=s.replace(old,new)
out.write_text(s,encoding='utf8')
