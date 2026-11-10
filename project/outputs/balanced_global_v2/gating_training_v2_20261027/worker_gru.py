import subprocess,sys,json,time
commands=[['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/historical/train_historical.py', '--data', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/data', '--out', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/runs', '--protocol', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/protocol.json', '--backbone', 'gru', '--policy', 'clean', '--device', 'cuda:0'], ['/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/historical/train_historical.py', '--data', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/data', '--out', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/runs', '--protocol', '/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/gating_training_v2_20261027/protocol.json', '--backbone', 'gru', '--policy', 'all', '--device', 'cuda:0']]
for cmd in commands:
 print("START",json.dumps(cmd),flush=True)
 r=subprocess.run(cmd)
 print("END",r.returncode,flush=True)
 if r.returncode:sys.exit(r.returncode)
