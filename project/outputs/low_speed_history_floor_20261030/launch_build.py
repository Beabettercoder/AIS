from pathlib import Path
import subprocess,json
O=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_history_floor_20261030'); P='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'; rows=[]
for split in ('train','val'):
 f=(O/('build_'+split+'.log')).open('wb');p=subprocess.Popen([P,str(O/'builder.py'),'--split',split],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);rows.append({'split':split,'pid':p.pid})
(O/'build_launch.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
