from pathlib import Path
import concurrent.futures,subprocess,os,json
R=Path(__file__).resolve().parent;PY='/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python'
jobs=[(0,['neuralcde']),(1,['mamba']),(2,['patchtst']),(3,['tcn','gru']),(5,['transformer']),(6,['itransformer']),(7,['graph_temporal'])]
def run(gpu,archs):
    rows=[]
    for arch in archs:
        if (R/'STOP').exists():break
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (R/'logs'/f'field_stress_{arch}.log').open('w') as f:r=subprocess.run([PY,str(R/'evaluate_field_stress.py'),'--backbone',arch,'--device','cuda:0'],stdout=f,stderr=subprocess.STDOUT,env=env)
        row={'backbone':arch,'gpu':gpu,'returncode':r.returncode};print(json.dumps(row),flush=True);rows.append(row)
    return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:rows=[r for f in [pool.submit(run,*j) for j in jobs] for r in f.result()]
(R/'field_stress_launcher_results.json').write_text(json.dumps(rows,indent=2))
