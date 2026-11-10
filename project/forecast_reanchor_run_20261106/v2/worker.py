import argparse,json
from pathlib import Path
from engine import train

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--device',required=True)
    a=p.parse_args();s=json.loads((a.experiment/'experiment.json').read_text())
    for seed in s['seeds']:
        for arm in ['P0','P1','P2','P3']:
            train(Path(s['data_dir']),a.experiment/f'runs/{a.model}/seed_{seed}/{arm}',a.model,seed,arm,
                  hidden=s['hidden'],batch=s['batch_size'],epochs=s['itransformer_epochs'] if a.model=='itransformer' else s['epochs'],
                  lr=s['learning_rate'],policy=s['policy'],norm_path=Path(s['norm_path']),device=a.device)
