"""Synthetic-only worker -> freeze -> selected test -> report integration check."""
import tempfile
from pathlib import Path
import numpy as np
import engine as E
from engine_smoke import dataset
from pipeline import freeze
from report import summarize

def main():
    with tempfile.TemporaryDirectory(prefix='v2_integration_') as tmp:
        root=Path(tmp);data=root/'data';data.mkdir();exp=root/'experiment';exp.mkdir()
        for split,n,mid in [('train',8,100),('val',4,200),('test',4,300)]:
            np.savez_compressed(data/f'{split}.npz',**dataset(n,24,4,10,mid))
        z=E.load_data(data/'train.npz',True);norm=root/'norm.json';E.save(norm,E.legacy.fit_normalizer(z))
        settings=dict(name='synthetic_qa_not_experiment',models=['tcn'],seeds=[0],hidden=128,batch_size=4,epochs=2,itransformer_epochs=2,
            history_steps=24,future_steps=4,policy='cosine_best',train_per_region=8,data_dir=str(data),norm_sha256=E.sha(norm),lr=.002)
        E.save(exp/'experiment.json',settings)
        for arm in ['P0','P1','P2','P3']:
            E.train(data,exp/f'runs/tcn/seed_0/{arm}','tcn',0,arm,128,4,2,.002,'cosine_best',norm,'cpu')
        f=freeze(exp,settings,norm)
        for row in f['models']:
            E.evaluate(data,exp/f"test_runs/tcn/seed_0/{row['arm']}",row['run_dir'],device='cpu',freeze_path=exp/'freeze.json')
        for split in ['val','test']:summarize(exp,root/f'report_{split}',split,data_dir=data,draws=32)
        assert len(f['models'])==4
    print('PASS synthetic integration: dynamic H24/K4 hidden128; all four arms; shared initialization; best checkpoint freeze and test; val/test summaries. No real test data or full training was used.')

if __name__=='__main__':main()
