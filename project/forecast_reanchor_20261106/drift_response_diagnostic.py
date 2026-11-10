"""Post-hoc paired drift-sensitivity diagnosis. Does not select a new candidate."""
import argparse,json,sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,default=Path('/mnt/sda/WangZijia/AIS/forecast_pilot_20261105'))
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args()
    sys.path.insert(0,str(args.pilot));from summarize import paired_bootstrap
    with np.load(args.pilot/'data/val.npz') as z:old={k:z[k] for k in z.files}
    with np.load(args.root/'data/val.npz') as z:new={k:z[k] for k in z.files}
    meta={k:old[k] for k in ['window_id','mmsi','region','future_xy','query_s','maneuver']}
    meta['mmsi']=meta['mmsi'].astype(str)
    def error(z,view):
        pred=z[view+'__anchor_gate'][:,None]+z[view+'__vel_gate'].astype(float)[:,None]*z['query_s'].astype(float)[...,None]
        e=np.linalg.norm(pred-z['future_xy'],axis=-1);return {'ADE':e.mean(1),'FDE':e[:,-1]}
    oldbase=error(old,'base');newbase=error(new,'base');olddrift=error(old,'target_drift');newdrift=error(new,'target_drift')
    a={**meta,**{k:newdrift[k]-newbase[k] for k in ['ADE','FDE']}}
    b={**meta,**{k:olddrift[k]-oldbase[k] for k in ['ADE','FDE']}}
    intervals=paired_bootstrap([a],[b],'all',2000,20261106)
    out={'purpose':'Post-hoc mechanism diagnosis only; fixed CV continuation decision is unchanged',
         'quantity':'(candidate drift error - candidate base error) - (old drift error - old base error)',
         'interpretation':'Positive values mean the candidate is more sensitive to the imposed drift, after pairing each source with its base view.',
         'intervals':intervals,
         'drift_last_step_reset_windows':int(new['target_drift__reanchor_reset'][:,-1].sum()),
         'drift_last_anchor_matches_visible_raw_windows':int(np.sum(np.linalg.norm(new['target_drift__anchor_gate']-old['target_drift__anchor_raw'],axis=1)<1e-8)),
         'per_view_error_changes':{v:{m:{'improved':int(np.sum(error(new,v)[m]<error(old,v)[m]-1e-8)),
             'worsened':int(np.sum(error(new,v)[m]>error(old,v)[m]+1e-8))} for m in ['ADE','FDE']}
             for v in ['base','target_jump','target_drift','target_speed']}}
    (args.root/'drift_response_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
