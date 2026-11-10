"""Score paired stress using frozen base-training thresholds."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from observed_statistics_v2 import OBSERVED,NAMES,score_observed,metric,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--base-results',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());base=json.loads((a.base_results/'statistics_summary.json').read_text());rows=[];details=[];unavailable=[]
    for entry in manifest['samples']:
        path=a.samples/entry['file'];assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as z:
            sc=score_observed({key:z[key] for key in OBSERVED});y=z['y'];split=str(z['split']);region=str(z['region']);wid=str(z['base_window_id']);mmsi=str(z['target_mmsi'])
        if not all(x['available'] for x in sc):unavailable.append(wid);continue
        for i,x in enumerate(sc):
            meta=dict(base_window_id=wid,mmsi=mmsi,region=region,split=split,case_index=i,y=int(y[i]))
            rows.append({**meta,**{name:x[name] for name in NAMES}});details.append({**meta,**x})
    with (a.base_results/'scores.csv').open(newline='') as f:br=list(csv.DictReader(f))
    assert [(r['base_window_id'],r['case_index'],r['split']) for r in rows]==[(r['base_window_id'],int(r['case_index']),r['split']) for r in br]
    for r,b in zip(rows,br):assert r['target_only']==float(b['target_only'])
    output={'status':'completed','scope':'Neighbor-speed stress with all base training thresholds fixed; no threshold recalibration','protocol_sha256':sha(a.protocol),'scorer_sha256':sha(Path(__file__).with_name('observed_statistics_v2.py')),'code_sha256':sha(Path(__file__)),'base_statistics_sha256':sha(a.base_results/'statistics_summary.json'),'stress_manifest_sha256':sha(a.manifest),'source_windows':len(rows)//4,'scenarios':len(rows),'unavailable_sources':unavailable,'metrics':{}}
    for name in NAMES:
        th=base['metrics'][name]['threshold_train_max_F1'];output['metrics'][name]={'threshold_from_base_train':th,'splits':{}}
        for split in ['train','val']:
            rr=[r for r in rows if r['split']==split];per={}
            for region in sorted(set(r['region'] for r in rr)):
                zz=[r for r in rr if r['region']==region];per[region]=metric([r['y'] for r in zz],[r[name] for r in zz],th);per[region].update(source_windows=len(set(r['base_window_id'] for r in zz)),target_mmsi=len(set(r['mmsi'] for r in zz)))
            pooled=metric([r['y'] for r in rr],[r[name] for r in rr],th)
            output['metrics'][name]['splits'][split]={'pooled':pooled,'macro':{key:float(np.mean([v[key] for v in per.values() if v[key] is not None])) for key in ['AP','AUROC','F1','precision','recall']},'per_region':per}
            if name in ['target_only','neighbor_only']:assert abs(pooled['AP']-.5)<1e-12 and abs(pooled['AUROC']-.5)<1e-12
    for name,value in [('statistics_summary.json',output),('score_components.json',details)]:
        (a.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (a.out/'scores.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({n:output['metrics'][n]['splits']['val']['macro'] for n in NAMES}),flush=True)
if __name__=='__main__':main()
