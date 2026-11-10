"""Exploratory, isolated report-weighted neighbor speed aggregation."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from observed_statistics_v2 import OBSERVED,score_observed,kinematic_weights,metric,threshold,sha

def weighted_segment(s,w):
    if len(s)==0:raise ValueError('No segment values')
    total=float(np.sum(w));fallback=total==0.
    return (float(np.mean(s)) if fallback else float(np.sum(s*w)/total)),fallback,total
def score_report_weighted(o):
    old=score_observed(o) # observed-only validation and exact existing target/member choices
    output=[]
    for case,row in enumerate(old):
        if not row['available']:output.append(dict(available=False));continue
        t=o['target_t'][case];anchor=max(2,len(t)//2-1);pre=np.arange(anchor+1);post=np.arange(anchor+1,len(t))[-3:]
        changes=[];details=[]
        for member in row['usable_members']:
            lo,hi=o['member_offsets'][member:member+2];tt=o['neighbor_t'][case,lo:hi];ss=o['neighbor_sog_kn'][case,lo:hi]
            w=kinematic_weights(tt,o['neighbor_pos'][case,lo:hi],ss,o['neighbor_cog_deg'][case,lo:hi])
            ii=np.searchsorted(tt,t,side='right')-1;safe=np.maximum(ii,0)
            valid=(ii>=0)&(t>=tt[safe])&(t-tt[safe]<=300)
            a=pre[valid[pre]];b=post[valid[post]];assert len(a)>=2 and len(b)>=2
            pi=safe[a];qi=safe[b];ps,pf,pw=weighted_segment(ss[pi],w[pi]);qs,qf,qw=weighted_segment(ss[qi],w[qi]);change=qs/max(ps,.5)-1
            changes.append(change);details.append(dict(member_index=int(member),prefix_report_indices=pi.tolist(),suffix_report_indices=qi.tolist(),prefix_unique_report_count=len(set(pi.tolist())),suffix_unique_report_count=len(set(qi.tolist())),prefix_grid_values=len(pi),suffix_grid_values=len(qi),prefix_plain_mean_kn=float(ss[pi].mean()),suffix_plain_mean_kn=float(ss[qi].mean()),prefix_weighted_mean_kn=ps,suffix_weighted_mean_kn=qs,prefix_sum_w=pw,suffix_sum_w=qw,prefix_fallback=pf,suffix_fallback=qf,relative_change=float(change)))
        assert len(changes)==len(row['usable_members'])
        group=float(np.mean(changes));output.append(dict(available=True,target_change=row['target_change'],neighbor_report_weighted_change=group,relative_report_weighted=abs(row['target_change']-group),original_relative_mean=row['relative_mean'],usable_members=row['usable_members'],members=details,
            prefix_fallbacks=sum(x['prefix_fallback'] for x in details),suffix_fallbacks=sum(x['suffix_fallback'] for x in details)))
    if not all(x['available'] for x in output):return output
    assert output[0]['neighbor_report_weighted_change']==output[2]['neighbor_report_weighted_change']
    assert output[1]['neighbor_report_weighted_change']==output[3]['neighbor_report_weighted_change']
    return output
def load_view(root,view):
    if view=='base':folder=root/'all_sources_v1';manifest_path=folder/'relation_manifest.json'
    else:folder=root/'neighbor_speed_stress_v1';manifest_path=folder/'stress_manifest.json'
    manifest=json.loads(manifest_path.read_text());rows=[];details=[];coverage=[]
    for entry in manifest['samples']:
        p=folder/entry['file'];assert sha(p)==entry['sha256']
        with np.load(p,allow_pickle=False) as z:
            observed={k:z[k] for k in OBSERVED};sc=score_report_weighted(observed)
            y=z['y'];metadata=dict(base_window_id=str(z['base_window_id']),mmsi=str(z['target_mmsi']),split=str(z['split']),region=str(z['region']))
        if not all(s['available'] for s in sc):raise ValueError('Followup must retain all prior available quartets')
        for case,s in enumerate(sc):
            meta={**metadata,'case_index':case,'y':int(y[case])};rows.append({**meta,'relative_report_weighted':s['relative_report_weighted']});details.append({**meta,**s})
        coverage.append({'base_window_id':metadata['base_window_id'],'split':metadata['split'],'prefix_fallbacks_unique_neighbor_states':sc[0]['prefix_fallbacks']+sc[1]['prefix_fallbacks'],'suffix_fallbacks_unique_neighbor_states':sc[0]['suffix_fallbacks']+sc[1]['suffix_fallbacks'],'scored_member_states':len(sc[0]['members'])+len(sc[1]['members'])})
    return rows,details,coverage,sha(manifest_path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    views={v:load_view(a.root,v) for v in ['base','speed_stress']}
    keys=lambda rows:[tuple(r[k] for k in ['base_window_id','mmsi','region','split','case_index','y']) for r in rows]
    assert keys(views['base'][0])==keys(views['speed_stress'][0])
    tr=[r for r in views['base'][0] if r['split']=='train'];th=threshold([r['y'] for r in tr],[r['relative_report_weighted'] for r in tr])
    result={'status':'completed','scope':'Post-diagnostic exploratory report-weighted follow-up, not primary preregistration','protocol_sha256':sha(a.protocol),'code_sha256':sha(Path(__file__)),'unchanged_v2_scorer_sha256':sha(Path(__file__).with_name('observed_statistics_v2.py')),'threshold_base_train':th,'views':{}}
    for view,(rows,details,coverage,manifest_hash) in views.items():
        original_dir=a.root/('base_statistics_v2' if view=='base' else 'speed_stress_statistics_v2')
        with (original_dir/'scores.csv').open(newline='') as f:original=list(csv.DictReader(f))
        expected=[tuple(str(r[k]) for k in ['base_window_id','mmsi','region','split','case_index','y']) for r in original]
        assert [tuple(str(x) for x in key) for key in keys(rows)]==expected
        for detail,old in zip(details,original):assert detail['original_relative_mean']==float(old['relative_mean'])
        view_dir=a.out/view;view_dir.mkdir()
        with (view_dir/'scores.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (view_dir/'score_components.json').write_text(json.dumps(details,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        entry={'source_manifest_sha256':manifest_hash,'same_samples_and_ordinary_mean_replayed':True,'splits':{}}
        for split in ['train','val']:
            rr=[r for r in rows if r['split']==split];cc=[c for c in coverage if c['split']==split];per={}
            for region in sorted(set(r['region'] for r in rr)):
                z=[r for r in rr if r['region']==region];per[region]=metric([r['y'] for r in z],[r['relative_report_weighted'] for r in z],th);per[region].update(source_windows=len(set(r['base_window_id'] for r in z)),target_mmsi=len(set(r['mmsi'] for r in z)))
            entry['splits'][split]={'pooled':metric([r['y'] for r in rr],[r['relative_report_weighted'] for r in rr],th),'macro':{m:float(np.mean([v[m] for v in per.values()])) for m in ['AP','AUROC','F1','precision','recall']},'per_region':per,'source_windows':len(cc),'target_mmsi':len(set(r['mmsi'] for r in rr)),
                'fallbacks_unique_neighbor_states':{key:sum(c[key] for c in cc) for key in ['prefix_fallbacks_unique_neighbor_states','suffix_fallbacks_unique_neighbor_states','scored_member_states']},'sources_with_fallback':sum(c['prefix_fallbacks_unique_neighbor_states']+c['suffix_fallbacks_unique_neighbor_states']>0 for c in cc)}
        result['views'][view]=entry
    (a.out/'statistics_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({v:x['splits']['val'] for v,x in result['views'].items()}),flush=True)
if __name__=='__main__':main()
