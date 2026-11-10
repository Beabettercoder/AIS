"""Independent observed-array and cluster-CI audit of the exploratory followup."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from audit_relation_scores import OBS,SCORERS,observed_weights,select_threshold,metrics,compare_metrics,sha,require

def load_csv(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def reconstruct(o):
    out=[]
    for case in range(4):
        tt=o['target_t'][case];ss=o['target_sog_kn'][case];anchor=max(2,len(tt)//2-1);pre=list(range(anchor+1));post=list(range(anchor+1,len(tt)))[-3:]
        target=float(ss[post].mean()/max(float(ss[pre].mean()),.5)-1);changes=[];details=[]
        for member in range(len(o['member_offsets'])-1):
            lo,hi=map(int,o['member_offsets'][member:member+2]);t=o['neighbor_t'][case,lo:hi];s=o['neighbor_sog_kn'][case,lo:hi]
            w=observed_weights(t,o['neighbor_pos'][case,lo:hi],s,o['neighbor_cog_deg'][case,lo:hi]);aligned=[]
            for time in tt:
                ix=np.flatnonzero(t<=time);j=int(ix[-1]) if len(ix) else -1
                aligned.append(j if j>=0 and time-t[j]<=300 else -1)
            before=[aligned[k] for k in pre if aligned[k]>=0];after=[aligned[k] for k in post if aligned[k]>=0]
            if len(before)<2 or len(after)<2:continue
            means=[];totals=[];fallback=[]
            for indices in [before,after]:
                ww=w[indices];vv=s[indices];total=float(ww.sum());f=total==0
                means.append(float(vv.mean()) if f else float(np.dot(ww,vv)/total));totals.append(total);fallback.append(f)
            change=means[1]/max(means[0],.5)-1;changes.append(change)
            details.append(dict(member_index=member,prefix_report_indices=before,suffix_report_indices=after,prefix_weighted_mean_kn=means[0],suffix_weighted_mean_kn=means[1],prefix_sum_w=totals[0],suffix_sum_w=totals[1],prefix_fallback=fallback[0],suffix_fallback=fallback[1],relative_change=change))
        require(changes,'Followup removed a source')
        out.append(dict(score=abs(target-float(np.mean(changes))),target_change=target,members=details))
    return out

def ci(x):return dict(lower_2_5=float(np.percentile(x,2.5)),upper_97_5=float(np.percentile(x,97.5)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--relation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();root=args.relation;follow=root/'report_weighted_statistics_v1'
    require(not args.out.exists(),'Refusing overwrite');summary=json.loads((follow/'statistics_summary.json').read_text());all_rows={};maxerr=0.;fallcounts={};manifest_hashes={}
    for view,folder,manifest_name,oldscore in [('base','all_sources_v1','relation_manifest.json','base_statistics_v2'),('speed_stress','neighbor_speed_stress_v1','stress_manifest.json','speed_stress_statistics_v2')]:
        manifest=json.loads((root/folder/manifest_name).read_text());manifest_hashes[view]=sha(root/folder/manifest_name)
        require(manifest_hashes[view]==summary['views'][view]['source_manifest_sha256'],'manifest identity')
        rows=load_csv(follow/view/'scores.csv');original=load_csv(root/oldscore/'scores.csv')
        details=json.loads((follow/view/'score_components.json').read_text())
        lookup={(r['base_window_id'],int(r['case_index'])):r for r in rows};oldlookup={(r['base_window_id'],int(r['case_index'])):r for r in original};component={(r['base_window_id'],int(r['case_index'])):r for r in details}
        require(list(lookup)==list(oldlookup),'Changed source/row selection');verified=[]
        fallcounts[view]={s:dict(prefix_fallbacks_unique_neighbor_states=0,suffix_fallbacks_unique_neighbor_states=0,scored_member_states=0,sources_with_fallback=0) for s in ['train','val']}
        for source in manifest['samples']:
            p=root/folder/source['file'];require(sha(p)==source['sha256'],'source changed')
            with np.load(p,allow_pickle=False) as z:
                calculated=reconstruct({k:z[k].copy() for k in OBS})
                wid=str(z['base_window_id']);mid=str(z['target_mmsi']);region=str(z['region']);split=str(z['split']);y=z['y'].copy()
            source_has_fallback=False
            for case,calc in enumerate(calculated):
                row=lookup[wid,case];comp=component[wid,case];old=oldlookup[wid,case]
                for key,value in [('mmsi',mid),('region',region),('split',split),('y',str(y[case]))]:require(row[key]==old[key]==value,'metadata changed')
                err=abs(calc['score']-float(row['relative_report_weighted']));maxerr=max(maxerr,err);require(err<1e-10,'weighted score')
                require(abs(calc['target_change']-comp['target_change'])<1e-12,'target changed')
                require(comp['original_relative_mean']==float(old['relative_mean']),'ordinary comparator altered')
                require(len(calc['members'])==len(comp['members']),'member selection changed')
                for actual,recorded in zip(calc['members'],comp['members']):
                    for key in ('member_index','prefix_report_indices','suffix_report_indices','prefix_fallback','suffix_fallback'):require(actual[key]==recorded[key],'segment metadata '+key)
                    for key in ('prefix_weighted_mean_kn','suffix_weighted_mean_kn','prefix_sum_w','suffix_sum_w','relative_change'):require(abs(actual[key]-recorded[key])<1e-10,'segment values '+key)
                if case<2:
                    for member in calc['members']:
                        fallcounts[view][split]['prefix_fallbacks_unique_neighbor_states']+=int(member['prefix_fallback']);fallcounts[view][split]['suffix_fallbacks_unique_neighbor_states']+=int(member['suffix_fallback']);fallcounts[view][split]['scored_member_states']+=1
                        source_has_fallback|=member['prefix_fallback'] or member['suffix_fallback']
                verified.append(dict(base_window_id=wid,mmsi=mid,region=region,split=split,case_index=case,y=int(y[case]),score=calc['score']))
            fallcounts[view][split]['sources_with_fallback']+=int(source_has_fallback)
        require(len(verified)==1540,'instance count');all_rows[view]=verified
    train=[x for x in all_rows['base'] if x['split']=='train'];threshold=select_threshold(np.array([x['y'] for x in train]),np.array([x['score'] for x in train]));require(abs(threshold-summary['threshold_base_train'])<1e-10,'train-only threshold')
    for view,rows in all_rows.items():
        for split in ['train','val']:
            rr=[x for x in rows if x['split']==split];report=summary['views'][view]['splits'][split]
            actual=metrics(np.array([x['y'] for x in rr]),np.array([x['score'] for x in rr]),threshold);compare_metrics(actual,report['pooled'],'pooled');domains={}
            for region in sorted(set(x['region'] for x in rr)):
                z=[x for x in rr if x['region']==region];m=metrics(np.array([x['y'] for x in z]),np.array([x['score'] for x in z]),threshold);domains[region]=m;compare_metrics(m,report['per_region'][region],region)
            for key in ['AP','AUROC','F1','precision','recall']:require(abs(np.mean([x[key] for x in domains.values()])-report['macro'][key])<1e-10,'macro metric')
            for key,value in report['fallbacks_unique_neighbor_states'].items():require(value==fallcounts[view][split][key],'fallback denominator')
            require(report['sources_with_fallback']==fallcounts[view][split]['sources_with_fallback'],'fallback source count')
    vals=[[x for x in all_rows[v] if x['split']=='val'] for v in ['base','speed_stress']]
    require([[x[k] for k in ('base_window_id','mmsi','region','case_index','y')] for x in vals[0]]==[[x[k] for k in ('base_window_id','mmsi','region','case_index','y')] for x in vals[1]],'paired val')
    y=np.array([x['y'] for x in vals[0]]);ids=np.array([x['mmsi'] for x in vals[0]]);u=np.unique(ids);groups=[np.flatnonzero(ids==m) for m in u];scores=np.array([[x['score'] for x in rr] for rr in vals])
    old=json.loads((root/'bootstrap_v1/paired_mmsi_bootstrap.json').read_text());boot=json.loads((follow/'bootstrap/paired_bootstrap.json').read_text());rng=np.random.default_rng(old['seed']);digest=hashlib.sha256()
    with np.load(follow/'bootstrap/draw_metrics.npz',allow_pickle=False) as z:stored=z['metrics'];require(np.array_equal(z['mmsi'],u),'cluster identities')
    with np.load(root/'bootstrap_v1/bootstrap_draw_metrics.npz',allow_pickle=False) as z:oldvalues=z['metrics'];oldnames=z['scorers'].astype(str).tolist()
    expected=np.empty_like(stored)
    for b in range(old['replicates']):
        draw=rng.integers(len(u),size=len(u));digest.update(draw.astype('<i8').tobytes());ii=np.concatenate([groups[k] for k in draw])
        for view in range(2):expected[b,view]=[average_precision_score(y[ii],scores[view,ii]),roc_auc_score(y[ii],scores[view,ii])]
    bootstrap_error=float(np.max(abs(expected-stored)));require(bootstrap_error<1e-12,'bootstrap values');require(digest.hexdigest()==old['draws_sha256']==boot['draws_sha256'],'same bootstrap draws')
    points=np.array([[average_precision_score(y,s),roc_auc_score(y,s)] for s in scores])
    for view,name in enumerate(['base','speed_stress']):
        for m,metric in enumerate(['AP','AUROC']):
            obj=boot['points_and_intervals'][name][metric];require(abs(obj['point']-points[view,m])<1e-12,'bootstrap point')
            for key,value in ci(expected[:,view,m]).items():require(abs(value-obj[key])<1e-12,'bootstrap CI')
    for obj in boot['paired_contrasts']:
        view=['base','speed_stress'].index(obj['view']);j=oldnames.index(obj['comparator'])
        for m,metric in enumerate(['AP','AUROC']):
            delta=expected[:,view,m]-oldvalues[:,view,j,m];point=points[view,m]-old['metric_intervals'][obj['view']][obj['comparator']][metric]['point']
            require(abs(point-obj[metric]['point'])<1e-12,'paired point')
            for key,value in ci(delta).items():require(abs(value-obj[metric][key])<1e-12,'paired interval')
    for m,metric in enumerate(['AP','AUROC']):
        require(abs(points[1,m]-points[0,m]-boot['stress_minus_base'][metric]['point'])<1e-12,'stress delta')
        for key,value in ci(expected[:,1,m]-expected[:,0,m]).items():require(abs(value-boot['stress_minus_base'][metric][key])<1e-12,'stress CI')
    output=dict(status='passed',scope='Post-diagnostic exploratory candidate: observed-array formulas, target/member/sample invariance, segment fallbacks, fixed base-training threshold and all2000 reused target-MMSI bootstrap draws independently verified.',source_windows=385,instances_per_view=1540,maximum_score_error=maxerr,maximum_bootstrap_metric_error=bootstrap_error,threshold_base_train=threshold,fallback_counts=fallcounts,draws_sha256=digest.hexdigest(),source_summary_sha256=sha(follow/'statistics_summary.json'),source_bootstrap_sha256=sha(follow/'bootstrap/paired_bootstrap.json'),auditor_sha256=sha(Path(__file__)))
    with args.out.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2)
    print(json.dumps(output))
if __name__=='__main__':main()
