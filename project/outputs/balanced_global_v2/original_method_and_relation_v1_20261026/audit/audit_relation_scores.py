"""Independent score/metric replay from observed quartet arrays only."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

SCORERS=('target_only','neighbor_only','relative_mean','relative_median','relative_reliable')
OBS=('target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','member_offsets')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(ok,message):
    if not ok: raise AssertionError(message)

def observed_weights(t,p,s,c):
    w=np.zeros(len(t),np.float64)
    for j in range(1,len(t)):
        dt=float(t[j]-t[j-1])
        if not 0<dt<=300: continue
        if not (np.isfinite(p[j-1:j+1]).all() and np.isfinite(s[j-1:j+1]).all() and np.isfinite(c[j-1:j+1]).all()):continue
        vp=[]
        for k in (j-1,j):
            a=float(c[k])*np.pi/180
            vp.append(float(s[k])*1852/3600*np.array([np.sin(a),np.cos(a)]))
        error=float(np.linalg.norm((p[j]-p[j-1])/dt-(vp[0]+vp[1])/2))
        w[j]=math.exp(-max(0,error-2.5)**2/8)
    return w

def reconstruct(o):
    result=[]
    for case in range(4):
        target_times=o['target_t'][case];target_speed=o['target_sog_kn'][case]
        anchor=max(2,len(target_times)//2-1)
        before=list(range(anchor+1));after=list(range(anchor+1,len(target_times)))[-3:]
        require(len(after)>=2,'insufficient suffix')
        target_change=float(np.mean(target_speed[after])/max(.5,float(np.mean(target_speed[before])))-1)
        changes=[];weights=[];member_details=[]
        for member in range(len(o['member_offsets'])-1):
            lo,hi=map(int,o['member_offsets'][member:member+2])
            times=o['neighbor_t'][case,lo:hi];speed=o['neighbor_sog_kn'][case,lo:hi]
            reliability=observed_weights(times,o['neighbor_pos'][case,lo:hi],speed,o['neighbor_cog_deg'][case,lo:hi])
            mapped=[]
            for t in target_times:
                eligible=np.flatnonzero((times<=t)&(times>=t-300)&np.isfinite(speed))
                mapped.append(int(eligible[-1]) if len(eligible) else -1)
            a=[mapped[k] for k in before if mapped[k]>=0];b=[mapped[k] for k in after if mapped[k]>=0]
            if len(a)<2 or len(b)<2:continue
            change=float(np.mean(speed[b])/max(.5,float(np.mean(speed[a])))-1)
            weight=float(np.sqrt(np.mean(reliability[a])*np.mean(reliability[b])))
            changes.append(change);weights.append(weight)
            member_details.append(dict(member=member,aligned_prefix_values=len(a),aligned_suffix_values=len(b),
                                       distinct_prefix_reports=len(set(a)),distinct_suffix_reports=len(set(b))))
        if not changes:
            result.append(dict(available=False));continue
        ordinary=float(np.mean(changes));median=float(np.median(changes))
        trusted=ordinary if sum(weights)<=0 else float(sum(v*w for v,w in zip(changes,weights))/sum(weights))
        result.append(dict(available=True,target_only=abs(target_change),neighbor_only=abs(ordinary),
                           relative_mean=abs(target_change-ordinary),relative_median=abs(target_change-median),
                           relative_reliable=abs(target_change-trusted),member_details=member_details))
    return result

def select_threshold(y,s):
    best=(-1.,-float('inf'))
    for threshold in np.unique(s):
        pred=s>=threshold;tp=int(np.sum(pred & (y==1)));fp=int(np.sum(pred & (y==0)));fn=int(np.sum(~pred & (y==1)))
        f1=2*tp/max(2*tp+fp+fn,1)
        if f1>best[0]+1e-12 or (abs(f1-best[0])<=1e-12 and threshold>best[1]):best=(f1,float(threshold))
    return best[1]

def metrics(y,s,threshold):
    pred=s>=threshold;tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)));tn=int(np.sum(~pred&(y==0)))
    both=len(np.unique(y))==2
    return dict(samples=len(y),positive=int(y.sum()),AP=float(average_precision_score(y,s)) if both else None,
                AUROC=float(roc_auc_score(y,s)) if both else None,F1=2*tp/max(2*tp+fp+fn,1),
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),confusion=[[tn,fp],[fn,tp]])

def compare_metrics(actual,recorded,label):
    for k in ('samples','positive','confusion'):require(actual[k]==recorded[k],f'{label}/{k}')
    for k in ('AP','AUROC','F1','precision','recall'):
        if actual[k] is None:require(recorded[k] is None,f'{label}/{k}')
        else:require(abs(actual[k]-recorded[k])<1e-11,f'{label}/{k}')

def audit_stress_manifest(path):
    manifest=json.loads(path.read_text(encoding='utf-8'));sources=[];counts={s:dict(sources=0,requested_members=0,applied_members=0,numeric_effective_sources=0,suffix_overlap_sources=0) for s in ['train','val']}
    for entry in manifest['samples']:
        current=path.parent/entry['file'];base=Path(entry['base_file']);require(sha(current)==entry['sha256'] and sha(base)==entry['base_sha256'],'stress/base hash')
        with np.load(base,allow_pickle=False) as z:original={k:z[k].copy() for k in OBS+('cohort_mmsi','target_mmsi','base_window_id','source_window_id','split','region','y')}
        with np.load(current,allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
        wid=str(original['source_window_id']);members=original['cohort_mmsi'].astype(str);offsets=original['member_offsets'];t=original['target_t'][0];anchor=max(2,len(t)//2-1);anchor_time=float(t[anchor]);suffix=t[anchor+1:][-3:]
        selected=sorted(range(len(members)),key=lambda j:hashlib.sha256(('speed-stress-member:'+wid+':'+str(members[j])).encode()).digest())[:math.ceil(.15*len(members))]
        proof=entry['speed_stress'];require(proof['anchor_time_s']==anchor_time and proof['selected_member_count']==len(selected),'stress plan')
        require([x['member_index'] for x in proof['members']]==selected,'stress member choice')
        expected=original['neighbor_sog_kn'].copy();applied=0;effective=False;overlap=False
        for index,member in enumerate(selected):
            record=proof['members'][index];lo,hi=map(int,offsets[member:member+2]);tt=original['neighbor_t'][0,lo:hi]
            seed=int.from_bytes(hashlib.sha256(('speed-stress:'+wid+':'+str(members[member])).encode()).digest()[:8],'little')
            require(seed==record['seed'] and str(members[member])==record['mmsi'],'stress seed/member')
            candidates=np.flatnonzero(tt>anchor_time)
            if len(candidates)<2:require(not record['applied'],'stress unavailable');continue
            rng=np.random.default_rng(seed);start=int(rng.choice(candidates[:-1]));factor=float(rng.uniform(3,5));applied+=1
            require(record['applied'] and record['report_index']==start and record['factor']==factor,'stress sampled parameters')
            require(tt[start]>anchor_time and record['times_s']==tt[start:start+2].tolist(),'stress causal onset')
            expected[:,lo+start:lo+start+2]=np.minimum(expected[:,lo+start:lo+start+2]*factor,102.1)
            changed=bool(np.any(expected[:,lo:hi]!=original['neighbor_sog_kn'][:,lo:hi]));require(changed==record['numeric_changed'],'stress numeric effect');effective|=changed
            hits=0
            for query in suffix:
                eligible=np.flatnonzero(tt<=query)
                if len(eligible):
                    last=int(eligible[-1]);hits+=int(query-tt[last]<=300 and start<=last<start+2)
            require(hits==record['suffix_grid_hits'],'stress suffix hits');overlap|=hits>0
        require(np.array_equal(actual['neighbor_sog_kn'],expected),'stress speed values')
        for key in original:
            if key!='neighbor_sog_kn':require(np.array_equal(actual[key],original[key]),'stress changed forbidden field '+key)
        require(proof['source_numeric_effective']==effective and proof['source_suffix_overlap']==overlap,'stress source flag')
        split=str(original['split']);c=counts[split];c['sources']+=1;c['requested_members']+=len(selected);c['applied_members']+=applied;c['numeric_effective_sources']+=int(effective);c['suffix_overlap_sources']+=int(overlap)
        sources.append(dict(file=str(current),sha256=sha(current),base_window_id=str(original['base_window_id']),split=split))
    require(counts==manifest['counts'],'stress aggregate counts')
    return sources,dict(status='passed',stress_manifest_sha256=sha(path),counts=counts,all_target_and_nonspeed_neighbor_fields_unchanged=True,shared_label_blind_operator_replayed=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--statistics',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--stress-manifest',type=Path);parser.add_argument('--calibration-statistics',type=Path)
    args=parser.parse_args();require(not args.out.exists(),'Refusing overwrite')
    summary=json.loads((args.statistics/'statistics_summary.json').read_text(encoding='utf-8'))
    with (args.statistics/'scores.csv').open(newline='',encoding='utf-8') as stream:recorded=list(csv.DictReader(stream))
    recorded_index={(r['base_window_id'],int(r['case_index'])):r for r in recorded}
    require(len(recorded_index)==len(recorded),'Duplicate score keys')
    excluded={x['base_window_id'] if isinstance(x,dict) else x for x in summary['unavailable_sources']}
    stress_proof=None
    if args.stress_manifest:
        require(args.calibration_statistics is not None,'Stress needs base-train calibration source')
        require(sha(args.stress_manifest)==summary['stress_manifest_sha256'],'Stress manifest identity')
        require(sha(args.calibration_statistics/'statistics_summary.json')==summary['base_statistics_sha256'],'Base calibration identity')
        sources,stress_proof=audit_stress_manifest(args.stress_manifest)
    else:sources=summary['sources']
    replay=[];diagnostics=[];file_checks=[];maximum_error=0.
    for source in sources:
        path=Path(source['file']);require(sha(path)==source['sha256'],'Source hash changed')
        with np.load(path,allow_pickle=False) as archive:
            observed={k:archive[k].copy() for k in OBS}
            for k in ('target_sog_kn','neighbor_sog_kn'):
                require(np.isfinite(observed[k]).all() and (observed[k]>=0).all() and (observed[k]<102.2).all(),'Invalid speed data requires explicit protocol change')
            for k in ('target_cog_deg','neighbor_cog_deg'):
                require(np.isfinite(observed[k]).all() and (observed[k]>=0).all() and (observed[k]<360).all(),'Invalid course data requires explicit protocol change')
            scores=reconstruct(observed)
            # Labels/IDs are not used inside reconstruct().
            y=archive['y'].copy();wid=str(archive['base_window_id']);mid=str(archive['target_mmsi']);split=str(archive['split']);region=str(archive['region'])
        available=all(x['available'] for x in scores)
        require((wid in excluded)==(not available),'Unavailable quartet mismatch')
        if not available:continue
        require(np.array_equal(y,[0,1,1,0]),'Not the declared quartet')
        require(scores[0]['target_only']==scores[1]['target_only'] and scores[2]['target_only']==scores[3]['target_only'],'Target-only scores leak counterpart')
        require(scores[0]['neighbor_only']==scores[2]['neighbor_only'] and scores[1]['neighbor_only']==scores[3]['neighbor_only'],'Neighbor-only scores leak counterpart')
        for case,score in enumerate(scores):
            row=recorded_index[(wid,case)]
            for k,value in [('mmsi',mid),('split',split),('region',region),('y',str(int(y[case])))]:require(row[k]==value,'Metadata mismatch')
            for k in SCORERS:
                error=abs(float(row[k])-score[k]);maximum_error=max(maximum_error,error);require(error<1e-10,'Score mismatch')
            replay.append(dict(base_window_id=wid,mmsi=mid,split=split,region=region,y=int(y[case]),**{k:score[k] for k in SCORERS}))
            diagnostics.append(dict(base_window_id=wid,case_index=case,member_details=score['member_details']))
        file_checks.append(dict(path=str(path),sha256=sha(path)))
    require(len(replay)==len(recorded),'Missing rows')
    train=[x for x in replay if x['split']=='train'];val=[x for x in replay if x['split']=='val']
    for key in ('base_window_id','mmsi'):require(not set(x[key] for x in train)&set(x[key] for x in val),'Split leakage')
    checked={};calibration_rows=train
    if args.calibration_statistics:
        with (args.calibration_statistics/'scores.csv').open(newline='',encoding='utf-8') as stream:
            base_rows=list(csv.DictReader(stream))
        require([(r['base_window_id'],int(r['case_index']),r['split'],int(r['y'])) for r in base_rows]==[(r['base_window_id'],int(r['case_index']),r['split'],int(r['y'])) for r in recorded],'Base/stress pairing')
        calibration_rows=[{'y':int(row['y']),**{key:float(row[key]) for key in SCORERS}} for row in base_rows if row['split']=='train']
    for scorer in SCORERS:
        threshold=select_threshold(np.array([x['y'] for x in calibration_rows]),np.array([x[scorer] for x in calibration_rows]))
        key='threshold_from_base_train' if args.stress_manifest else 'threshold_train_max_F1'
        require(abs(threshold-summary['metrics'][scorer][key])<1e-10,'Training threshold mismatch')
        checked[scorer]={}
        for split,rows in [('train',train),('val',val)]:
            report=summary['metrics'][scorer]['splits'][split];yy=np.array([x['y'] for x in rows]);ss=np.array([x[scorer] for x in rows]);mm=metrics(yy,ss,threshold)
            compare_metrics(mm,report['pooled'],f'{scorer}/{split}/pooled');domains={}
            for region in sorted(set(x['region'] for x in rows)):
                z=[x for x in rows if x['region']==region];m=metrics(np.array([x['y'] for x in z]),np.array([x[scorer] for x in z]),threshold)
                compare_metrics(m,report['per_region'][region],f'{scorer}/{split}/{region}');domains[region]=m
            macro={k:float(np.mean([m[k] for m in domains.values() if m[k] is not None])) for k in ('AP','AUROC','F1','precision','recall')}
            for k,value in macro.items():require(abs(value-report['macro'][k])<1e-11,'Macro mismatch')
            if scorer in ('target_only','neighbor_only'):require(abs(mm['AP']-.5)<1e-12 and abs(mm['AUROC']-.5)<1e-12,'Single-side negative control failed')
            checked[scorer][split]=dict(pooled=mm,macro=macro)
    args.out.mkdir(parents=True)
    out=dict(status='passed',scope='Independent observed-array score replay, all retained quartet instances, train-only threshold and pooled/per-region metrics; no model training.',
             statistics_summary_sha256=sha(args.statistics/'statistics_summary.json'),sources=file_checks,
             independent_source_windows=len(file_checks),instances=len(replay),maximum_score_error=maximum_error,
             all_sog_cog_finite_and_valid=True,train_val_target_identity_overlap=0,metrics=checked,stress_generation_audit=stress_proof,auditor_sha256=sha(Path(__file__)),
             alignment_note='>=2 aligned grid values do not necessarily mean two independent reports; distinct report counts are retained separately.')
    (args.out/'independent_relation_score_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.out/'alignment_distinct_report_counts.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','sources':len(file_checks),'instances':len(replay),'max_score_error':maximum_error}))

if __name__=='__main__':main()
