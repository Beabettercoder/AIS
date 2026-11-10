"""Summarize all selected-model strata from the independently verified audit."""
from pathlib import Path
import argparse,json
from audit_helpers import distribution,jwrite,sha,require,METRICS


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.audit.read_text());require(d['status']=='passed','Complete verified audit required')
    grouped={}
    for run in d['runs']:
        for axis,levels in run['validation_strata'].items():
            for level,metrics in levels.items():
                grouped.setdefault((run['backbone'],run['arm'],axis,level),[]).append((run['seed'],metrics))
    rows=[]
    for (b,arm,axis,level),values in sorted(grouped.items()):
        values.sort();require([s for s,_ in values]==[0,1,2],'All three seeds required')
        first=values[0][1]
        counts={r:{k:x[k] for k in ('samples','positive_samples','negative_samples','ranking_metric_status')} for r,x in first['per_region'].items()}
        for _,m in values[1:]:
            require({r:{k:x[k] for k in ('samples','positive_samples','negative_samples','ranking_metric_status')} for r,x in m['per_region'].items()}==counts,'Stratum membership differs by seed')
        row=dict(backbone=b,arm=arm,stratifier=axis,level=level,windows=first['pooled']['samples'],positive_windows=first['pooled']['positive_samples'],per_region_counts=counts,seeds=[0,1,2])
        for metric in METRICS+('FPR',):
            valid=[m['macro'][metric] for _,m in values if m['macro'][metric] is not None]
            row[metric]=distribution(valid)
            row[metric]['contributing_regions_per_seed']=[m['metric_contributors'][metric] for _,m in values]
        rows.append(row)
    jwrite(a.out,dict(status='passed',source_audit_sha256=sha(a.audit),rows=rows,
                      scope='Selected-model validation strata. All arms retained; missing single-class ranking metrics are not filled.',
                      threshold_scope='Each run has its own validation-max-F1 threshold; FPR is not a matched-FAR comparison.',
                      context_scope='Shared common-support strata; self_quality includes this same availability policy.'))
    print(json.dumps(dict(status='passed',stratum_rows=len(rows))))


if __name__=='__main__':main()
