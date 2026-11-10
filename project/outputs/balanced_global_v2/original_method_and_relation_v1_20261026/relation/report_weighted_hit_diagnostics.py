"""Read-only operation hits and normalized report weights for the followup."""
from pathlib import Path
import argparse,json
import numpy as np
from observed_statistics_v2 import kinematic_weights,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--followup',type=Path,required=True);a=p.parse_args();r=a.root
    manifest=json.loads((r/'neighbor_speed_stress_v1/stress_manifest.json').read_text());details=json.loads((a.followup/'speed_stress/score_components.json').read_text());lookup={(x['base_window_id'],x['case_index']):x for x in details};records=[]
    for item in manifest['samples']:
        with np.load(r/'neighbor_speed_stress_v1'/item['file'],allow_pickle=False) as z,np.load(item['base_file'],allow_pickle=False) as base:
            tt=z['neighbor_t'];pos=z['neighbor_pos'];s=z['neighbor_sog_kn'];c=z['neighbor_cog_deg'];offsets=z['member_offsets'];old_s=base['neighbor_sog_kn']
            for case in [0,1]:
                row=lookup[item['base_window_id'],case];members={x['member_index']:x for x in row['members']}
                for operator in item['speed_stress']['members']:
                    j=operator['member_index'];lo,hi=map(int,offsets[j:j+2]);entry={'base_window_id':item['base_window_id'],'mmsi':item['mmsi'],'region':item['region'],'split':item['split'],'neighbor_state_representative_case':case,'neighbor_mmsi':operator['mmsi'],'member_index':j,'operator_applied':operator['applied'],'member_used':j in members}
                    if j not in members:
                        records.append(entry);continue
                    md=members[j];idx=np.asarray(md['suffix_report_indices']);w=kinematic_weights(tt[case,lo:hi],pos[case,lo:hi],s[case,lo:hi],c[case,lo:hi]);total=float(w[idx].sum());coeff=np.full(len(idx),1/len(idx)) if total==0 else w[idx]/total
                    changed=np.flatnonzero(s[case,lo:hi]!=old_s[case,lo:hi]);hit=np.isin(idx,changed)
                    entry.update(suffix_fallback=md['suffix_fallback'],suffix_grid_count=len(idx),changed_suffix_grid_hits=int(hit.sum()),changed_suffix_unique_reports=len(set(idx[hit].tolist())),
                                 ordinary_mean_weight_share_on_changed_reports=float(hit.mean()),report_weighted_share_on_changed_reports=float(coeff[hit].sum()),
                                 suffix_report_indices=idx.tolist(),suffix_report_w=w[idx].tolist(),normalized_suffix_weights=coeff.tolist())
                    records.append(entry)
    output={'status':'passed','scope':'Known stress-operation diagnostic only. a-duplicate siblings are collapsed; both neighbor states retained. No predictions or thresholds changed. Positive weights may be numerically tiny; direct share is not a claim of full causal influence.',
            'followup_components_sha256':sha(a.followup/'speed_stress/score_components.json'),'stress_manifest_sha256':sha(r/'neighbor_speed_stress_v1/stress_manifest.json'),'code_sha256':sha(Path(__file__)),'splits':{},'records':records}
    for split in ['train','val']:
        rows=[x for x in records if x['split']==split and x['member_used']];hit=[x for x in rows if x['changed_suffix_grid_hits']>0];shares=np.array([x['report_weighted_share_on_changed_reports'] for x in hit]);ordinary=np.array([x['ordinary_mean_weight_share_on_changed_reports'] for x in hit])
        output['splits'][split]={'requested_member_states':sum(x['split']==split for x in records),'scorer_used_member_states':len(rows),'raw_suffix_hit_member_states':len(hit),'raw_suffix_hit_sources':len(set(x['base_window_id'] for x in hit)),
                                'mean_ordinary_bad_report_share':float(ordinary.mean()) if len(hit) else None,'mean_report_weighted_bad_report_share':float(shares.mean()) if len(hit) else None,
                                'median_report_weighted_bad_report_share':float(np.median(shares)) if len(hit) else None,'member_states_bad_share_increased':sum(x['report_weighted_share_on_changed_reports']>x['ordinary_mean_weight_share_on_changed_reports']+1e-12 for x in hit),
                                'fallback_hit_member_states':sum(x['suffix_fallback'] for x in hit)}
    (a.followup/'hit_diagnostics.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(output['splits']),flush=True)
if __name__=='__main__':main()
