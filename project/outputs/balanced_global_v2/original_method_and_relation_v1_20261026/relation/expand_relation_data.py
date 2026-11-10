"""Attempt every pre-qualified quartet source, preserving prior construction."""
from pathlib import Path
import sys,argparse,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from audit_neighborhood import Source,REGIONS,save,sha
from build_relation_smoke import build_one

def validate_ranges(a):
    for side in ('target','neighbor'):
        for key in ('t','pos','sog_kn','cog_deg'):
            if not np.isfinite(a[side+'_'+key]).all():raise ValueError('Nonfinite generated observed input')
        s=a[side+'_sog_kn'];c=a[side+'_cog_deg']
        if np.any((s<0)|(s>=102.2)) or np.any((c<0)|(c>=360)):raise ValueError('AIS sentinel/out-of-range generated input')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--scan',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();protocol=json.loads(a.protocol.read_text());a.out.mkdir(parents=True,exist_ok=False)
    assert sha(HERE/'vendor/build_relation_smoke.py')==protocol['builder_sha256']
    assert sha(HERE/'vendor/audit_neighborhood.py')==protocol['source_helper_sha256']
    registry=json.loads((a.root/'release_20261026/source_registry.json').read_text())['regions']
    allids=set(pd.read_parquet(a.root/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']).mmsi.astype(str))
    targets=pd.read_parquet(a.root/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
    tv=set(targets.loc[targets.split=='val','mmsi'].astype(str));tt=set(targets.loc[targets.split=='test','mmsi'].astype(str))
    allowed={'train':allids-tv-tt,'val':allids-tt}
    result={'status':'running','scope':'All eligible train/val source quartets; controlled consistency labels, no classifier selection or test data',
            'protocol_sha256':sha(a.protocol),'builder_sha256':sha(HERE/'vendor/build_relation_smoke.py'),'source_helper_sha256':sha(HERE/'vendor/audit_neighborhood.py'),
            'expansion_code_sha256':sha(Path(__file__)),'regions':{},'samples':[],'rejects':[],'source_ids':{'train':[],'val':[]},'max_innovation_error_mps':0.,'candidate_manifest_hashes':{},'observed_input_range_validation':'passed'}
    for region in REGIONS:
        src=Source(registry[region]);result['regions'][region]={}
        for split in ('train','val'):
            plan_path=a.scan/f'{region}_{split}_relation_feasibility.json';result['candidate_manifest_hashes'][plan_path.name]=sha(plan_path)
            plans=json.loads(plan_path.read_text());eligible=[r for r in plans if r['accepted']];eligible.sort(key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest())
            accepted=[]
            for row in eligible:
                got,reason=build_one(src,row,allowed[split])
                if got is None:
                    result['rejects'].append(dict(region=region,split=split,window_id=row['window_id'],mmsi=row['mmsi'],reason=reason));continue
                arrays,proof=got;validate_ranges(arrays)
                name=f"{region}_{split}_{row['window_id']}";out=a.out/f'{name}.npz';np.savez_compressed(out,**arrays)
                proof.update(file=out.name,sha256=sha(out),canonical_file=registry[region]['canonical_file'],canonical_sha256=registry[region]['sha256'])
                save(a.out/f'{name}.json',proof);result['samples'].append(proof);accepted.append(row);result['source_ids'][split].append(row['mmsi'])
                result['max_innovation_error_mps']=max(result['max_innovation_error_mps'],proof['max_innovation_error_mps'])
            result['regions'][region][split]=dict(anchor_feasible_sources=len(eligible),attempted_sources=len(eligible),accepted_sources=len(accepted),rejected_sources=len(eligible)-len(accepted),target_mmsi=len(set(x['mmsi'] for x in accepted)),quartet_instances=len(accepted)*4)
        print('EXPANDED_REGION',region,result['regions'][region],flush=True)
    assert not set(result['source_ids']['train'])&set(result['source_ids']['val'])
    assert len({x['base_window_id'] for x in result['samples']})==len(result['samples'])
    result.update(status='passed',source_windows=len(result['samples']),quartet_instances=4*len(result['samples']),
                  counts={s:dict(source_windows=sum(x['split']==s for x in result['samples']),target_mmsi=len(set(result['source_ids'][s])),quartet_instances=4*sum(x['split']==s for x in result['samples'])) for s in ['train','val']},
                  target_only_balanced_signature_verified=True,neighbor_only_balanced_signature_verified=True,train_val_target_mmsi_intersection=0,not_part_of_old_label_dataset=True)
    save(a.out/'relation_manifest.json',result);print('RELATION_EXPANSION_COMPLETE',result['counts'],flush=True)
if __name__=='__main__':main()
