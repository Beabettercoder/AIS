"""A label-blind paired neighbor-speed sensor-stress view."""
from pathlib import Path
import argparse,hashlib,json,sys,math
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'vendor'))
from maneuver_generator import apply_observation_noise
from observed_statistics_v2 import score_observed,OBSERVED

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed(wid,mid):return int.from_bytes(hashlib.sha256(('speed-stress:'+wid+':'+mid).encode()).digest()[:8],'little')
def select_members(wid,members):return sorted(range(len(members)),key=lambda j:hashlib.sha256(('speed-stress-member:'+wid+':'+str(members[j])).encode()).digest())[:math.ceil(.15*len(members))]
def stress_one(a,wid):
    # Operate on copied public observed arrays; labels/factor/source-truth fields are never read.
    out={k:np.asarray(a[k]).copy() for k in OBSERVED}
    members=np.asarray(a['cohort_mmsi']).astype(str);offsets=out['member_offsets'];t=out['target_t'][0];anchor=max(2,len(t)//2-1);anchor_time=float(t[anchor]);suffix=t[np.arange(anchor+1,len(t))[-3:]]
    records=[]
    for j in select_members(wid,members):
        lo,hi=int(offsets[j]),int(offsets[j+1]);tt=out['neighbor_t'][0,lo:hi];candidates=np.flatnonzero(tt>anchor_time)
        record=dict(member_index=j,mmsi=str(members[j]),seed=seed(wid,str(members[j])),requested=True)
        if len(candidates)<2:
            record.update(applied=False,reason='fewer_than_two_post_anchor_reports');records.append(record);continue
        rng=np.random.default_rng(record['seed']);ix=int(rng.choice(candidates[:-1]));factor=float(rng.uniform(3.,5.))
        assert tt[ix]>anchor_time and ix+1<len(tt)
        before=out['neighbor_sog_kn'][:,lo:hi].copy();proofs=[]
        for case in range(4):
            changed=apply_observation_noise(out['neighbor_t'][case,lo:hi],out['neighbor_pos'][case,lo:hi],out['neighbor_sog_kn'][case,lo:hi],out['neighbor_cog_deg'][case,lo:hi],noise='speed_spike',seed=record['seed'],params={'report_index':ix,'reports':2,'factor':factor})
            out['neighbor_sog_kn'][case,lo:hi]=changed['sog_kn'];proofs.append(changed['metadata']['parameters'])
        assert all(p==proofs[0] for p in proofs)
        latest=np.searchsorted(tt,suffix,side='right')-1;valid=latest>=0;safe=np.maximum(latest,0);valid&=(suffix-tt[safe]<=300)
        hit=valid&(latest>=ix)&(latest<ix+2)
        record.update(applied=True,report_index=ix,flat_indices=[lo+ix,lo+ix+1],factor=factor,times_s=tt[ix:ix+2].tolist(),numeric_changed=bool(np.any(out['neighbor_sog_kn'][:,lo:hi]!=before)),suffix_grid_hits=int(hit.sum()),suffix_grid_count=len(suffix))
        records.append(record)
    # Required observational marginals remain paired under shared corruption.
    assert np.array_equal(out['neighbor_sog_kn'][0],out['neighbor_sog_kn'][2]) and np.array_equal(out['neighbor_sog_kn'][1],out['neighbor_sog_kn'][3])
    for key in ['target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_pos','neighbor_cog_deg','neighbor_t']:
        assert np.array_equal(out[key],a[key]),key
    return out,dict(anchor_time_s=anchor_time,selected_member_count=len(records),members=records,source_numeric_effective=any(x.get('numeric_changed',False) for x in records),source_suffix_overlap=any(x.get('suffix_grid_hits',0)>0 for x in records))

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(a.protocol.read_text());assert sha(HERE/'vendor/maneuver_generator.py')==protocol['generator_sha256']
    old=json.loads(a.manifest.read_text());result=dict(status='running',scope='Separate quartet-shared neighbor report-speed stress; no target change, no label-dependent operator',samples=[],protocol_sha256=sha(a.protocol),base_manifest_sha256=sha(a.manifest),generator_sha256=protocol['generator_sha256'],code_sha256=sha(Path(__file__)),counts={s:{'sources':0,'requested_members':0,'applied_members':0,'numeric_effective_sources':0,'suffix_overlap_sources':0} for s in ['train','val']})
    for item in old['samples']:
        path=a.samples/item['file'];assert sha(path)==item['sha256']
        with np.load(path,allow_pickle=False) as z:
            observed={k:z[k] for k in OBSERVED};observed['cohort_mmsi']=z['cohort_mmsi'];wid=str(z['source_window_id'])
            changed,proof=stress_one(observed,wid)
            # Preserve only evaluation metadata; original hidden source arrays are not copied.
            for key in ['cohort_mmsi','target_mmsi','base_window_id','source_window_id','split','region','y']:changed[key]=z[key].copy()
        score_observed({k:changed[k] for k in OBSERVED}) # validate legal observed data before writing
        out=a.out/item['file'];np.savez_compressed(out,**changed)
        entry={'file':out.name,'sha256':sha(out),'base_file':str(path),'base_sha256':sha(path),'base_window_id':str(changed['base_window_id']),'window_id':wid,'mmsi':str(changed['target_mmsi']),'region':str(changed['region']),'split':str(changed['split']),'speed_stress':proof}
        (a.out/out.with_suffix('.json').name).write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8');result['samples'].append(entry)
        count=result['counts'][entry['split']];count['sources']+=1;count['requested_members']+=proof['selected_member_count'];count['applied_members']+=sum(x['applied'] for x in proof['members']);count['numeric_effective_sources']+=int(proof['source_numeric_effective']);count['suffix_overlap_sources']+=int(proof['source_suffix_overlap'])
    result.update(status='passed',source_windows=len(result['samples']),quartet_instances=4*len(result['samples']))
    (a.out/'stress_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','counts':result['counts']}),flush=True)
if __name__=='__main__':main()
