"""Independent count, coverage and four-cell source-observation audit."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def require(v,msg):
    if not v:raise AssertionError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def velocity(s,c):
    a=np.deg2rad(c);return np.stack((s*np.sin(a),s*np.cos(a)),axis=-1)*(1852/3600)
def innovation(t,p,s,c):
    v=velocity(s,c);return np.diff(p,axis=0)/np.diff(t)[:,None]-.5*(v[1:]+v[:-1])
def transformed(t,p,s,c,onset):
    u=np.clip((t-onset)/300.,0,1);new_s=s*(1-.3*(3*u*u-2*u*u*u))
    dv=velocity(new_s,c)-velocity(s,c);offset=np.zeros_like(p)
    for k in range(1,len(t)):offset[k]=offset[k-1]+.5*(dv[k]+dv[k-1])*(t[k]-t[k-1])
    return p+offset,new_s,c.copy()
def box_for(meta,start,end):
    days=range(int(np.floor((start-600)/86400)),int(np.floor(end/86400))+1)
    keys=[datetime.datetime.fromtimestamp(d*86400,datetime.timezone.utc).strftime('%Y-%m-%d') for d in days]
    if any(k not in meta['coverage_bbox_by_utc_day'] for k in keys):return None
    boxes=np.asarray([meta['coverage_bbox_by_utc_day'][k] for k in keys],float)
    b=np.array([boxes[:,0].max(),boxes[:,1].max(),boxes[:,2].min(),boxes[:,3].min()])
    if b[0]>=b[2] or b[1]>=b[3]:return None
    lon0=(meta['core_bbox'][0]+meta['core_bbox'][2])/2;lat0=(meta['core_bbox'][1]+meta['core_bbox'][3])/2
    return np.array([(b[0]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[1]-lat0)*111195,
                     (b[2]-lon0)*111195*np.cos(np.deg2rad(lat0)),(b[3]-lat0)*111195])


def audit_scan(stage,parent,registry):
    nb=stage/'neighborhood';scan=nb/'scan_final';summary=read(scan/'neighborhood_scan_summary.json')
    require(summary['status']=='passed','Scan status')
    regions=summary['region_sources'];region_proof=[];allsteps=0;total_edges=0
    for item in regions:
        region=item['region'];report_path=scan/item['report'];report=read(report_path)
        require(sha(report_path)==item['report_sha256'],'Final region report hash')
        require(sha(nb/item['result_source_dir']/item['report'])==item['report_sha256'],'Version source report differs')
        require(sha(nb/item['scanner'])==item['scanner_sha256'],'Versioned scanner changed')
        require(report['status']=='passed','Incomplete region')
        require(report['source_sha256']==registry[region]['sha256'],'Source version differs')
        configs=report['configs'];require(len(configs)==17,'Expected 17 configs')
        cfg={c['name']:c for c in configs};total_edges+=report['neighbor_reports_recovered']
        for split in ('train','val'):
            source=parent/'data'/region/f'{split}.npz'
            with np.load(source,allow_pickle=False) as z:
                mask=z['mask'];pos=z['target_pos'];times=z['time_s'];original=z['context_valid'];ids=z['window_id']
                rows=read(scan/f'{region}_{split}_window_scan.json')
                require(len(rows)==len(mask),'Window scan count')
                valid_counts=np.asarray([r['valid_steps_by_config'] for r in rows],int)
                require(np.array_equal([r['window_id'] for r in rows],ids),'Window scan order')
                require(np.array_equal(valid_counts[:,0],(original&mask).sum(1)),'Baseline per-window validity differs')
                require(np.array_equal([r['steps'] for r in rows],mask.sum(1)),'Window mask count')
                boxes=[box_for(registry[region],float(times[j,0]),float(times[j,int(mask[j].sum())-1])) for j in range(len(mask))]
                covered={r:0 for r in (8.,4.,3.5,3.)};unknown=0
                for j,box in enumerate(boxes):
                    p=pos[j,mask[j]]
                    if box is None:
                        unknown+=len(p)
                        for radius in covered:covered[radius]+=len(p)
                    else:
                        for radius in covered:
                            r=radius*1852
                            covered[radius]+=int(((p[:,0]-r>=box[0])&(p[:,0]+r<=box[2])&(p[:,1]-r>=box[1])&(p[:,1]+r<=box[3])).sum())
                stats=report['splits'][split]['config_results'];steps=int(mask.sum());allsteps+=steps
                for ci,c in enumerate(configs):
                    q=stats[c['name']]
                    require(q['valid_steps']==steps,'Config step denominator')
                    require(q['affine_valid_steps']==int(valid_counts[:,ci].sum()),'Window/config validity aggregate')
                    require(q['covered_steps']==covered[c['radius_nm']],'Radius-specific source coverage')
                    require(q['unknown_coverage_steps']==unknown,'Unknown coverage count')
                    require(q['affine_valid_steps']<=q['trusted_ge4_steps']<=q['covered_steps'],'Impossible support bounds')
                # Counts before source-coverage masking must shrink when membership thresholds tighten.
                factorial=configs[:16]
                for a in factorial:
                    for b in factorial:
                        if b['radius_nm']<=a['radius_nm'] and b['heading_deg']<=a['heading_deg'] and b['min_speed_kn']>=a['min_speed_kn']:
                            require(stats[b['name']]['total_trusted_count']<=stats[a['name']]['total_trusted_count'],'Non-monotone membership count')
                require(covered[3.]>=covered[3.5]>=covered[4.]>=covered[8.],'Non-monotone ROI coverage')
        region_proof.append(dict(region=region,source_version=item['result_source_dir'],scanner=item['scanner'],scanner_sha256=item['scanner_sha256'],baseline_per_window_replay=True,all_radius_coverage_recomputed=True))
    require(total_edges==summary['neighbor_reports_recovered_and_checked'],'Recovered edge aggregate')
    # Regional aggregation check; all additive measures are verified independently.
    additive=('valid_steps','covered_steps','unknown_coverage_steps','plain_ge4_steps','trusted_ge3_steps','trusted_ge4_steps','kish_ge3_steps','affine_valid_steps','geometry_rejections','condition_rejections','total_trusted_count')
    for split in ('train','val'):
        for name,q in summary['splits'][split]['config_results'].items():
            for key in additive:
                require(q[key]==sum(read(scan/i['report'])['splits'][split]['config_results'][name][key] for i in regions),'Summary aggregate '+key)
    return dict(status='passed',target_steps=allsteps,source_windows=9520,configurations=17,neighbor_reports_checked_by_scanner=total_edges,
                scope='Independent baseline per-window agreement, complete radius-specific ROI coverage recomputation, aggregate counts and membership monotonicity. The scanner separately replays all listed neighbor observations; that full replay is not duplicated here.',
                region_versions=region_proof,summary_sha256=sha(scan/'neighborhood_scan_summary.json'))


def load_canonical(meta,ids):
    path=Path(meta['canonical_file']);require(sha(path)==meta['sha256'],'Canonical source hash')
    field=pq.read_schema(path).field('mmsi').type
    selected=[int(x) for x in ids] if pa.types.is_integer(field) else list(ids)
    d=pd.read_parquet(path,columns=['mmsi','timestamp','lon','lat','sog','cog'],filters=[('mmsi','in',selected)])
    d['id']=d.mmsi.astype(str);d['order']=np.arange(len(d))
    d=d.sort_values(['id','timestamp','order'],kind='stable').drop_duplicates(['id','timestamp'],keep='last')
    lon0=(meta['core_bbox'][0]+meta['core_bbox'][2])/2;lat0=(meta['core_bbox'][1]+meta['core_bbox'][3])/2
    d['x']=(d.lon-lon0)*111195*np.cos(np.deg2rad(lat0));d['y']=(d.lat-lat0)*111195
    return {m:g.reset_index(drop=True) for m,g in d.groupby('id',sort=False)}


def check_source_arrays(z,members,source,meta,report,allowed):
    t=z['source_target_t'];p=z['source_target_pos'];s=z['source_target_sog_kn'];c=z['source_target_cog_deg']
    target=str(z['target_mmsi']);anchor=float(report['anchor_time_s']);onset=float(report['onset_s']);last=float(t[-1])
    require(target not in members and 4<=len(members)<=8,'Fixed cohort size/target exclusion')
    require(all(m in allowed for m in members),'Disallowed source context identity')
    def retrieve(m,times):
        frame=source[m].set_index('timestamp').loc[times]
        return frame[['x','y']].to_numpy(float),frame.sog.to_numpy(float),frame.cog.to_numpy(float)
    sp,ss,sc=retrieve(target,t)
    require(np.array_equal(s,ss) and np.array_equal(c,sc) and np.allclose(p,sp,rtol=0,atol=1e-8),'Original target source mismatch')
    require(onset==anchor+1 and anchor==t[int(report['anchor_step'])],'Causal anchor/time')
    require(int(report['anchor_step'])==max(2,len(t)//2-1),'Label-blind anchor index')
    offsets=z['member_offsets'];nt=z['source_neighbor_t']
    for k,m in enumerate(members):
        sl=slice(offsets[k],offsets[k+1]);tt=nt[sl]
        pp,ss,cc=retrieve(m,tt)
        require(np.allclose(z['source_neighbor_pos'][sl],pp,rtol=0,atol=1e-8) and np.array_equal(z['source_neighbor_sog_kn'][sl],ss) and np.array_equal(z['source_neighbor_cog_deg'][sl],cc),'Original neighbor source mismatch')
        frame=source[m];good=np.isfinite(frame[['x','y','sog','cog']].to_numpy()).all(1)&(frame.sog>=0)&(frame.sog<102.2)&(frame.cog>=0)&(frame.cog<360)
        expected=frame.loc[good&(frame.timestamp>=t[0]-300)&(frame.timestamp<=last),'timestamp'].to_numpy(float)
        require(np.array_equal(tt,expected),'Filtered source history contract differs')
        candidates=frame.loc[(frame.timestamp<=anchor)&(frame.timestamp>=anchor-300)]
        require(len(candidates)>0,'No causal anchor report')
        row=candidates.iloc[-1];require(float(row.timestamp)==report['cohort_source_report_time_s'][k],'Not latest causal member report')
        idx=int(frame.index[frame.timestamp==row.timestamp][0]);require(idx>0 and bool(good[idx-1]) and bool(good[idx]),'Unknown anchor kinematics')
        dt=float(row.timestamp-frame.timestamp.iloc[idx-1]);require(0<dt<=300,'Anchor gap')
        prev=frame.iloc[idx-1];reported=.5*(velocity(np.array([prev.sog]),np.array([prev.cog]))[0]+velocity(np.array([row.sog]),np.array([row.cog]))[0])
        e=float(np.linalg.norm((np.array([row.x,row.y])-np.array([prev.x,prev.y]))/dt-reported));w=float(np.exp(-max(0,e-2.5)**2/8))
        target_pos=p[int(report['anchor_step'])];speed=s[int(report['anchor_step'])];course=c[int(report['anchor_step'])]
        require(w>=.3 and row.sog>=3 and abs(row.sog-speed)<=3,'Anchor comparability')
        require(np.linalg.norm(np.array([row.x,row.y])-target_pos)<=4*1852 and abs((row.cog-course+180)%360-180)<=30,'Anchor spatial/directional comparator')
    return onset


def audit_quartets(folder,plan_root,root,registry):
    manifest=read(folder/'relation_manifest.json')
    require(manifest['status']=='passed','Smoke status')
    identities=pd.read_parquet(root/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
    val_ids=set(identities.loc[identities.split=='val','mmsi'].astype(str));test_ids=set(identities.loc[identities.split=='test','mmsi'].astype(str))
    all_ids=set(pd.read_parquet(root/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']).mmsi.astype(str))
    allowed={'train':all_ids-val_ids-test_ids,'val':all_ids-test_ids}
    proofs=[];source_ids={'train':set(),'val':set()};base_ids={'train':set(),'val':set()};maxerr=0.;source_reports=0
    for region,meta in registry.items():
        samples=[x for x in manifest['samples'] if x['region']==region]
        if not samples:continue
        wanted=set(m for x in samples for m in [x['mmsi']]+x['fixed_cohort_mmsi']);source=load_canonical(meta,wanted)
        for report in samples:
            path=folder/report['file'];require(sha(path)==report['sha256'],'Quartet file hash')
            with np.load(path,allow_pickle=False) as zz:z={k:zz[k] for k in zz.files}
            split=str(z['split']);require(split in ('train','val'),'Forbidden split')
            require(str(z['base_window_id'])==report['base_window_id'] and str(z['source_window_id'])==report['window_id'],'Source IDs')
            members=z['cohort_mmsi'].astype(str).tolist();require(members==report['cohort_mmsi'][:8]==report['fixed_cohort_mmsi'],'Fixed past-selected members changed')
            plans=read(plan_root/f'{region}_{split}_relation_feasibility.json')
            plan=next(x for x in plans if x['window_id']==report['window_id']);require(plan['cohort_mmsi'][:8]==members,'Member subset differs from precommitted anchor plan')
            onset=check_source_arrays(z,members,source,meta,report,allowed[split])
            ab=np.array([[0,0],[0,1],[1,0],[1,1]],np.int8)
            require(np.array_equal(z['meta_target_factor'],ab[:,0]) and np.array_equal(z['meta_cohort_factor'],ab[:,1]) and np.array_equal(z['y'],ab[:,0]^ab[:,1]),'Four-cell factor/label design')
            require(np.all(z['target_mask']) and np.all(z['neighbor_mask']),'Unexpected placeholder observations')
            t=z['source_target_t'];p=z['source_target_pos'];s=z['source_target_sog_kn'];c=z['source_target_cog_deg']
            require(np.all(np.diff(t)>0),'Target chronology')
            p1,s1,c1=transformed(t,p,s,c,onset)
            require(np.max(np.abs(s1-s))>=.5,'Insufficient target treatment')
            intended_p=np.stack([p,p,p1,p1]);intended_s=np.stack([s,s,s1,s1]);intended_c=np.stack([c]*4)
            require(np.allclose(z['pre_noise_target_pos'],intended_p,rtol=0,atol=1e-8),'Target displacement integral')
            require(np.array_equal(z['target_sog_kn'],intended_s) and np.array_equal(z['target_cog_deg'],intended_c),'Target speed/course profile')
            maxerr=max(maxerr,float(np.max(np.abs(innovation(t,p1,s1,c1)-innovation(t,p,s,c)))))
            np0=z['source_neighbor_pos'];ns0=z['source_neighbor_sog_kn'];nc0=z['source_neighbor_cog_deg'];nt=z['source_neighbor_t'];offs=z['member_offsets']
            np1=np0.copy();ns1=ns0.copy()
            for k,m in enumerate(members):
                sl=slice(offs[k],offs[k+1]);tt=nt[sl];require(np.all(np.diff(tt)>0) and tt.max()<=t[-1],'Neighbor chronology/future observation')
                pp,ss,cc=transformed(tt,np0[sl],ns0[sl],nc0[sl],onset);np1[sl]=pp;ns1[sl]=ss
                require(np.max(np.abs(ss-ns0[sl]))>=.5 and tt[-1]>=onset+180,'Member acceptance')
                require(np.all(z['neighbor_mmsi'][sl].astype(str)==m),'Member segmentation')
                maxerr=max(maxerr,float(np.max(np.abs(innovation(tt,pp,ss,cc)-innovation(tt,np0[sl],ns0[sl],nc0[sl])))))
            intended_np=np.stack([np0,np1,np0,np1]);intended_ns=np.stack([ns0,ns1,ns0,ns1])
            require(np.allclose(z['pre_noise_neighbor_pos'],intended_np,rtol=0,atol=1e-8),'Neighbor displacement integral')
            require(np.array_equal(z['neighbor_sog_kn'],intended_ns) and np.array_equal(z['neighbor_cog_deg'],np.stack([nc0]*4)),'Neighbor speed/course profile')
            # Replay quartet-shared positional corruption independently.
            tag=int(hashlib.sha256(('quartet-noise:'+report['window_id']).encode()).hexdigest()[:8],16)%4
            require(report['noise']['scope']==['none','target_only','neighbor_only','both'][tag],'Noise assignment depends on factor/label')
            if tag in (1,3):
                ix=min(len(t)-1,int(report['anchor_step'])+2);require(t[ix]>report['anchor_time_s'],'Target noise before anchor')
                intended_p[:,ix,0]+=500
            if tag in (2,3):
                ix=int(offs[0]+min(np.searchsorted(nt[offs[0]:offs[1]],onset+100),offs[1]-offs[0]-1));require(nt[ix]>report['anchor_time_s'],'Neighbor noise before anchor')
                intended_np[:,ix,0]+=500
            require(np.allclose(z['target_pos'],intended_p,rtol=0,atol=1e-8) and np.allclose(z['neighbor_pos'],intended_np,rtol=0,atol=1e-8),'Shared observation corruption differs')
            for key in ('target_t','target_pos','target_sog_kn','target_cog_deg','target_mask'):
                require(np.array_equal(z[key][0],z[key][1]) and np.array_equal(z[key][2],z[key][3]),'Target-only marginal leakage')
            for key in ('neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mask'):
                require(np.array_equal(z[key][0],z[key][2]) and np.array_equal(z[key][1],z[key][3]),'Neighbor-only marginal leakage')
            require(np.array_equal(z['target_t'],np.tile(t,(4,1))) and np.array_equal(z['neighbor_t'],np.tile(nt,(4,1))),'Timestamp treatment changed')
            source_ids[split].add(str(z['target_mmsi']));base_ids[split].add(str(z['base_window_id']));source_reports+=len(t)+len(nt)
            proofs.append(dict(region=region,split=split,window_id=report['window_id'],cohort_size=len(members),source_reports=len(t)+len(nt),fixed_a_target_equal=True,fixed_b_cohort_equal=True,source_canonical_rows_checked=True,all_injected_noise_after_anchor=True,file_sha256=sha(path)))
    require(not(source_ids['train']&source_ids['val']) and not(base_ids['train']&base_ids['val']),'Train/val cluster leakage')
    require(len(proofs)==manifest['source_windows'] and manifest['quartet_instances']==4*len(proofs),'Expanded count')
    require(maxerr<1e-8,'Innovation preservation failed')
    return dict(status='passed',source_windows=len(proofs),quartet_instances=4*len(proofs),source_reports_rechecked=source_reports,
                target_mmsi_counts={k:len(v) for k,v in source_ids.items()},max_independent_innovation_error_mps=maxerr,
                source_selection='Cohort membership verified against the precommitted causal anchor plan and source latest reports. Post-anchor numeric/coverage failure rejects the complete source quartet, not individual members.',
                noise_timing='All added target/cohort teleports occur strictly after the source selection anchor. This verifies a fixed-cohort construction contract, not deployed real-time noisy neighborhood selection.',
                innovation_scope='Retained finite source-history subsequences; no claim that filtering invalid gaps preserves full-source kinematic gate semantics.',
                marginal_scope='Fixed-a target observations are exactly identical across b and fixed-b cohort observations across a, with opposite XOR labels. Factor fields are declared metadata, not model inputs. This is controlled information separation, not a trained anomaly benchmark.',
                manifest_sha256=sha(folder/'relation_manifest.json'),samples=proofs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=Path,required=True);parser.add_argument('--plans',type=Path,required=True);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Refusing overwrite')
    registry=read(args.root/'release_20261026/source_registry.json')['regions']
    manifest=read(args.samples/'relation_manifest.json');attempted=set();actual=set(x['window_id'] for x in manifest['samples']);rejected=set(x['window_id'] for x in manifest['rejects'])
    for region in registry:
        for split in ['train','val']:
            p=args.plans/f'{region}_{split}_relation_feasibility.json'
            if p.exists():attempted.update(x['window_id'] for x in read(p) if x['accepted'])
    require(not actual & rejected and actual | rejected==attempted,'All anchor candidates must be retained or explicitly rejected')
    result=audit_quartets(args.samples,args.plans,args.root,registry)
    result.update(accepted_sources=len(actual),rejected_sources=len(rejected),attempted_sources=len(attempted),all_candidates_accounted=True,auditor_sha256=sha(Path(__file__)))
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status='passed',sources=result['source_windows'],source_reports=result['source_reports_rechecked'],accepted=len(actual),rejected=len(rejected))))

if __name__=='__main__':main()
