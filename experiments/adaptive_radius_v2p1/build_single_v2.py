"""Build label-blind field features from recorded causal observation candidates."""
from pathlib import Path
import argparse,gzip,hashlib,importlib.util,json,sys,time
from collections import Counter
import numpy as np
from radius_core import choose,ARMS,NM

ROOT=Path('/mnt/sda/WangZijia/AIS/outputs')
BASE=ROOT/'framework_validation_v1_20261028'
BAL=ROOT/'balanced_global_v2'
DEFAULT_OUT=Path('/home/ps/AIS_adaptive_radius_relation_v2_20261109')
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o);return o

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def extract(split,regions,out,limit=0):
    assert split in ('train','val','test')
    protocol=Path(__file__).with_name('protocol.json');code=Path(__file__).with_name('radius_core.py')
    protocol_sha=sha(protocol);rootout=out/('smoke' if limit else 'features');rootout.mkdir(parents=True,exist_ok=True)
    source_registry=json.loads((BAL/'release_20261026/source_registry.json').read_text())['regions']
    helper=module(BAL/'context_adapter_v1_20261026/neighborhood/audit_neighborhood.py','radius_source_audit')
    core_path=BASE/'data/individual'/f'{split}.npz';field_path=BASE/'extra_data'/f'field_{split}.npz'
    core=load_npz(core_path);extra=load_npz(field_path)
    for k in ('window_id','mmsi','mask','time_s'):
        if not np.array_equal(core[k],extra[k]):raise ValueError('unaligned source '+k)
    for region in regions:
        t0=time.time();dest=rootout/f'{split}_{region}.npz';meta_dest=rootout/f'{split}_{region}.json'
        if dest.exists() or meta_dest.exists():raise FileExistsError(dest)
        src_path=BAL/'labeling/labeled_v1/features_scene'/region/f'{split}.npz'
        audit_path=src_path.with_name(split+'_context_audit.jsonl.gz')
        meta=source_registry[region];src=helper.Source(meta)
        obs=load_npz(src_path);ind=np.flatnonzero(core['region'].astype(str)==region)
        for k in ('window_id','mmsi','mask','time_s'):
            if not np.array_equal(core[k][ind].astype(str) if k=='mmsi' else core[k][ind],obs[k].astype(str) if k=='mmsi' else obs[k]):raise ValueError('region source mismatch '+region+' '+k)
        if limit:ind=ind[:limit]
        n=len(ind);shape=(n,20);N=np.zeros(shape,np.int16)
        arrays={'window_id':core['window_id'][ind], 'mmsi':core['mmsi'][ind], 'region':core['region'][ind], 'y':core['y'][ind],
                'mask':core['mask'][ind], 'time_s':core['time_s'][ind], 'behavior_type':core['behavior_type'][ind],
                'observation_noise':core['observation_noise'][ind], 'target_noise_effective':core['target_noise_effective'][ind],
                'neighbor_noise_effective':core['neighbor_noise_effective'][ind], 'target':extra['gated7_legacy'][ind],
                'raw':core['base__C0'][ind,:,:7], 'eligible_count_8nm':N}
        for arm in ARMS:
            arrays[arm+'__field']=np.zeros((*shape,4),np.float32)
            arrays[arm+'__bg']=np.zeros((*shape,2),np.float32)
            arrays[arm+'__rho']=np.zeros(shape,np.float32)
            arrays[arm+'__mode']=np.zeros(shape,np.uint8)
            arrays[arm+'__radius_nm']=np.zeros(shape,np.float32)
            arrays[arm+'__count']=np.zeros(shape,np.int16)
        counts=Counter();seen=0
        with gzip.open(audit_path,'rt',encoding='utf8') as f:
          for local,line in enumerate(f):
            if local>=n:break
            md=json.loads(line);g=int(ind[local]);wid=str(core['window_id'][g]);target=str(core['mmsi'][g])
            if md['window_id']!=wid or str(md['target_mmsi'])!=target:raise ValueError('audit row mismatch')
            overlay=src.overlay(md);box=md.get('known_coverage_bbox_m')
            for step in md['steps']:
                j=int(step['step'])
                if not core['mask'][g,j]:raise ValueError('audit contains padded step')
                now=float(core['time_s'][g,j]);pp=np.asarray(obs['target_pos'][local,j],float)
                if abs(now-float(step['time_s']))>1e-6:raise ValueError('target timestamp mismatch')
                positions=[];velocities=[];weights=[];valid=[];ids=[]
                for entry in step.get('neighbors',[]):
                    if not entry['plain_accepted']:continue
                    mid=str(entry['mmsi']);stamp=float(entry['time_s'])
                    if mid==target or not 0<=now-stamp<=300:raise ValueError('invalid candidate identity/time')
                    pos,speed,cog,kin,error,w=src.observation(mid,stamp,overlay)
                    pos=np.asarray(pos,float);a=np.deg2rad(cog);vel=np.array([np.sin(a),np.cos(a)])*speed*(NM/3600.)
                    if not (np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(w)):raise ValueError('nonfinite observation')
                    distance=float(np.linalg.norm(pos-pp))
                    if abs(distance-float(entry['distance_m']))>1e-5 or distance>8*NM+1e-5:raise ValueError('source distance mismatch')
                    positions.append(pos-pp);velocities.append(vel);weights.append(float(w));valid.append(bool(kin));ids.append(mid)
                coverage=lambda radius:helper.coverage_ok(pp,box,radius*NM)
                result,diagnosis=choose(positions,velocities,velocities,weights,valid,ids,target,coverage)
                arrays['eligible_count_8nm'][local,j]=diagnosis['plain_count_8nm'];counts['target_steps']+=1
                if box is None:counts['unknown_roi_steps']+=1
                for arm,item in result.items():
                    arrays[arm+'__rho'][local,j]=item['rho'];arrays[arm+'__mode'][local,j]=item['mode'];arrays[arm+'__radius_nm'][local,j]=item['radius_nm'];arrays[arm+'__count'][local,j]=item['count']
                    counts[arm+'_mode_'+str(item['mode'])]+=1
                    if item['mode']:
                        bg=np.asarray(item['bg']);grad=np.asarray(item['grad']);tv=arrays['target'][local,j,2:4]
                        arrays[arm+'__bg'][local,j]=bg
                        arrays[arm+'__field'][local,j]=[*(tv-bg),np.trace(grad),grad[1,0]-grad[0,1]]
            seen+=1
            if seen%100==0:print('WINDOWS',split,region,seen,'/',n,flush=True)
        if seen!=n:raise ValueError('missing audit rows')
        for arm in ARMS:
            mode=arrays[arm+'__mode'];bg=arrays[arm+'__bg'];speed=np.linalg.norm(bg,axis=-1);rate=np.zeros_like(speed)
            ts=arrays['time_s'];dt=np.diff(ts,axis=1);ok=(mode[:,1:]>0)&(mode[:,:-1]>0)&(dt>0)
            np.divide(np.abs(np.diff(speed,axis=1)),dt,out=rate[:,1:],where=ok)
            ff=arrays[arm+'__field'];arrays[arm+'__state']=np.stack((speed,rate,ff[:,:,2],ff[:,:,3]),axis=-1).astype(np.float32)
            if not all(np.isfinite(arrays[arm+k]).all() for k in ('__field','__bg','__rho','__state')):raise ValueError('nonfinite output '+arm)
        if sha(protocol)!=protocol_sha:raise ValueError('protocol changed during run')
        np.savez_compressed(dest,**arrays)
        report={'split':split,'region':region,'windows':n,'counts':dict(counts),'out_sha256':sha(dest),'protocol_sha256':protocol_sha,'operator_sha256':sha(code),'builder_sha256':sha(Path(__file__)),'input_sha256':{'core':sha(core_path),'field':sha(field_path),'audit':sha(audit_path),'region_features':sha(src_path),'canonical':meta['sha256']},'elapsed_seconds':time.time()-t0,'candidate_pool':'recorded observed baseline eligible members <=8nm; no outside-candidate reselection'}
        meta_dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print('COMPLETE',split,region,n,'seconds',round(time.time()-t0,1),dict(counts),flush=True)
        del src,obs

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['train','val','test']);p.add_argument('--regions',nargs='+',default=REGIONS);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if any(x not in REGIONS for x in a.regions):raise ValueError('unknown region')
    extract(a.split,a.regions,a.out,a.limit)
if __name__=='__main__':main()
