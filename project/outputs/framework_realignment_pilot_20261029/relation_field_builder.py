"""Development-only relation field inputs. Never reads scores or clean truth."""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np

BASE=Path('/mnt/sda/WangZijia/AIS/outputs')
ROOT=BASE/'balanced_global_v2/relation_dual_pressure_v2_20261027'
STAGE=BASE/'framework_validation_v1_20261028'
OUT=BASE/'framework_realignment_pilot_20261029'
def module(file,name):
    s=importlib.util.spec_from_file_location(name,file); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
core=module(STAGE/'data_builder.py','prior_data')
field=module(STAGE/'field_builder.py','prior_field')
gate=module(BASE/'balanced_global_v2/causal_gate_main_v1_20261027/gate_builder.py','prior_gate')
def velocity(s,c):
    a=np.deg2rad(c); return np.column_stack([s*np.sin(a),s*np.cos(a)])*0.514444444444
def direction(window,member):
    digest=hashlib.sha256(('framework-stress-position-v1:'+window+':'+member).encode()).digest()
    theta=2*np.pi*int.from_bytes(digest[:8],'big')/2**64
    return np.array([np.cos(theta),np.sin(theta)])
def calc(o,case,target_id):
    t=o['target_t'][case]; p=o['target_pos'][case]; targetv=velocity(o['target_sog_kn'][case],o['target_cog_deg'][case]); n=len(t)
    obs=[]
    for member,lo,hi in zip(o['cohort_mmsi'],o['member_offsets'][:-1],o['member_offsets'][1:]):
        lo,hi=int(lo),int(hi); nt=o['neighbor_t'][case,lo:hi]; ns=o['neighbor_sog_kn'][case,lo:hi]; nc=o['neighbor_cog_deg'][case,lo:hi]; np_=o['neighbor_pos'][case,lo:hi]
        w,known=core.absolute_weights(nt,np_,ns,nc); ix=np.searchsorted(nt,t,side='right')-1
        obs.append((str(member),nt,np_,velocity(ns,nc),w,ix))
    result=np.zeros((2,n,7),np.float32); count=np.zeros(n,np.int16)
    for j in range(n):
        records=[]
        for member,nt,pos,vel,w,ix in obs:
            q=int(ix[j]);
            if q>=0 and 0<=t[j]-nt[q]<=300: records.append((member,pos[q]-p[j],vel[q],w[q]))
        count[j]=len(records)
        if len(records)<4: continue
        ids=[x[0] for x in records]; pos=np.array([x[1] for x in records]); v=np.array([x[2] for x in records]); w=np.array([x[3] for x in records]); dist=np.exp(-.5*np.sum(pos**2,axis=1)/field.SIGMA**2)
        fits=[field.fit_field(pos,v,dist,ids,target_id),field.fit_field(pos,v,dist*w,ids,target_id)]
        if not all(f['valid'] for f in fits): continue
        for arm,fit in enumerate(fits):
            b=fit['coefficient']; grad=b[1:,:].T/field.R; bg=b[0]; rho=min(1,len(records)/4)*float(np.mean(w)); result[arm,j]=[*bg,*(targetv[j]-bg),grad[0,0]+grad[1,1],grad[1,0]-grad[0,1],rho]
    return result,count
def main():
    a=argparse.ArgumentParser();a.add_argument('--split',choices=['train','val','test'],required=True);a.add_argument('--limit',type=int,default=0); opt=a.parse_args()
    manifest=json.loads((ROOT/'dataset_manifest_combined.json').read_text()); rows=sorted([r for r in manifest['samples'] if r['version']=='relaxed' and r['split']==opt.split],key=lambda r:(r['region'],r['window_id']))
    if opt.limit: rows=rows[:opt.limit]
    output={'source':[],'case':[],'y':[],'length':[],'mmsi':[],'region':[],'raw7':[],'causal7':[],'self_w':[],'time_s':[]}
    views=('base','natural','suffix','combined_l100_m600')
    for view in views: output[view+'__field']=[]; output[view+'__count']=[]
    for row in rows:
        data={}
        for view in ('base','natural','suffix'):
            entry=row['files'][view]; path=ROOT/entry['path']; assert field.sha(path)==entry['sha256']
            keys=['target_t','target_pos','target_sog_kn','target_cog_deg','neighbor_t','neighbor_pos','neighbor_sog_kn','neighbor_cog_deg','neighbor_mmsi','member_offsets','cohort_mmsi','y']
            data[view]=field.read_npz(path,keys)
        for key in keys:
            if key!='neighbor_sog_kn': np.testing.assert_array_equal(data['base'][key],data['suffix'][key])
        combined={k:v.copy() for k,v in data['base'].items()}; combined['neighbor_sog_kn']=data['suffix']['neighbor_sog_kn'].copy()
        for m in row['stress']['members']:
            lo=int(combined['member_offsets'][m['member_index']]); hi=int(combined['member_offsets'][m['member_index']+1]); ix=lo+int(m['indices']['suffix']['report_index']); assert lo<=ix<ix+1<hi
            for q in [ix,ix+1]: combined['neighbor_pos'][:,q]+=600*direction(row['window_id'],m['mmsi'])
        data['combined_l100_m600']=combined
        for case in range(4):
            o=data['base']; n=len(o['target_t'][case]); assert n<=20 and str(row['mmsi']) not in o['cohort_mmsi'].astype(str)
            output['source'].append(row['window_id']);output['case'].append(case);output['y'].append(int(o['y'][case]));output['length'].append(n);output['mmsi'].append(str(row['mmsi']));output['region'].append(row['region'])
            t=o['target_t'][case]; p=o['target_pos'][case]; speed=o['target_sog_kn'][case]; course=o['target_cog_deg'][case]; v=velocity(speed,course); theta=np.deg2rad(course)
            raw=np.column_stack([p-p[0],v,speed,np.sin(theta),np.cos(theta)]).astype(np.float32); w,valid=core.absolute_weights(t,p,speed,course)
            causal,_=gate.causal(raw,w,np.ones(n,bool),valid,t,13.73566722869873)
            for j in range(1,n):
                if w[j]<.2: causal[j,2:7]=causal[j-1,2:7]
            for name,x in [('raw7',raw),('causal7',causal),('self_w',w),('time_s',t)]:
                z=np.zeros((20,)+x.shape[1:],dtype=x.dtype);z[:n]=x;output[name].append(z)
            for view in views:
                v=data[view]; assert np.array_equal(v['target_t'][case],o['target_t'][case])
                values,count=calc(v,case,row['mmsi']); pad=np.zeros((2,20,7),np.float32);pad[:,:n]=values; cp=np.zeros(20,np.int16);cp[:n]=count
                output[view+'__field'].append(pad);output[view+'__count'].append(cp)
        if (len(output['source'])//4)%100==0: print('PROGRESS',opt.split,len(output['source'])//4,'/',len(rows),flush=True)
    arrays={k:np.asarray(v) for k,v in output.items()}; name=opt.split+('_smoke'+str(opt.limit) if opt.limit else '')+'_relation_field.npz'; dest=OUT/name
    assert not dest.exists();np.savez_compressed(dest,**arrays)
    audit={'file':str(dest),'file_sha256':field.sha(dest),'builder_sha256':field.sha(Path(__file__)),'source_manifest_sha256':field.sha(ROOT/'dataset_manifest_combined.json'),'sources':len(rows),'cases':len(arrays['y']),'valid_steps':{v:int((arrays[v+'__field'][:,0,:,6]>0).sum()) for v in views},'total_valid_target_steps':int(arrays['length'].sum()),'scope':'development, no model training','cohort':'existing offline fixed cohort, common support intersection of F0 and F1; >=4 aligned reports and acceptable geometry; no reselection by distance or course after stress','rho':'shared min(1, n/4) multiplied by mean absolute reliability of aligned members; zero when fit invalid','stress':'combined suffix speed with deterministic 600m offset at same suffix report and next report for selected members','field_channels':['background_vx','background_vy','raw_velocity_residual_x','raw_velocity_residual_y','divergence','curl','rho'],'units':'positions m, velocity m/s, derivatives 1/s, time seconds','train_cap_mps':13.73566722869873}
    with dest.with_suffix('.json').open('x',encoding='utf8') as f: json.dump(audit,f,indent=2)
    print(json.dumps(audit))
if __name__=='__main__':main()
