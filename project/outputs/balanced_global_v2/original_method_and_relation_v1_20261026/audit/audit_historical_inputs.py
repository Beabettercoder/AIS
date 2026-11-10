"""Independent physical-schema and graph-history audit for legacy transfer."""
from pathlib import Path
import argparse,gzip,hashlib,json
import numpy as np
import pandas as pd

KNOT=1852/3600
REGIONS=('houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai')
CLASS={'none':0,'turn_back':1,'deviation':2,'stall':3,'sinuous':4,'speed_drop':5}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(ok,msg):
    if not ok:raise AssertionError(msg)
def same(actual,expected,name):
    check(actual.shape==expected.shape,name+' shape')
    if actual.dtype.kind in 'bUi':check(np.array_equal(actual,expected),name)
    else:check(np.allclose(actual,expected,rtol=2e-6,atol=3e-6),name)
def velocity(s,c):
    a=np.deg2rad(c);return np.asarray(s)[...,None]*KNOT*np.stack([np.sin(a),np.cos(a)],-1)
def good(p,s,c):return np.isfinite(p).all(-1)&np.isfinite(s)&np.isfinite(c)&(s>=0)&(s<102.2)&(c>=0)&(c<360)

class Canonical:
    def __init__(self,meta):
        self.path=Path(meta['canonical_file']);check(sha(self.path)==meta['sha256'],'canonical hash')
        d=pd.read_parquet(self.path,columns=['mmsi','timestamp','lon','lat','sog','cog'])
        d['identity']=d.mmsi.astype(str);d['input_order']=np.arange(len(d))
        d=d.sort_values(['identity','timestamp','input_order'],kind='stable').drop_duplicates(['identity','timestamp'],keep='last').reset_index(drop=True)
        self.ids=d.identity.to_numpy();self.t=d.timestamp.to_numpy(float);self.s=d.sog.to_numpy(float);self.c=d.cog.to_numpy(float)
        box=meta['core_bbox'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        self.p=np.column_stack([(d.lon.to_numpy()-cx)*111195*np.cos(np.deg2rad(cy)),(d.lat.to_numpy()-cy)*111195])
        self.ships={k:v.index.to_numpy() for k,v in d.groupby('identity',sort=False)}
        self.lookup={(m,float(t)):i for i,(m,t) in enumerate(zip(self.ids,self.t))}
    def noise(self,meta):
        out={};decision=meta['decision_time_s']
        for change in meta['neighbor_observation_noise']['changes']:
            if not change.get('effective'):continue
            candidates=self.ships[str(change['mmsi'])]
            valid=good(self.p[candidates],self.s[candidates],self.c[candidates])
            take=candidates[valid & (self.t[candidates]>=decision-1980)&(self.t[candidates]<=decision)]
            times=self.t[take];pos=self.p[take].copy();speed=self.s[take].copy();course=self.c[take].copy();params=change['parameters']
            if change['noise']=='teleport':
                at=int(params['report_index']);check(times[at]==params['time_s'],'noise time')
                ang=np.deg2rad(params['angle_deg']);pos[at]+=params['offset_m']*np.array([np.sin(ang),np.cos(ang)])
            elif change['noise']=='speed_spike':
                at=int(params['report_index']);check(times[at]==params['time_s'],'noise time')
                speed[at:at+int(params['reports'])]=np.minimum(speed[at:at+int(params['reports'])]*params['factor'],102.1)
            elif change['noise']=='drift':
                u=(times-params['onset_s'])/params['duration_s'];env=np.where((u>=0)&(u<=1),.5-.5*np.cos(2*np.pi*np.clip(u,0,1)),0)
                ang=np.deg2rad(params['angle_deg']);pos+=params['offset_m']*env[:,None]*np.array([np.sin(ang),np.cos(ang)])
            else:raise ValueError(change['noise'])
            changed=np.any(pos!=self.p[take],1)|(speed!=self.s[take])|(course!=self.c[take])
            check(times[changed].tolist()==change['affected_time_s'],'changed source times')
            for k,ix in enumerate(take):out[int(ix)]=(pos[k],speed[k],course[k])
        return out
    def obs(self,index,overlay):return overlay.get(int(index),(self.p[index],self.s[index],self.c[index]))
    def weight(self,index,overlay):
        members=self.ships[self.ids[index]];at=int(np.searchsorted(members,index))
        if at==0:return 0.
        previous=int(members[at-1]);dt=float(self.t[index]-self.t[previous])
        p,s,c=self.obs(index,overlay);q,v,b=self.obs(previous,overlay)
        if not 0<dt<=300 or not bool(good(np.asarray([p,q]),np.asarray([s,v]),np.asarray([c,b])).all()):return 0.
        error=float(np.linalg.norm((p-q)/dt-(velocity(s,c)+velocity(v,b))/2))
        return float(np.exp(-max(0,error-2.5)**2/8))

def expected_target(parent):
    mask=parent['mask'];N,T=mask.shape;raw=np.zeros((N,T,7),np.float32)
    raw[:,:,:2]=(parent['target_pos']-parent['target_pos'][:,0:1,:]).astype(np.float32)
    raw[:,:,2:4]=velocity(parent['target_sog_kn'],parent['target_cog_deg'])
    raw[:,:,4]=parent['target_sog_kn'];raw[:,:,5]=np.sin(np.deg2rad(parent['target_cog_deg']));raw[:,:,6]=np.cos(np.deg2rad(parent['target_cog_deg']));raw[~mask]=0
    w=parent['self_gate_weight'].astype(np.float32).copy();w[~mask]=0
    rf=np.linalg.norm(parent['residual_mps'],axis=-1).astype(np.float32);rf[~mask]=0
    def gated(masked):
        x=raw.copy()
        for i in range(N):
            length=int(mask[i].sum()) if masked else T;running=np.zeros(2)
            for j in range(length):
                previous=raw[i,j-1,:2].astype(float) if j else np.zeros(2)
                running+=(raw[i,j,:2].astype(float)-previous)*float(w[i,j]);x[i,j,:2]=running
                if j and w[i,j]<.2:x[i,j,2:7]=x[i,j-1,2:7]
            if masked:x[i,length:]=0
        return x
    compat=gated(False);masked=gated(True);ep=np.zeros((N,T,4),np.float32)
    ep[:,:,0]=np.linalg.norm(parent['background_field_mps'],axis=-1);ep[:,:,2]=parent['divergence_per_s'];ep[:,:,3]=parent['curl_per_s']
    for t in range(1,T):ep[:,t,1]=abs(ep[:,t,0]-ep[:,t-1,0])/np.maximum(parent['time_s'][:,t]-parent['time_s'][:,t-1],1)
    ep[~mask]=0
    return {'raw_legacy7':raw,'raw_incremental7':parent['raw'],'w_self':w,'rf':rf,'ep':ep,
            'kg_compat8':np.concatenate([compat,w[:,:,None]],2),'kglf_compat9':np.concatenate([compat,rf[:,:,None],w[:,:,None]],2),
            'kglf_masked9':np.concatenate([masked,rf[:,:,None],w[:,:,None]],2),
            'label_class':np.array([CLASS[str(v)] for v in parent['behavior_type']],np.int64)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--regions',nargs='+',default=list(REGIONS));args=ap.parse_args()
    check(not args.out.exists(),'Refusing overwrite');reg=json.loads((args.root/'release_20261026/source_registry.json').read_text())['regions']
    allids=set(pd.read_parquet(args.root/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']).mmsi.astype(str))
    membership=pd.read_parquet(args.root/'release_20261026/selected_windows.parquet',columns=['mmsi','split']);v=set(membership.loc[membership.split=='val','mmsi'].astype(str));t=set(membership.loc[membership.split=='test','mmsi'].astype(str));allowed={'train':allids-v-t,'val':allids-t}
    records=[];counts={'windows':0,'neighbors':0,'history_reports':0,'invalid_history_zeroed':0};identities={'train':set(),'val':set()}
    for region in args.regions:
        source=Canonical(reg[region])
        for split in ('train','val'):
            original=args.root/'labeling/labeled_v1/features_scene'/region/f'{split}.npz';actual=args.data/region/f'{split}.npz'
            with np.load(original,allow_pickle=False) as z:parent={k:z[k] for k in z.files}
            with np.load(actual,allow_pickle=False) as z:values={k:z[k] for k in z.files}
            for name,expected in expected_target(parent).items():same(values[name],expected,region+'/'+split+'/'+name)
            for name in ('y','mask','window_id','base_window_id','mmsi','time_s','observation_noise','behavior_type','region'):check(np.array_equal(values[name],parent[name]),'metadata '+name)
            identities[split].update(values['mmsi'].astype(str));N=len(values['y']);total_history=0;total_neighbors=0
            upstream=original.with_name(f'{split}_context_audit.jsonl.gz');trace=args.data/region/f'{split}_graph_provenance.jsonl.gz'
            with gzip.open(upstream,'rt',encoding='utf-8') as f,gzip.open(trace,'rt',encoding='utf-8') as g:
                for j,(line,annotation) in enumerate(zip(f,g)):
                    meta=json.loads(line);proof=json.loads(annotation);length=int(values['mask'][j].sum());last=length-1;decision=float(values['time_s'][j,last]);mid=str(values['mmsi'][j])
                    check(meta['window_id']==str(values['window_id'][j])==proof['window_id'],'trace identity')
                    step=next(x for x in meta['steps'] if x['step']==last);pool=[x for x in step.get('neighbors',[]) if x['plain_accepted']]
                    chosen=sorted(pool,key=lambda x:(float(x['distance_m']),str(x['mmsi'])))[:32] if len(pool)>=4 and values['boundary_context_valid'][j,last] else []
                    check(len(chosen)==int(values['nb_mask'][j].sum())==len(proof['selected']),'neighbor count')
                    check(np.array_equal(values['nb_mask'][j],np.arange(32)<len(chosen)),'neighbor padding')
                    overlay=source.noise(meta) if chosen else {}
                    for k,entry in enumerate(chosen):
                        member=str(entry['mmsi']);stamp=float(entry['time_s']);check(member!=mid and member in allowed[split],'identity exclusion')
                        check(member==str(values['nb_mmsi'][j,k])==proof['selected'][k]['mmsi'],'selected identities')
                        check(stamp==values['nb_time'][j,k] and 0<=decision-stamp<=300,'causal current time')
                        ix=source.lookup[member,stamp];pos,sp,cog=source.obs(ix,overlay);same(values['nb_xy'][j,k],(pos-parent['target_pos'][j,last]).astype(np.float32),'neighbor xy')
                        same(values['nb_v'][j,k],velocity(sp,cog).astype(np.float32),'neighbor velocity')
                        check(abs(float(values['nb_w'][j,k])-float(entry['reliability']))<1e-7,'neighbor recorded weight')
                        check(abs(float(values['nb_w'][j,k])-source.weight(ix,overlay))<2e-7,'neighbor independent weight')
                        ship=source.ships[member];eligible=ship[source.t[ship]<=decision];history=eligible[-5:];check(source.t[history[-1]]==stamp,'latest identity report')
                        expected_times=np.zeros(5);expected_mask=np.zeros(5,bool);expected_seq=np.zeros((5,4),np.float32);pad=5-len(history);bad=0
                        expected_times[pad:]=source.t[history]
                        for index,hist in enumerate(history,pad):
                            hp,hs,hc=source.obs(hist,overlay);valid=bool(good(np.asarray([hp]),np.asarray([hs]),np.asarray([hc]))[0]);expected_mask[index]=valid
                            if valid:expected_seq[index]=np.r_[hp-pos,velocity(hs,hc)]
                            else:bad+=1
                        same(values['nb_hist_time'][j,k],expected_times,'history timestamps');same(values['nb_hist_mask'][j,k],expected_mask,'history mask');same(values['nb_seq'][j,k],expected_seq,'history values')
                        check(proof['selected'][k]['history_time_s']==source.t[history].tolist(),'provenance times')
                        check(proof['selected'][k]['invalid_history_rows_zeroed']==bad,'provenance invalid')
                        counts['invalid_history_zeroed']+=bad;total_history+=len(history);total_neighbors+=1
                    for name in ('nb_seq','nb_xy','nb_v','nb_w','nb_time','nb_hist_time','nb_hist_mask'):check(not np.any(values[name][j,len(chosen):]),'inactive neighbor slots')
                check(j+1==N,'trace length')
            counts['windows']+=N;counts['neighbors']+=total_neighbors;counts['history_reports']+=total_history
            records.append(dict(region=region,split=split,windows=N,clean_gradient_windows=int((~values['observation_noise']).sum()),source_sha256=sha(original),output_sha256=sha(actual),provenance_sha256=sha(trace),neighbor_reports=total_neighbors,history_reports=total_history))
            print('INPUT_AUDITED',region,split,N,total_neighbors,total_history,flush=True)
    check(not identities['train']&identities['val'],'target MMSI overlap')
    args.out.mkdir(parents=True)
    report=dict(status='passed',scope='All target schema arrays and all chosen graph histories independently reconstructed from parent physical arrays/canonical observations; no builder imported and no test samples read.',counts=counts,files=records,train_val_target_mmsi_overlap=0,canonical_sources=[dict(region=x,path=reg[x]['canonical_file'],sha256=reg[x]['sha256']) for x in args.regions])
    (args.out/'independent_historical_input_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed',**counts}),flush=True)

if __name__=='__main__':main()
