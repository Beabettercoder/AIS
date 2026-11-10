"""Small source-grounded four-cell consistency task; not a training benchmark."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from audit_neighborhood import Source,REGIONS,vel,save,sha

AB=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8)

def innovation(t,p,s,c):return np.diff(p,axis=0)/np.diff(t)[:,None]-(vel(s,c)[1:]+vel(s,c)[:-1])/2
def slowdown(t,p,s,c,onset):
 u=np.clip((t-onset)/300.,0,1);factor=1-.3*(3*u*u-2*u*u*u);s2=s*factor
 dv=vel(s2,c)-vel(s,c);offset=np.zeros_like(p);offset[1:]=np.cumsum((dv[1:]+dv[:-1])/2*np.diff(t)[:,None],axis=0)
 p2=p+offset;assert np.max(np.abs(innovation(t,p2,s2,c)-innovation(t,p,s,c)))<1e-8
 return p2,s2,c.copy()
def signature(arrays):
 h=hashlib.sha256()
 for x in arrays:h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
 return h.hexdigest()

def build_one(src,row,allowed):
 t=np.asarray(row['source_timestamps'],float);p,s,c=src.source_target(row['mmsi'],t);anchor=row['anchor_step'];onset=row['anchor_time_s']+1.
 # Membership is finalized from the saved causal anchor before any future
 # acceptance checks. A failed member rejects the entire source quartet.
 members=row['cohort_mmsi'][:8];assert len(members)>=4 and row['mmsi'] not in members and all(m in allowed for m in members)
 nt=[];nps=[];nss=[];ncs=[];nids=[];offsets=[0]
 for m in members:
  ii=src.history(m,t[0]-300,t[-1]);ii=ii[src.good[ii]]
  if len(ii)<3 or src.t[ii][-1]<onset+180:return None,'cohort_insufficient_observed_future'
  tt=src.t[ii];pp=src.p[ii];ss=src.s[ii];cc=src.c[ii];pa,sa,ca=slowdown(tt,pp,ss,cc,onset)
  if np.max(np.abs(sa-ss))<.5:return None,'cohort_no_sufficient_numeric_change'
  nt.extend(tt);nps.extend(pp);nss.extend(ss);ncs.extend(cc);nids.extend([m]*len(ii));offsets.append(len(nt))
 p1,s1,c1=slowdown(t,p,s,c,onset)
 if np.max(np.abs(s1-s))<.5:return None,'target_no_sufficient_numeric_change'
 nt=np.asarray(nt,float);np0=np.asarray(nps,float);ns0=np.asarray(nss,float);nc0=np.asarray(ncs,float);nids=np.asarray(nids,dtype=str);offsets=np.asarray(offsets,int)
 np1=np0.copy();ns1=ns0.copy();nc1=nc0.copy();maxerr=0.
 for i,m in enumerate(members):
  sl=slice(offsets[i],offsets[i+1]);pp,ss,cc=slowdown(nt[sl],np0[sl],ns0[sl],nc0[sl],onset)
  np1[sl]=pp;ns1[sl]=ss;nc1[sl]=cc;maxerr=max(maxerr,float(np.max(np.abs(innovation(nt[sl],pp,ss,cc)-innovation(nt[sl],np0[sl],ns0[sl],nc0[sl])))))
 tp=np.stack([p if a==0 else p1 for a,b in AB]);ts=np.stack([s if a==0 else s1 for a,b in AB]);tc=np.stack([c if a==0 else c1 for a,b in AB])
 np4=np.stack([np0 if b==0 else np1 for a,b in AB]);ns4=np.stack([ns0 if b==0 else ns1 for a,b in AB]);nc4=np.stack([nc0 if b==0 else nc1 for a,b in AB])
 pre_tp=tp.copy();pre_np=np4.copy()
 # One quartet-shared positional operator, chosen from source identity only.
 tag=int(hashlib.sha256(('quartet-noise:'+row['window_id']).encode()).hexdigest()[:8],16)%4
 noise={'kind':'none' if tag==0 else 'teleport','scope':['none','target_only','neighbor_only','both'][tag],'offset_m':[500.,0.]}
 if tag in (1,3):
  ix=min(len(t)-1,anchor+2);tp[:,ix,0]+=500;noise['target_index']=ix;noise['target_time_s']=float(t[ix])
 if tag in (2,3):
  start,end=offsets[0],offsets[1];local=int(np.searchsorted(nt[start:end],onset+100));ix=int(start+min(local,end-start-1));np4[:,ix,0]+=500;noise['neighbor_flat_index']=ix;noise['neighbor_mmsi']=members[0];noise['neighbor_time_s']=float(nt[ix])
 # Exact input marginal balance: each target-only and each neighbor-only
 # observation signature occurs once per class, independent of factor state.
 y=np.bitwise_xor(AB[:,0],AB[:,1]);target_signatures=[];neighbor_signatures=[]
 for i in range(4):
  target_signatures.append(signature([t,tp[i],ts[i],tc[i]]));neighbor_signatures.append(signature([nt,nids,np4[i],ns4[i],nc4[i]]))
 for hashes in [target_signatures,neighbor_signatures]:
  assert len(set(hashes))==2
  for h in set(hashes):assert sorted(y[np.array(hashes)==h].tolist())==[0,1]
 assert np.array_equal(tp[0],tp[1]) and np.array_equal(tp[2],tp[3])
 assert np.array_equal(np4[0],np4[2]) and np.array_equal(np4[1],np4[3])
 assert max(row['cohort_report_time_s'])<=row['anchor_time_s']<onset
 targeterr=float(np.max(np.abs(innovation(t,p1,s1,c1)-innovation(t,p,s,c))))
 arrays=dict(source_target_t=t,source_target_pos=p,source_target_sog_kn=s,source_target_cog_deg=c,
             source_neighbor_t=nt,source_neighbor_pos=np0,source_neighbor_sog_kn=ns0,source_neighbor_cog_deg=nc0,
             target_t=np.tile(t,(4,1)),target_pos=tp,target_sog_kn=ts,target_cog_deg=tc,target_mask=np.ones(ts.shape,bool),
             pre_noise_target_pos=pre_tp,neighbor_t=np.tile(nt,(4,1)),neighbor_pos=np4,neighbor_sog_kn=ns4,neighbor_cog_deg=nc4,
             pre_noise_neighbor_pos=pre_np,neighbor_mask=np.ones(ns4.shape,bool),neighbor_mmsi=nids,member_offsets=offsets,
             cohort_mmsi=np.asarray(members),target_mmsi=np.asarray(row['mmsi']),base_window_id=np.asarray(row['base_window_id']),
             source_window_id=np.asarray(row['window_id']),split=np.asarray(row['split']),region=np.asarray(row['region']),
             meta_target_factor=AB[:,0],meta_cohort_factor=AB[:,1],y=y)
 report={**row,'onset_s':onset,'slowdown_factor':.7,'duration_s':300.,'fixed_cohort_mmsi':members,
         'cohort_source_report_time_s':row['cohort_report_time_s'][:len(members)],'noise':noise,
         'target_signatures':target_signatures,'neighbor_signatures':neighbor_signatures,
         'max_innovation_error_mps':max(maxerr,targeterr),'siblings':4,'independent_sources':1,
         'factor_metadata_not_model_inputs':True,'all_four_siblings_same_split':True}
 return (arrays,report),None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--scan',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(exist_ok=False)
 registry=json.loads((args.root/'release_20261026/source_registry.json').read_text())['regions']
 g=pd.read_parquet(args.root/'release_20261026/global_all_mmsi_split.parquet',columns=['mmsi']);targets=pd.read_parquet(args.root/'release_20261026/selected_windows.parquet',columns=['mmsi','split'])
 tv=set(targets.loc[targets.split=='val','mmsi'].astype(str));tt=set(targets.loc[targets.split=='test','mmsi'].astype(str));allids=set(g.mmsi.astype(str));allowed={'train':allids-tv-tt,'val':allids-tt}
 result={'status':'running','scope':'Small four-cell construction acceptance only; no trained model or test data','protocol':'neighborhood_relation_protocol.json','regions':{},'samples':[],'rejects':[],'source_ids':{'train':[],'val':[]},'max_innovation_error_mps':0.}
 for region in REGIONS:
  src=Source(registry[region]);result['regions'][region]={}
  for split in ('train','val'):
   path=args.scan/f'{region}_{split}_relation_feasibility.json';plans=json.loads(path.read_text());eligible=[r for r in plans if r['accepted']]
   eligible.sort(key=lambda r:hashlib.sha256(r['window_id'].encode()).hexdigest());accepted=0;attempted=0
   for row in eligible:
    attempted+=1;got,reason=build_one(src,row,allowed[split])
    if got is None:result['rejects'].append(dict(region=region,split=split,window_id=row['window_id'],reason=reason));continue
    arrays,proof=got;name=f"{region}_{split}_{row['window_id']}";p=args.out/f'{name}.npz';np.savez_compressed(p,**arrays);proof.update(file=p.name,sha256=sha(p),canonical_file=registry[region]['canonical_file'],canonical_sha256=registry[region]['sha256'])
    save(args.out/f'{name}.json',proof);result['samples'].append(proof);result['source_ids'][split].append(row['mmsi']);result['max_innovation_error_mps']=max(result['max_innovation_error_mps'],proof['max_innovation_error_mps']);accepted+=1
    if accepted==2:break
   result['regions'][region][split]=dict(anchor_feasible_sources=len(eligible),attempted_smoke_candidates=attempted,accepted_sources=accepted,quartet_instances=4*accepted)
  print('QUARTET_REGION',region,result['regions'][region],flush=True)
 assert not set(result['source_ids']['train'])&set(result['source_ids']['val'])
 result.update(status='passed',independent_source_windows=len(result['samples']),quartet_instances=4*len(result['samples']),target_only_balanced_signature_verified=True,neighbor_only_balanced_signature_verified=True,train_val_target_mmsi_intersection=0,not_part_of_old_label_dataset=True)
 save(args.out/'relation_smoke_manifest.json',result);print('SMOKE_COMPLETE',len(result['samples']),flush=True)

if __name__=='__main__':main()
