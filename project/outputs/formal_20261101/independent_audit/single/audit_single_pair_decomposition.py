from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
from sklearn.metrics import average_precision_score
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single'); O=P.parent/'independent_audit/single';N=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']; A=['raw','causal_bundle','f0_bundle','f1_bundle']; K=[('causal_minus_raw','causal_bundle','raw'),('f0_minus_causal','f0_bundle','causal_bundle'),('f1_minus_f0','f1_bundle','f0_bundle')]
R=json.loads((P/'single_pair_decomposition.json').read_text()); S=json.loads((P/'single_formal_summary.json').read_text()); rows=list(csv.DictReader((P/'single_pair_decomposition.csv').open(encoding='utf8'))); assert len(rows)==24
md=(P/'单主体四组完整对照与配对区间.md').read_text(); fig=(P/'render_four_groups.py').read_text()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'single_formal_summary.json')==R['source_summary_sha256'] and sha(P/'frozen_checkpoint_thresholds.json')==R['checkpoint_lock_sha256']
assert "single_formal_summary.json" in fig and "single_pair_decomposition.json" in fig and "['macro_AP']['mean']" in fig and "[key]['pooled']" in fig

def reps(y,s,inv,counts):
 ix=np.argsort(-s,kind='stable'); y=y[ix]; ss=s[ix]; inv=inv[ix]; endpoints=np.r_[np.where(ss[:-1]!=ss[1:])[0],len(ss)-1]; out=[]
 for start in range(0,len(counts),100):
  w=counts[start:start+100,inv].T.astype(float); tp=np.cumsum(w*y[:,None],axis=0)[endpoints]; den=np.cumsum(w,axis=0)[endpoints]; dp=np.diff(np.vstack([np.zeros((1,tp.shape[1])),tp]),axis=0); pp=np.divide(tp,den,out=np.zeros_like(tp),where=den>0); out.extend(np.divide((pp*dp).sum(axis=0),tp[-1],out=np.full(tp.shape[1],np.nan),where=tp[-1]>0))
 return np.array(out)
results={'paired':{},'checked':{'backbones':0,'pair_metric_intervals':0,'paired_seed_values':0,'csv_values':0,'markdown_pair_cells':0,'markdown_group_cells':0},'max_error':0,'draws':2000,'method':'independent per-score cumsum with tied-score end blocks','figure_reads':'single_formal_summary.json macro AP for panel a; single_pair_decomposition.json pooled paired delta and CI for panels b-d','input_hashes':{p:sha(P/p) for p in ['single_formal_summary.json','single_pair_decomposition.json','single_pair_decomposition.csv','单主体四组完整对照与配对区间.md','render_four_groups.py','figures/single_four_arm_paired.png','figures/single_four_arm_paired.svg','figures/single_four_arm_paired.pdf']}}
for n in N:
 data={a:[dict(np.load(P/'runs'/n/f'seed{i}'/'test'/f'{a}_scores.npz',allow_pickle=False)) for i in range(3)] for a in A}; ref=data['raw'][0]; y=ref['y'];m=ref['mmsi'].astype(str);reg=ref['region'].astype(str); _,inv=np.unique(m,return_inverse=True); nn=inv.max()+1;draw=np.random.default_rng(20261101).integers(0,nn,(2000,nn));counts=np.array([np.bincount(x,minlength=nn) for x in draw]);regions=np.unique(reg); bs={};pts={}
 for a in A:
  seedp=[];seedm=[]; bp=[];bm=[]
  for z in data[a]:
   for k in ('y','mmsi','region','window_id'):assert np.array_equal(z[k],ref[k])
   sc=z['score']; seedp.append(average_precision_score(y,sc)); seedm.append(np.mean([average_precision_score(y[reg==r],sc[reg==r]) for r in regions]));bp.append(reps(y,sc,inv,counts));bm.append(np.nanmean([reps(y[reg==r],sc[reg==r],inv[reg==r],counts) for r in regions],axis=0))
  pts[a]={'pooled':np.array(seedp),'macro':np.array(seedm)};bs[a]={'pooled':np.array(bp),'macro':np.array(bm)}
  for metric,key in [('pooled','pooled_AP'),('macro','macro_AP')]:
   q=S['models'][n]['test']['performance'][a][key]; assert np.allclose(pts[a][metric],q['seeds'],rtol=0,atol=1e-12); cell=f"{q['mean']:.4f} ± {q['sd']:.4f}"; assert cell in md;results['checked']['markdown_group_cells']+=1
 results['paired'][n]={}
 for name,a,b in K:
  row=next(x for x in rows if x['backbone']==n and x['comparison']==name);results['paired'][n][name]={}
  for metric in ('pooled','macro'):
   d=pts[a][metric]-pts[b][metric]; boot=np.nanmean(bs[a][metric]-bs[b][metric],axis=0); ci=np.nanpercentile(boot,[2.5,97.5]); target=R['paired'][n][name][metric]; err=max(abs(d.mean()-target['mean']),np.max(abs(d-np.array(target['per_seed']))),np.max(abs(ci-np.array(target['CI95'])))); results['max_error']=max(float(err),results['max_error']);assert err<1e-12,(n,name,metric,err)
   for idx,col in enumerate([f'{metric}_delta_pp',f'{metric}_lo_pp',f'{metric}_hi_pp']):
    val=[d.mean()*100,ci[0]*100,ci[1]*100][idx];assert abs(float(row[col])-val)<1e-10;results['checked']['csv_values']+=1
   text=f"{d.mean()*100:+.2f} [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]";assert text in md,(n,name,metric,text);results['checked']['markdown_pair_cells']+=1;results['checked']['pair_metric_intervals']+=1;results['checked']['paired_seed_values']+=3;results['paired'][n][name][metric]={'mean':float(d.mean()),'ci95':ci.tolist()}
 results['checked']['backbones']+=1; print('DONE',n,flush=True)
out=O/'single_pair_decomposition_audit.json';assert not out.exists();out.write_text(json.dumps(results,indent=2,ensure_ascii=False));print('COMPLETE',json.dumps(results['checked']),results['max_error'])
