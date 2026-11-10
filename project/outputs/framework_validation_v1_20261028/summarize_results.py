"""Recompute paired AP and operating points from saved per-window predictions."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parent
ARCH=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph_temporal']
DISPLAY={'tcn':'TCN','gru':'GRU','transformer':'Transformer','itransformer':'iTransformer adaptation','patchtst':'PatchTST adaptation','mamba':'Mamba selective-SSM reference','neuralcde':'Neural-CDE RK4','graph_temporal':'Target-reference GAT–TCN'}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def ap_matrix(y,p,weights):
    """Tie-aware weighted AP; each weights row is one cluster bootstrap draw."""
    order=np.argsort(-p,kind='stable');ys=np.asarray(y)[order];s=np.asarray(p)[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    w=weights[:,order];tp=np.cumsum(w*ys[None,:],axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends]
    added=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
    prec=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    denom=tp[:,-1];out=np.divide((prec*added).sum(1),denom,out=np.full(len(w),np.nan),where=denom>0)
    return out
def ap_draws(y,p,weights,regions,macro):
    if not macro:return ap_matrix(y,p,weights)
    parts=[]
    for r in np.unique(regions):
        sel=regions==r;parts.append(ap_matrix(y[sel],p[sel],weights[:,sel]))
    return np.mean(np.stack(parts),axis=0)
def primary(y,p,region,task):
    if task=='relation':return float(average_precision_score(y,p))
    return float(np.mean([average_precision_score(y[region==r],p[region==r]) for r in np.unique(region)]))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def fmt(mean,std):return f'{mean:.4f} ± {std:.4f}'
def interval(d):return f"{d['mean_delta_pp']:+.2f} [{d['ci95_pp'][0]:+.2f}, {d['ci95_pp'][1]:+.2f}]"
def run(draws):
    out=R/'analysis';out.mkdir(exist_ok=True);metrics=[];paired=[];draw_registry={}
    checkpoints=json.loads((R/'checkpoint_lock.json').read_text());assert checkpoints['count']==240
    for task in ['individual','relation']:
        data=load(R/'data'/task/'test.npz');y=data['y'];reg=data['region'].astype(str);ships=data['mmsi'].astype(str);unique,inv=np.unique(ships,return_inverse=True)
        rng=np.random.default_rng(772026 if task=='individual' else 882026)
        counts=np.asarray([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique)) for _ in range(draws)],np.int16)
        np.savez_compressed(out/f'bootstrap_{task}.npz',unique_mmsi=unique,cluster_counts=counts)
        draw_registry[task]={'draws':draws,'clusters':len(unique),'draw_file':f'bootstrap_{task}.npz'}
        views=['base'] if task=='individual' else ['base','natural','suffix']
        for arch in ARCH:
          for view in views:
            predictions={}
            for arm in ['C0','C1','C2','C3','C4']:
                ps=[];rows=[]
                for seed in [0,1,2]:
                    folder=R/'runs'/task/arch/f'seed_{seed}'/arm
                    p=load(folder/f'test_{view}.npz');assert np.array_equal(p['y'],y) and np.array_equal(p['window_id'],data['window_id']);assert np.array_equal(p['mmsi'].astype(str),ships)
                    result=json.loads((folder/'result.json').read_text());tested=json.loads((folder/'test_results.json').read_text())['views'][view]
                    score=p['score'];ps.append(score)
                    actual=primary(y,score,reg,task);pooled=float(average_precision_score(y,score));auc=float(roc_auc_score(y,score))
                    assert abs(pooled-tested['pooled_AP'])<1e-12
                    sample={'task':task,'backbone':arch,'display_name':DISPLAY[arch],'view':view,'arm':arm,'seed':seed,'primary_AP':actual,'pooled_AP':pooled,'macro_AP':float(np.mean([average_precision_score(y[reg==r],score[reg==r]) for r in np.unique(reg)])),'AUROC':auc,'parameters':result['parameters'],'train_seconds':result['train_elapsed_seconds'],'initial_state_sha256':result['initial_state_sha256'],'validation_primary':result['validation']['macro_AP' if task=='individual' else 'pooled_AP']}
                    for budget,op in tested['operating'].items():
                        pred=score>op['threshold'];far=float(pred[y==0].mean());rec=float(pred[y==1].mean());assert abs(far-op['FAR'])<1e-12 and abs(rec-op['recall'])<1e-12
                        sample['FAR_'+budget]=far;sample['Recall_'+budget]=rec
                    metrics.append(sample)
                predictions[arm]=np.stack(ps)
                # Independent weighted-AP implementation must equal sklearn before bootstrap.
                for score in ps:
                    ref=ap_matrix(y,score,np.ones((1,len(y))))[0];assert abs(ref-average_precision_score(y,score))<1e-12
            distribution={arm:[] for arm in predictions}
            for start in range(0,draws,128):
                w=counts[start:start+128,inv].astype(float)
                for arm,ps in predictions.items():
                    ds=np.mean([ap_draws(y,score,w,reg,task=='individual') for score in ps],axis=0);distribution[arm].extend(ds.tolist())
            for enhanced,baseline in [('C3','C0'),('C3','C2'),('C4','C3'),('C2','C1')]:
                d=np.asarray(distribution[enhanced])-np.asarray(distribution[baseline]);d=d[np.isfinite(d)]
                ds=[primary(y,p,reg,task) for p in predictions[enhanced]];bs=[primary(y,p,reg,task) for p in predictions[baseline]]
                delta=np.asarray(ds)-np.asarray(bs)
                paired.append({'task':task,'backbone':arch,'view':view,'comparison':enhanced+'-'+baseline,'mean_delta_pp':float(delta.mean()*100),'seed_deltas_pp':(delta*100).tolist(),'ci95_pp':(np.quantile(d,[.025,.975])*100).tolist(),'valid_bootstrap_draws':len(d),'clusters':len(unique),'inference_scope':'pointwise descriptive MMSI-cluster percentile interval, conditional on three trained seeds; no simultaneous significance claim'})
            print('SUMMARIZED',task,arch,view,flush=True)
    aggregate=[]
    for task in ['individual','relation']:
     for arch in ARCH:
      for view in (['base'] if task=='individual' else ['base','natural','suffix']):
       for arm in ['C0','C1','C2','C3','C4']:
        rr=[r for r in metrics if (r['task'],r['backbone'],r['view'],r['arm'])==(task,arch,view,arm)]
        obj={'task':task,'backbone':arch,'view':view,'arm':arm}
        for key in ['primary_AP','pooled_AP','macro_AP','AUROC','validation_primary','parameters','train_seconds']+[k for k in rr[0] if k.startswith(('FAR_','Recall_'))]:
            values=np.array([r[key] for r in rr],float);obj[key]={'mean':float(values.mean()),'std':float(values.std(ddof=1))}
        # Paired architecture capacity and initial states must match all five arms per seed.
        for seed in [0,1,2]:
            states={r['initial_state_sha256'] for r in metrics if (r['task'],r['backbone'],r['view'],r['seed'])==(task,arch,view,seed)};assert len(states)==1
        aggregate.append(obj)
    write(out/'per_seed_metrics.json',metrics);write(out/'aggregate_metrics.json',aggregate);write(out/'paired_bootstrap.json',paired);write(out/'bootstrap_manifest.json',draw_registry)
    with (out/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    def agg(task,arch,view,arm):return next(r for r in aggregate if (r['task'],r['backbone'],r['view'],r['arm'])==(task,arch,view,arm))
    def pair(task,arch,view,comparison):return next(r for r in paired if (r['task'],r['backbone'],r['view'],r['comparison'])==(task,arch,view,comparison))
    txt='# Table 2 跨骨干框架有效性\n\n新统一接口开发实验；三个种子均值±SD。单主体为区域等权AP，关系为pooled AP。测试来源此前已经曝光。\n'
    for task,title in [('individual','A 单主体任务'),('relation','B 群体关系任务 Base')]:
        rows=[]
        for arch in ARCH:
            rows.append([DISPLAY[arch]]+[fmt(agg(task,arch,'base',a)['primary_AP']['mean'],agg(task,arch,'base',a)['primary_AP']['std']) for a in ['C0','C1','C2','C3','C4']]+[interval(pair(task,arch,'base',c)) for c in ['C3-C0','C3-C2']])
        txt+='\n## '+title+'\n\n'+table(['骨干','C0','C1','C2','C3','C4','C3−C0 pp [95% CI]','C3−C2 pp [95% CI]'],rows)+'\n'
    txt+='\nC4是新FSC-v1训练扩展；不是历史4D训练包。模型是明确记录的分类适配/参考实现。区间为逐比较描述性区间，条件于现有种子与目标MMSI，不等同多重校正显著性或未见区域证据。\n'
    (R/'tables').mkdir(exist_ok=True);(R/'tables'/'Table2_cross_backbone.md').write_text(txt,encoding='utf8')
    rows=[]
    for task in ['individual','relation']:
     for arch in ARCH:
      rows.append([task,DISPLAY[arch]]+[interval(pair(task,arch,'base',c)) for c in ['C2-C1','C3-C2','C4-C3']])
    (R/'tables'/'Table4_core_components.md').write_text('# Table 4 核心聚合与训练扩展配对\n\n'+table(['任务','骨干','稳健统计 C2−C1','可信度 C3−C2','训练扩展 C4−C3'],rows)+'\n\n单位为百分点及目标MMSI聚类区间。目标修复和完整历史损失正交实验为独立旧接口，不在此表混入。',encoding='utf8')
    rows=[]
    for arch in ARCH:
     for arm in ['C0','C1','C2','C3','C4']:
      rows.append([DISPLAY[arch],arm]+[fmt(agg('relation',arch,v,arm)['pooled_AP']['mean'],agg('relation',arch,v,arm)['pooled_AP']['std']) for v in ['base','natural','suffix']])
    (R/'tables'/'Table6_relation_neural.md').write_text('# Table 6 群体关系三视图神经模型结果\n\n全部只用Base训练，固定25轮；每格pooled AP均值±SD。880个四格变体对应220来源、83目标船。\n\n'+table(['骨干','配置','Base','自然压力','后段定向压力'],rows)+'\n',encoding='utf8')
    write(out/'summary_verification.json',{'status':'passed','model_seed_view_rows':len(metrics),'aggregate_cells':len(aggregate),'paired_cells':len(paired),'pointwise_intervals':True,'seed_count':3,'initialization_pairing':'all matched','metrics_recomputed':'AP and operating points from per-window scores'})
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--draws',type=int,default=2000);run(a.parse_args().draws)
