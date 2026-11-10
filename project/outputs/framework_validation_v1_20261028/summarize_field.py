from pathlib import Path
import json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from summarize_results import ARCH,DISPLAY,ap_draws,primary,table,fmt,interval
R=Path(__file__).resolve().parent;O=R/'field_extension'
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    assert json.loads((O/'checkpoint_lock.json').read_text())['count']==72
    d=load(R/'data'/'individual'/'test.npz');y=d['y'];regions=d['region'].astype(str);ships=d['mmsi'].astype(str);u,inv=np.unique(ships,return_inverse=True)
    boot=load(R/'analysis'/'bootstrap_individual.npz');assert np.array_equal(u,boot['unique_mmsi']);counts=boot['cluster_counts'];allmetrics=[];pairs=[];agg=[]
    for arch in ARCH:
        pred={};configs=['C0','C2','C3','F0','F1','F2']
        for arm in configs:
            root=O if arm.startswith('F') else R;ps=[];rr=[]
            for seed in [0,1,2]:
                folder=root/'runs'/'individual'/arch/f'seed_{seed}'/arm;p=load(folder/'test_base.npz');tr=json.loads((folder/'result.json').read_text());te=json.loads((folder/'test_results.json').read_text())['views']['base']
                assert np.array_equal(p['window_id'],d['window_id']) and np.array_equal(p['y'],y)
                s=p['score'];ps.append(s);m={'backbone':arch,'arm':arm,'seed':seed,'macro_AP':primary(y,s,regions,'individual'),'pooled_AP':float(average_precision_score(y,s)),'AUROC':float(roc_auc_score(y,s)),'validation_macro_AP':tr['validation']['macro_AP'],'initial_state_sha256':tr['initial_state_sha256'],'parameters':tr['parameters'],'train_seconds':tr['train_elapsed_seconds']}
                for k,v in te['operating'].items():m['FAR_'+k]=v['FAR'];m['Recall_'+k]=v['recall']
                rr.append(m);allmetrics.append(m)
            pred[arm]=ps
            ag={'backbone':arch,'arm':arm}
            for key in ['macro_AP','pooled_AP','AUROC','validation_macro_AP','parameters','train_seconds']+[k for k in rr[0] if k.startswith(('FAR_','Recall_'))]:
                a=np.asarray([r[key] for r in rr],float);ag[key]={'mean':float(a.mean()),'std':float(a.std(ddof=1))}
            agg.append(ag)
        for seed in [0,1,2]:
            hashes={r['initial_state_sha256'] for r in allmetrics if r['backbone']==arch and r['seed']==seed};assert len(hashes)==1
        ds={a:[] for a in configs}
        for start in range(0,len(counts),128):
            w=counts[start:start+128,inv].astype(float)
            for arm in configs:ds[arm].extend(np.mean([ap_draws(y,s,w,regions,True) for s in pred[arm]],axis=0).tolist())
        for plus,minus in [('F1','F0'),('F1','C0'),('F2','C0'),('F2','F1'),('F1','C3')]:
            delta=np.asarray([primary(y,s,regions,'individual') for s in pred[plus]])-np.asarray([primary(y,s,regions,'individual') for s in pred[minus]])
            dist=np.asarray(ds[plus])-np.asarray(ds[minus]);dist=dist[np.isfinite(dist)]
            pairs.append({'backbone':arch,'comparison':plus+'-'+minus,'mean_delta_pp':float(delta.mean()*100),'seed_deltas_pp':(delta*100).tolist(),'ci95_pp':(np.quantile(dist,[.025,.975])*100).tolist(),'valid_draws':len(dist),'clusters':len(u),'scope':'pointwise descriptive MMSI cluster interval at fixed three trained seeds'})
        print('FIELD_SUMMARY',arch,flush=True)
    out=O/'analysis';out.mkdir(exist_ok=True)
    for name,obj in [('per_seed_metrics',allmetrics),('aggregate_metrics',agg),('paired_bootstrap',pairs)]: (out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    def a(b,c):return next(x for x in agg if x['backbone']==b and x['arm']==c)
    def p(b,c):return next(x for x in pairs if x['backbone']==b and x['comparison']==c)
    rows=[]
    for b in ARCH:
        rows.append([DISPLAY[b]]+[fmt(a(b,c)['macro_AP']['mean'],a(b,c)['macro_AP']['std']) for c in ['C0','F0','F1','F2','C3']]+[interval(p(b,c)) for c in ['F1-F0','F2-C0']])
    txt='# Table 2 主线补充：运动学可信度与留一局部仿射场\n\n八类分类编码器，固定25轮，3种子，区域等权AP。F0/F1同一候选和几何支持，F1仅增加绝对运动学可信权重；F2再加入已核验的目标门控输入包。C3中位数作为背景估计对照保留。所有输入使用同一训练集标准化。\n\n'
    txt+=table(['骨干','C0 原始','F0 无权LOO场','F1 可信LOO场','F2 目标门控+可信场','C3 可信中位数','F1−F0 pp [CI]','F2−C0 pp [CI]'],rows)
    txt+='\n\n这里的“无权”指无可信权重，F0仍使用距离高斯权重。场在观测目标位置查询；F2不根据修复位置重新检索邻居。新局部仿射参数和旧实现差异在field_protocol.json中记录。测试来源已用于开发，区间是逐比较描述性结果，不作未经校正的全局显著性宣称。\n'
    (R/'tables'/'Table2_kinematic_loo_field.md').write_text(txt,encoding='utf8')
    rows=[[DISPLAY[b]]+[interval(p(b,c)) for c in ['F1-F0','F2-F1','F1-C3']] for b in ARCH]
    (R/'tables'/'Table4_field_components.md').write_text('# Table 4 主线组件配对\n\n'+table(['骨干','邻船可信度 F1−F0','目标门控包 F2−F1','仿射场−中位数 F1−C3'],rows)+'\n\n最后一列是整个背景估计器的比较，包含距离权重与空间拟合，不能单独归于某个参数。',encoding='utf8')
    verification={'status':'passed','runs':72,'comparisons':len(pairs),'paired_initialization':'F0/F1/F2/C0/C2/C3 match for every backbone/seed','test_rows_identical':True}
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
if __name__=='__main__':main()
