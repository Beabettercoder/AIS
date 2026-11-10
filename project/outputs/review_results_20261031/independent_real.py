import pandas as pd,numpy as np,json
from pathlib import Path
from sklearn.metrics import roc_auc_score,average_precision_score
root=Path('/mnt/sda/WangZijia/AIS/outputs');R=root/'real_low_speed_expanded_20261031'; df=pd.read_csv(R/'review_manifest.csv',dtype={'mmsi':str});split=pd.read_parquet(root/'balanced_global_v2/release_20261026/global_all_mmsi_split.parquet',columns=['mmsi','split']);split['mmsi']=split['mmsi'].astype(str);assert not split.mmsi.duplicated().any();joined=df.merge(split,on='mmsi',how='left',validate='many_to_one');assert (joined['split']=='train').all();assert df.groupby('mmsi').size().max()<=6
assert set(df['group'])=={'turn_rule','straight_control'};turn=df.group.eq('turn_rule');d=np.abs(df.cog_delta_deg);assert (d[turn]>=30).all() and (d[~turn]<=10).all();assert df.heading_usable.dtype==bool
z=df[df.heading_usable];y=z.group.eq('turn_rule').to_numpy();s=z.heading_delta_deg.abs().to_numpy();ids=z.mmsi.to_numpy(); auc=roc_auc_score(y,s);ap=average_precision_score(y,s);summary=json.loads((R/'review_summary.json').read_text()); assert abs(auc-summary['heading_score_vs_cog_rule']['auc'])<1e-12 and abs(ap-summary['heading_score_vs_cog_rule']['ap'])<1e-12
rng=np.random.default_rng(20261031);u=np.unique(ids);ss=[]
for _ in range(2000):
 draws=rng.choice(u,len(u),replace=True);uniq,cnt=np.unique(draws,return_counts=True);W=dict(zip(uniq,cnt));w=np.array([W.get(i,0) for i in ids]);
 if w[y].sum()>0 and w[~y].sum()>0:ss.append(roc_auc_score(y,s,sample_weight=w))
ci=np.quantile(ss,[.025,.975]);assert np.allclose(ci,summary['bootstrap']['auc_95'],atol=1e-12)
res={'all_rows':len(df),'unique_mmsi':int(df.mmsi.nunique()),'mmsi_in_both_groups':len(set(df.loc[turn,'mmsi']) & set(df.loc[~turn,'mmsi'])),'all_train':True,'use_rows':len(z),'auc':auc,'ap':ap,'ci':ci.tolist(),'no_heading_in_selection':'review.py inspected, selection and group established before heading data read'};(root/'review_results_20261031/independent_real.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
