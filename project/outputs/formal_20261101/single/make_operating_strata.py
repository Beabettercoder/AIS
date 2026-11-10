from pathlib import Path
import json
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');j=json.loads((P/'single_formal_summary.json').read_text());N=['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph'];A=['raw','causal_bundle','f0_bundle','f1_bundle'];S=j['test_strata']
lines=['# 单主体：低误报工作点与受控污染分层','','数据来自此前曝光过的测试来源。阈值只由各自模型的开发验证负类确定，以下列的是测试集实际FAR及召回；模型固定第25轮，seed0/1/2平均±样本标准差，不能称为真实运营误报保证。','', '## 1%与5%验证负类阈值下的测试表现','', '|骨干|配置|1%阈值：实际FAR|1%阈值：召回|5%阈值：实际FAR|5%阈值：召回|','|---|---|---|---|---|---|']
def v(x):return f"{x['mean']:.4f} ± {x['sd']:.4f}"
for n in N:
 for a in A:
  op=j['models'][n]['test']['performance'][a]['operating_points'];lines.append('|'+n+'|'+a+'|'+v(op['0.01']['actual_FAR'])+'|'+v(op['0.01']['recall'])+'|'+v(op['0.05']['actual_FAR'])+'|'+v(op['0.05']['recall'])+'|')
for field,title in [('noise_type','受控噪声类型'),('noise_scope','受控噪声作用范围'),('observation_noise_effective','观测污染是否实际生效')]:
 lines+=['','## '+title+'：各配置 pooled AP','','各分组保留对应正负样本，AP受该分组阳性比例影响；受控标签不是人工确认真实事故。若某组缺少正类或负类则显示—。','', '|骨干|分组|窗口数（阳性/阴性）|Raw AP|Causal+bundle AP|F0 AP|F1 AP|','|---|---|---:|---|---|---|---|']
 for n in N:
  groups=sorted(set(x['group'] for x in S if x['backbone']==n and x['arm']=='raw' and x['stratum_type']==field))
  for g in groups:
   rec=[next(x for x in S if x['backbone']==n and x['arm']==a and x['stratum_type']==field and x['group']==g) for a in A];z=rec[0];items=['—' if r['AP_mean'] is None else f"{r['AP_mean']:.4f} ± {r['AP_sd']:.4f}" for r in rec];lines.append('|'+n+'|'+g+'|'+str(z['n'])+' ('+str(z['positive'])+'/'+str(z['negative'])+')|'+'|'.join(items)+'|')
lines+=['','0.1%预算、其他实际生效范围、每行为类别与每个区域数值及逐seed指标见 single_formal_summary.json、single_formal_strata.csv 与每组 test/results.json。','']
(P/'单主体低误报与污染分层.md').write_text('\n'.join(lines),encoding='utf8')
