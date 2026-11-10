from pathlib import Path
import json,hashlib,zipfile,html
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
table_names=[('Table 1','数据来源与划分','Table1_dataset.md'),('Table 2 主线','运动学可信度与留一仿射场','Table2_kinematic_loo_field.md'),('Table 2 对照','均值与中位数跨骨干对照','Table2_cross_backbone.md'),('Table 4 主线','场可信权重与目标门控包','Table4_field_components.md'),('Table 4 对照','聚合器与FSC训练扩展','Table4_core_components.md'),('Table 4 压力','场模型三类邻船观测压力','Table4_field_stress_summary.md'),('Table 6','八骨干群体关系三视图','Table6_relation_neural.md')]
fig_names=[('Fig. 1','运动学可信度—留一场—个体残差','fig1_kinematic_loo_mainline.png'),('Fig. 2','八区域与上下文支持','fig2_geography_support.png'),('Fig. 3','留一场的跨骨干配对结果','fig3_kinematic_loo_field_gain.png'),('Fig. 4','留一场观测压力与实际误报工作点','fig4_kinematic_loo_field_robustness.png'),('Fig. 5 对照版','中位数与图注意力分支验证敏感性','fig5_sensitivity.png'),('Fig. 8','固定案例的场模型响应','fig8_kinematic_loo_mainline.png')]
for _,_,name in table_names:assert (R/'tables'/name).exists(),name
for _,_,name in fig_names:assert (R/'figures'/name).exists(),name
for name in ['audit/FINAL_AUDIT.md','audit/FIELD_FINAL_AUDIT.md','本轮实验报告_20261029.md']:assert (R/name).exists(),name
status={'date':'2026-10-29','status':'completed_stage_not_submission_complete','formal_training_runs':312,'core_training_runs':240,'field_training_runs':72,'core_stress_prediction_files':1800,'field_stress_prediction_files':1080,'sensitivity_evaluations':207,'universal_gain_established':False,'mainline':'kinematic compatibility -> target-excluded background field -> individual motion residual -> multiple encoders','tables':table_names,'figures':fig_names,'not_completed':['Table 3 external-method reproduction','Table 5 / Fig 6 new-domain transfer','Table 7 unified LLM comparison','Fig 7 new formal generation/early-warning study','field-version parameter sensitivity','road-traffic validation','independent new-source confirmation set']}
(R/'交付状态.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
style="body{font-family:'Times New Roman','Microsoft YaHei',serif;background:#f5f8f7;color:#173b35;margin:0}main{max-width:1140px;margin:auto;padding:36px 28px}h1{font-size:32px}h2{font-size:23px}p{line-height:1.8}.box{background:white;border:1px solid #d4e3dc;border-radius:10px;padding:24px;margin:20px 0}.note{background:#fff8ec;border-left:4px solid #bb862d;padding:16px 20px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.n{font-size:32px;font-weight:bold;color:#0f6c5c}a{color:#0f6c5c}img{width:100%;height:auto;border:1px solid #e0e8e4}li{line-height:1.9}small{color:#586f67}@media(max-width:700px){.grid{grid-template-columns:1fr}}"
text='<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>框架实验阶段结果</title><style>'+style+'</style></head><body><main><h1>框架实验阶段结果</h1><p>2026年10月29日 · 原主线保留：运动学可信度 → 留一背景场 → 个体残差 → 多种骨干。</p><div class="grid"><div class="box"><div class="n">312</div>正式训练与相应测试</div><div class="box"><div class="n">8</div>明确记录的编码器适配</div><div class="box"><div class="n">2,880</div>观测压力模型/视图预测文件</div></div><p class="note">当前结果未证明在单主体基准上跨骨干普遍净增益。关系任务支持上下文作用；可信度在速度污染等特定条件下有额外收益。原始、无权、可信和负结果全部保留。</p><div class="box"><h2>先读报告</h2><ul><li><a href="本轮实验报告_20261029.md">本轮实验报告</a></li><li><a href="主线确认与实验映射.md">主线确认与实验映射</a></li><li><a href="audit/FINAL_AUDIT.md">核心实验独立核验</a></li><li><a href="audit/FIELD_FINAL_AUDIT.md">场扩展独立核验</a></li></ul></div><div class="box"><h2>结果表</h2><ul>'
for id,title,name in table_names:text+=f'<li><a href="tables/{html.escape(name)}">{id}：{html.escape(title)}</a></li>'
text+='</ul></div>'
for id,title,name in fig_names:text+=f'<div class="box"><h2>{id}　{html.escape(title)}</h2><img src="figures/{html.escape(name)}" alt="{html.escape(title)}"><p><small>图注、数值与版本记录保存在同目录；对照版本与局部仿射主线分别标注。</small></p></div>'
text+='<div class="box"><h2>未完成项</h2><p>新统一协议的外部检测方法、未见区域确认、大模型对比，以及道路交通验证仍待开展。数据、模型与训练配方的结果不混写成旧版本结论。</p></div></main></body></html>'
(R/'结果阅读入口.html').write_text(text,encoding='utf8')
readme='''# 本轮结果阅读入口

- [阶段报告](本轮实验报告_20261029.md)
- [可视阅读入口](结果阅读入口.html)
- [主线确认](主线确认与实验映射.md)
- [交付状态](交付状态.json)
- [核心核验](audit/FINAL_AUDIT.md)
- [场模型核验](audit/FIELD_FINAL_AUDIT.md)

本轮312次训练分为240组聚合对照与72组留一局部仿射场。原故事和两组实际实现已经分别记录。未证明普遍净增益的结果不删除，不替换成历史分数。

服务器完整目录：`/mnt/sda/WangZijia/AIS/outputs/framework_validation_v1_20261028/`。
本地报告和结果镜像：`E:/AIS/outputs/framework_validation_v1_20261028/`。

结果包包括报告、图表、协议、分析和独立核验。完整执行档案另含数据缓存、最终权重和逐窗口预测；复现数据构建仍依赖既有原始AIS及前序代码，不能把档案称为原始数据全集。运行环境与上游版本分别见protocol/model_inventory/vendor。
'''
(R/'README_结果.md').write_text(readme,encoding='utf8')
files=[]
for p in R.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(R)
    if any(part in ['runs','stress_results','field_stress_results','data','data_incomplete_attempt1','data_incomplete_attempt2','__pycache__','logs','job_events','sensitivity_results','sensitivity_results_cpu_partial'] for part in rel.parts):continue
    if p.suffix not in ['.md','.json','.py','.png','.svg','.csv'] and p.name!='结果阅读入口.html':continue
    if p.name in ['results_manifest.json']:continue
    files.append(p)
manifest={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in files}
(R/'results_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');files.append(R/'results_manifest.json')
dest=R.parent/'framework_results_20261029.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(R))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
print(json.dumps({'report_package':str(dest),'files':len(files),'bytes':dest.stat().st_size,'sha256':sha(dest)},ensure_ascii=False))
