"""Assemble the actual verified labeling/pilot stage without altering scores."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import pandas as pd
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;REL=BASE/'labeled_v1'
REGIONS=['houston','ny','la','puget','great_belt','haugesund','kattegat','shanghai']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())

labels=pd.read_parquet(REL/'label_manifest.parquet');assert len(labels)==11200 and labels.mmsi.nunique()==2065
label_audit=read(REL/'independent_label_audit.json');assert label_audit['status']=='passed'
cf=read(BASE/'counterfactuals_v1/independent_counterfactual_audit.json');assert cf['status']=='passed'
global_feature_audit=read(REL/'independent_labeled_features_audit.json');assert global_feature_audit['status']=='passed'
audits={}
for region in REGIONS:
 p=BASE/'validation'/f'{region}_feature_audit.json';a=read(p)
 assert a['status']=='passed',f'Feature audit not passed: {region}'
 assert set(a['views'])=={'features_scene','features_strict','features_hard_scene','paired_group_scene'}
 audits[region]=a
pilot=read(REL/'pilot_scene_seed0/pilot_summary.json');hard=read(REL/'pilot_hard_seed0/pilot_summary.json')
metric_audit=read(REL/'independent_pilot_metric_audit.json');assert metric_audit['status']=='passed' and len(metric_audit['runs'])==14
assert pilot['status']==hard['status']=='completed_single_seed_pilot' and len(pilot['results'])==12 and len(hard['results'])==2
assert pilot['source_data_unchanged'] and hard['source_data_unchanged']
probes=read(REL/'shortcut_audit/nuisance_logistic_results.json');assert probes['status']=='completed' and probes['invariant_checks_passed']
cross=read(REL/'paired_view_evaluation_seed0/cross_generation_matrix.json')
numeric=read(REL/'numeric_effect_evaluation_v2/numeric_reaggregation_integrity.json');assert numeric['status'] in ['passed','completed']
group=read(REL/'numeric_effect_evaluation_v2/numeric_group_variation_metrics.json')
assert sha(REL/'label_manifest.parquet')==read(REL/'benchmark_protocol.json')['labelled_core']['labels_sha256']

# Recheck the historical sources and experiment files against the old baseline.
old=read(ROOT/'source_preservation_verification.json');preserved=[]
for entry in old['files']+old['original_npz_files']:
 p=Path(entry['path']);actual=sha(p);assert actual==entry['expected_sha256'],str(p)
 preserved.append({'path':str(p),'sha256':actual,'bytes':p.stat().st_size})
assert len(preserved)==48
(REL/'historical_preservation_recheck.json').write_text(json.dumps({'status':'passed','files_checked':48,'files':preserved},indent=2))

evidence=REL/'evidence';evidence.mkdir(exist_ok=True)
for name in ['source_registry.json','global_all_mmsi_split.parquet','selection_report.json']:
 dest=evidence/'source_pool'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'release_20261026'/name,dest)
for relative in ['planning/capacity.json','planning/protocol_plan.json','planning/candidate_plan_validation.json','counterfactuals_v1/independent_counterfactual_audit.json','generator/unit_test_report.json','generator/real_source_diagnostic.json','pilot/pilot_selftest_report.json','builder_selftest_report.json']:
 src=BASE/relative
 if src.exists():
  dest=evidence/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for region in REGIONS:
 for src in [BASE/'validation'/f'{region}_feature_audit.json',BASE/'counterfactuals_v1'/region/'counterfactual_report.json']:
  dest=evidence/src.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
code=REL/'reproducibility';code.mkdir(exist_ok=True)
paths=['build_counterfactuals.py','build_labeled_features.py','freeze_protocol.py','physics_metrics.py','assemble_stage_review.py','render_pilot_review.py','render_generator_review.py','generator/maneuver_generator.py','generator/test_maneuver_generator.py','planning/build_label_plan.py','planning/freeze_label_plan.py','planning/verify_candidate_plan.py','planning/verify_frozen_label_plan.py','pilot/pilot_train.py','pilot/test_pilot_train.py','pilot/evaluate_paired_views.py','pilot/reaggregate_numeric_group_effects.py','pilot/verify_saved_metrics.py','validation/verify_labeled_features.py','validation/verify_labeled_features_v2.py','validation/audit_math.py','validation/audit_math_v2.py','validation/replay_group_numeric_effects.py','validate_counterfactuals.py']
for relative in paths:
 src=BASE/relative
 if src.exists():
  dest=code/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for src in [ROOT/'context/context_features.py',ROOT/'context/test_context_features.py',ROOT/'build_features.py']:
 dest=code/'source_context'/src.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(src,dest)

comparison=[]
for backbone in ['tcn','gru','transformer']:
 row={'backbone':backbone}
 for feature in ['raw','plain_context','plain_affine_context','context']:
  r=next(x for x in pilot['results'] if x['backbone']==backbone and x['feature_group']==feature)
  row[feature]=r['test']['macro']
 comparison.append(row)
matrix={f:{a:{b:cross['cells'][f][a][b]['macro']['AP'] for b in ['smooth','hard']} for a in ['smooth','hard']} for f in ['raw','context']}
probe_summary={k:v['splits']['test']['macro'] for k,v in probes['models'].items()}
summary={'version':'labeled_v1_stage_review_20261026','status':'verified_labeling_and_pipeline_pilot_complete','source_pool':{'windows':21600,'vessels_per_region':300,'windows_per_region':2700},'labelled_core':{'windows':11200,'regions':8,'vessels_per_region':260,'windows_per_region':1400,'unique_mmsi':2065,'positive':4480,'no_injected_behavior':6720,'source_observations':194324,'windows_per_region_split':{'train':980,'val':210,'test':210},'vessels_per_region_split':{'train':182,'val':39,'test':39}},'counterfactual_candidates':{'source_qualified':19044,'generator_accepted':16691,'generator_rejected':2353},'input_and_pipeline_hashes':{'labels':sha(REL/'label_manifest.parquet'),'protocol':sha(REL/'benchmark_protocol.json'),'generator':sha(BASE/'generator/maneuver_generator.py'),'context':sha(ROOT/'context/context_features.py'),'feature_builder':sha(BASE/'build_labeled_features.py')},'nuisance_probes_test':probe_summary,'pilot_comparison':comparison,'matched_tcn_cross_generation_macro_ap':matrix,'group_numeric_cohorts':group['cohort_counts'],'group_metric_definition':'Primary cohort: target actually changed and >=4 neighbor identities actually changed; fixed original validation-max-F1 thresholds, not a matched-FAR comparison','performance_observation':'With the current direct gated/context input and 8-epoch seed0 pilot, complete context does not consistently exceed raw input. This stage verifies data and execution; fusion, fallback and task coverage remain the next component review.','component_review_order':['KG gate targets, missingness and noise semantics','LF estimator, geometric degeneracy and hierarchical fallback','Raw-preserving shared context fusion','Separate PSA/consistency training ablation','Optional CDG on observation-level relational tasks'],'historical_files_rechecked':48,'server_full_release':str(REL),'review_bundle_scope':'Labels, feature NPZ, real predictions/checkpoints, source provenance, diagnostics, figures and code; full per-neighbor audit streams and source canonical/raw data stay at recorded server paths.'}
(REL/'stage_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

lines=['AIS项目下一阶段推进记录（2026-10-26）','','一、本轮已完成','1. 生成器升级：三次平滑剖面、速度向量差位移叠加、非均匀时间间隔、起始区间桥接、参数随机化、经验预算检查和明确拒绝。','2. 公平来源池：先对所有来源窗口生成同类counterfactual并验收，再分配正常/异常标签；负类只是未注入，不等同人工确认自然正常。','3. 标注主集：8域各1400窗/260船，共11200窗/2065不同MMSI；原始来源池仍为8域各2700窗/300船，共21600窗。','4. 每域train/val/test=980/210/210窗、182/39/39船，正类40%、未注入60%，五类异常每域各112例。','5. 扰动后重新计算目标门控、邻船可信度和闭式场；已生成场景、严格隔离、匹配硬阶跃和群体正常变化四视图。','6. 冻结AP/AUROC/F1、区域等权、训练集标准化和验证集阈值；完成3主干×4输入共12组seed0/8轮预跑，以及TCN两组hard训练。','','二、生成器和数据核验','19044个来源合格候选中16691个通过，2353个明确拒绝；最终共同池均衡选取11200个。','11200窗的194324个源观测、创新保留、MMSI与原观测跨组零交集均已独立核查。',f'前史统计探针test macro AUROC={probe_summary["prefix"]["auroc"]:.6f}，原始创新探针={probe_summary["original_innovation"]["auroc"]:.6f}；固定线性探针未检出明显这类标签线索。','源数据池图（300船/2700窗）和标注主集（260船/1400窗）必须分别命名；来源分布图不代表最终监督标签数。','','三、真实预跑结果（测试区域等权AP，seed0/8轮）','主干 | 原始输入 | 普通上下文 | 无权仿射场 | 可信上下文']
for r in comparison:lines.append(r['backbone']+' | '+' | '.join(f'{r[f]["AP"]:.6f}' for f in ['raw','plain_context','plain_affine_context','context']))
lines+=['','目前整套接入尚未稳定优于原始主干。下一项应核对KG门控对象、LF回退与原始特征保留/融合方式，并检查主任务对群体正常变化的覆盖。','本轮没有根据测试成绩改变模型、重选种子、重选阈值或重拟合标准化。','','四、同语义匹配的平滑/阶跃控制（TCN区域等权AP）','输入 | Smooth-Smooth | Smooth-Hard | Hard-Smooth | Hard-Hard']
for f in ['raw','context']:lines.append(f+' | '+' | '.join(f'{matrix[f][a][b]:.6f}' for a,b in [('smooth','smooth'),('smooth','hard'),('hard','smooth'),('hard','hard')]))
lines+=['','新匹配对照不重现历史图约26个百分点的落差。历史图是旧生成协议之间的迁移结果，新对照固定同源窗口、动作语义、计划幅度、起点和噪声，位置都随对应速度变化。','群体挑战按实际数值改变而非仅算子激活分层；原metadata中的operator-only成员已通过独立sidecar区分。','所有模型沿用原验证集最大F1阈值，群体FPR不是相同FAR预算下的比较。','','五、交付位置','服务器完整路径：'+str(REL),'本地核验包包含清单、特征、权重/真实预测、报告、图表及代码；完整邻船审计流和源Parquet/原始文件保留在服务器。']
(REL/'阶段推进报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
cards=['组件实现与核对说明','','1. 平滑机动生成（本轮完成加固）','作用：控制个体机动，避免把报文不相容误写成行为标签。','当前实现：p_aug = p_source + 累计梯形积分(v_aug-v_source)*真实dt；保留源位置-速度创新。方向、强度、时长、起点、蛇行相位/周期都由固定种子记录。','验证：源逐点回查、三次剖面、观测与1秒密网预算、真实窗口创新保留、匹配hard语义及参数一致。','','2. 观测污染（本轮完成）','作用：独立制造观测故障，不改变个体行为标签。','当前实现：位置跳变、报告速度尖峰、平滑漂移；分别作用目标/邻船/两者。实际生效与计划污染分别统计。背景变化后重新建立因果索引。','','3. 正常群体变化（本轮完成测试视图）','作用：模拟同向局部cohort与目标共同变化，检验个体机动和背景变化的分离。','当前实现：目标和cohort共享平滑减速因子，再施加同一观测污染规则。','验证口径：以真实数值改变的目标和邻船数为准；主要组需目标实际改变且至少4个邻船实际改变。','','4. 运动学可信度门控KG（下一项逐项讨论）','输入：相邻位置、报告SOG/COG、实际时间差。','数值：e = ||delta_p/dt - (v_k+v_(k-1))/2||，w=exp(-max(0,e-2.5)^2/8)。','当前实现：固定可配置阈值；未知可信度用显式kinematic_valid标志和中性目标权重处理。','待核对：哪些通道应门控、位置故障和速度故障如何区分、低速COG和长缺报如何处理、是否保留原始分支。','','5. 局部参照LF及回退（随后讨论）','当前实现：目标留一、每船最新因果报告、8nm内同向邻域、可信度×距离权重的局部仿射速度场。','输出：局部速度、残差向量、散度/旋度、邻域支持与有效性。','数学口径：当前拟合局部仿射场并计算导数统计，不是全局Helmholtz正交求解。','当前回退：邻船不足或几何病态时零场残差加显式无效/支持标志；支持度作为输入。','待核对：仿射场退化时是否采用可信常量场再退回自身，以及原始特征和上下文如何融合。','','6. 主干接口（本轮流程验证完成）','TCN：因果残差卷积，扩张率1/2/4/8；GRU：两层单向、packed序列；Transformer：两层编码器、位置编码、padding mask。','四组输入共享划分、样本、种子与预算；三个16通道上下文组参数量一致。原始7通道第一层参数量另报，容量匹配控制留正式实验。','','7. 4D-PSA与一致性训练（本轮关闭，后续单独讨论）','作用是训练期鲁棒性增强。应与上下文信息本身的增益分别验证。当前预跑没有把它混入完整机制。','','8. CDG关系图（本轮关闭，关系任务扩展）','需要在观测层构建有真实相对关系差异的任务，并重新计算场与门控。它不应成为所有主干必须绑定的网络。','','9. 物理指标与监督评测口径（本轮统一）','delta_velocity（m/s）表示相邻速度向量变化；e_kin（m/s）表示位置差分与报告速度相容误差，两者分开。','COG差按圆周处理、除真实dt；由COG推导的运动方向变化率不直接等同实测船体角速度。','AP采用标准average_precision；AUROC和F1按区域报告并等权汇总。标准化来自训练有效步；epoch与阈值来自验证集。','源窗口、MMSI、时间、原始行、随机化、实际参数、权重和每个预测均可追溯。']
(REL/'组件实现与核对说明.txt').write_text('\n'.join(cards)+'\n',encoding='utf-8')
files={str(p.relative_to(REL)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(REL.rglob('*')) if p.is_file() and p.name!='stage_file_manifest.json'}
(REL/'stage_file_manifest.json').write_text(json.dumps({'files':files,'scope':'All labeled-stage files except this manifest'},indent=2,ensure_ascii=False))
archive=BASE/'labeling_stage_review_20261026.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for p in sorted(REL.rglob('*')):
  if p.is_file() and not p.name.endswith('_context_audit.jsonl.gz'):
   z.write(p,str(p.relative_to(REL)),compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.png','.parquet','.pt'] else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;count=len(z.namelist())
package={'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':count,'zip_crc_passed':True,'omitted_streams':'Per-neighbor context_audit.jsonl.gz retained on server; hashes are in stage_file_manifest.json'}
(BASE/'stage_review_package_manifest.json').write_text(json.dumps(package,indent=2))
print(json.dumps({'summary':str(REL/'stage_summary.json'),'package':package},indent=2),flush=True)
