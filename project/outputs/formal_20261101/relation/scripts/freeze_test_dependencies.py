from pathlib import Path
import hashlib,json
O=Path('/mnt/sda/WangZijia/AIS/outputs');P=O/'formal_20261101/relation';deps=[O/'framework_validation_v1_20261028/data_builder.py',O/'framework_validation_v1_20261028/field_builder.py',O/'balanced_global_v2/causal_gate_main_v1_20261027/gate_builder.py',O/'balanced_global_v2/relation_dual_pressure_v2_20261027/dataset_manifest_combined.json']
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=P/'builder_dependencies.lock.json';assert not p.exists();p.write_text(json.dumps({'scope':'test-builder imports and upstream manifest, captured before test construction','files':{str(f):h(f) for f in deps}},indent=2),encoding='utf8')
print(str(p))
