from pathlib import Path
import hashlib,json
r=Path('/mnt/sda/WangZijia/AIS/outputs');src=r/'relation_repair_20261030/builder.py';out=r/'formal_20261101/relation';s=src.read_text(encoding='utf8')
old="OUT=BASE/'relation_repair_20261030'";new="OUT=BASE/'formal_20261101/relation'"
assert s.count(old)==1;s=s.replace(old,new)
t=out/'scripts/build_test.py';t.write_text(s,encoding='utf8')
report={'source_builder':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'isolated_builder_sha256':hashlib.sha256(t.read_bytes()).hexdigest(),'only_change':'output directory','before_test_construction':True}
(out/'test_builder.lock.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(report)
