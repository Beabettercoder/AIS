"""Scene audit tests use a fresh fixture, never formal release arrays."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

project, prior = map(lambda p:Path(p).resolve(),sys.argv[1:3])
temp=Path(tempfile.mkdtemp(prefix="ais_scene_verifier_selftest_"));release=temp/"release"
release.mkdir();(temp/"context").mkdir()
for name in ("build_features.py","build_features_scene.py"):
    shutil.copy2(project/name,temp/name)
shutil.copy2(project/"context/context_features.py",temp/"context/context_features.py")
for name in ("selected_windows.parquet","global_all_mmsi_split.parquet","selection_report.json"):
    shutil.copy2(prior/"selection"/name,release/name)
shutil.copy2(prior/"canonical.parquet",temp/"canonical.parquet")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
registry=json.loads((prior/"registry.json").read_text())
registry["regions"]["synthetic"].update(canonical_file=str(temp/"canonical.parquet"),sha256=sha(temp/"canonical.parquet"))
(release/"source_registry.json").write_text(json.dumps(registry))
for name in ("build_features.py","build_features_scene.py"):
    run=subprocess.run([sys.executable,str(temp/name),"--selection",str(release),"--registry",str(release/"source_registry.json")],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
command=[sys.executable,str(project/"validation/verify_scene_release.py"),"--release",str(release),
         "--expected-regions","1","--ships-per-region","3","--windows-per-region","3",
         "--split-ships","1,1,1","--split-windows","1,1,1"]
run=subprocess.run(command,capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
report_path=release/"independent_scene_release_audit.json"
clean=json.loads(report_path.read_text())
assert clean["same_source_windows_as_strict_verified"]
assert clean["number_of_independent_source_windows"]==3
assert clean["cross_split_checks"]["train_vs_test"]["all_accessed_identity_overlap"]>0
dest=release/"scene_features/synthetic"
audit=dest/"train_context_audit.jsonl.gz"
with gzip.open(audit,"rt") as f:md=json.loads(next(f))
md["steps"][1]["neighbors"][0]["mmsi"]="200000100"  # globally held-out val target
with gzip.open(audit,"wt") as f:f.write(json.dumps(md)+"\n")
report=json.loads((dest/"feature_report.json").read_text())
report["output_hashes"]["train"]["audit_sha256"]=sha(audit)
(dest/"feature_report.json").write_text(json.dumps(report))
run=subprocess.run(command,capture_output=True,text=True)
bad=json.loads(report_path.read_text())
assert run.returncode!=0 and any("Forbidden held-out target" in s for s in bad["errors"]),run.stdout+run.stderr
result=dict(scene_clean_passed=True,legitimate_background_recurrence_accepted=True,
            same_target_arrays_as_strict_verified=True,independent_window_count_not_doubled=True,
            forbidden_held_out_target_rejected=bad["errors"],fixture=str(temp))
(temp/"scene_verifier_selftest_result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
