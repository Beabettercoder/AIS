"""End-to-end verifier acceptance/rejection tests in a fresh /tmp fixture only."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

project = Path(sys.argv[1]).resolve()
prior = Path(sys.argv[2]).resolve()
temp = Path(tempfile.mkdtemp(prefix="ais_release_verifier_selftest_"))
release = temp/"release"
release.mkdir()
(temp/"context").mkdir()
shutil.copy2(project/"build_features.py",temp/"build_features.py")
shutil.copy2(project/"context/context_features.py",temp/"context/context_features.py")
for name in ("selected_windows.parquet","global_all_mmsi_split.parquet","selection_report.json"):
    shutil.copy2(prior/"selection"/name,release/name)
shutil.copy2(prior/"canonical.parquet",temp/"canonical.parquet")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
registry=json.loads((prior/"registry.json").read_text())
registry["regions"]["synthetic"]["canonical_file"]=str(temp/"canonical.parquet")
registry["regions"]["synthetic"]["sha256"]=sha(temp/"canonical.parquet")
(release/"source_registry.json").write_text(json.dumps(registry))
run=subprocess.run([sys.executable,str(temp/"build_features.py"),"--selection",str(release),"--registry",str(release/"source_registry.json")],capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
command=[sys.executable,str(project/"validation/verify_release.py"),"--release",str(release),
         "--expected-regions","1","--ships-per-region","3","--windows-per-region","3",
         "--split-ships","1,1,1","--split-windows","1,1,1"]
run=subprocess.run(command,capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
clean=json.loads((release/"independent_release_audit.json").read_text())
dest=release/"features/synthetic"
npz=dest/"train.npz"
report_path=dest/"feature_report.json"
original_npz=npz.read_bytes();original_report=report_path.read_bytes()
with np.load(npz) as f:arrays={k:f[k] for k in f.files}
arrays["raw"][0,1,3]+=1.
np.savez_compressed(npz,**arrays)
report=json.loads(original_report)
report["output_hashes"]["train"]["npz_sha256"]=sha(npz)
report_path.write_text(json.dumps(report))
run=subprocess.run(command,capture_output=True,text=True)
bad_raw=json.loads((release/"independent_release_audit.json").read_text())
assert run.returncode!=0 and any("physical recomputation mismatch" in s for s in bad_raw["errors"]),run.stdout+run.stderr
npz.write_bytes(original_npz);report_path.write_bytes(original_report)
audit=dest/"train_context_audit.jsonl.gz"
with gzip.open(audit,"rt") as f:md=json.loads(next(f))
md["steps"][1]["neighbors"][0]["mmsi"]="200000100"
with gzip.open(audit,"wt") as f:f.write(json.dumps(md)+"\n")
report=json.loads(original_report)
report["output_hashes"]["train"]["audit_sha256"]=sha(audit)
report_path.write_text(json.dumps(report))
run=subprocess.run(command,capture_output=True,text=True)
bad_identity=json.loads((release/"independent_release_audit.json").read_text())
assert run.returncode!=0 and any("Cross-group neighbor" in s for s in bad_identity["errors"]),run.stdout+run.stderr
result=dict(clean_release_passed=clean["status"]=="passed",tampered_raw_rejected=bad_raw["errors"],
            cross_split_neighbor_rejected=bad_identity["errors"],fixture=str(temp),formal_files_modified=False)
(temp/"verifier_selftest_result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
