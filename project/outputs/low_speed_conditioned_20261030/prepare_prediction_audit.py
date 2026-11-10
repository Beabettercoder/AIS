from pathlib import Path
P=Path('/mnt/sda/WangZijia/AIS/outputs')
s=(P/'gate_step_diagnostic_20261030/audit_fixed_predictions.py').read_text()
s=s.replace("parent=base/'neighbor_gate_safe_20261030'; root=parent/'fixed_normalizer'; out=base/'gate_step_diagnostic_20261030'", "parent=base/'low_speed_history_floor_20261030'; root=parent/'fixed_normalizer'; out=base/'low_speed_conditioned_20261030'")
s=s.replace("elif arm=='C0':path=parent/'runs/aligned_embed'", "elif arm=='C0':path=base/'neighbor_gate_safe_20261030/runs/aligned_embed'")
s=s.replace("(out/'fixed_normalizer_predictions_independent_audit.json')", "(out/'history_gate_predictions_independent_audit.json')")
assert "parent=base/'low_speed_history_floor_20261030'" in s and "elif arm=='C0':path=base/'neighbor_gate_safe_20261030/runs/aligned_embed'" in s
(P/'low_speed_conditioned_20261030/independent_prediction_audit.py').write_text(s)
