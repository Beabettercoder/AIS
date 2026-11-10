from pathlib import Path
base=Path('/mnt/sda/WangZijia/AIS/outputs'); o=base/'low_speed_history_floor_20261030/fixed_normalizer'
s=(base/'neighbor_gate_safe_20261030/fixed_normalizer/summary_fixed.py').read_text()
s=s.replace("out=base/'neighbor_gate_safe_20261030/fixed_normalizer'", "out=base/'low_speed_history_floor_20261030/fixed_normalizer'")
s=s.replace("p=out.parent/'runs/aligned_embed'/('seed'+str(seed))/name/arm", "p=base/'neighbor_gate_safe_20261030/runs/aligned_embed'/('seed'+str(seed))/name/arm")
s=s.replace('safe-neighbor gate vs historical absolute gate', 'bounded history-low-speed negative control vs historical absolute gate')
assert "out=base/'low_speed_history_floor_20261030/fixed_normalizer'" in s
(o/'summary_fixed.py').write_text(s,encoding='utf8')
