from pathlib import Path
p=Path('/mnt/sda/WangZijia/AIS/outputs/low_speed_quantile_20261030/calibrate.py');s=p.read_text()
s=s.replace('seen=set(); es=[]; nseen=0','seen=set(); es=[]; memberids=[]; nseen=0')
s=s.replace('seen.add(key); es.append(float(np.linalg.norm(vp-(v[i]+v[i-1])/2)))','seen.add(key); es.append(float(np.linalg.norm(vp-(v[i]+v[i-1])/2))); memberids.append(str(ids[j]))')
s=s.replace("f=O/'calibration_full.json'", "f=O/'calibration_full_cluster.json'")
s=s.replace("f.write_text(json.dumps(d,indent=2),encoding='utf8'); print(json.dumps(d,indent=2))", """arr=np.asarray(es); memberids=np.asarray(memberids); uid,inv=np.unique(memberids,return_inverse=True); order=np.argsort(arr); x=arr[order]; g=inv[order]; rng=np.random.default_rng(20261030); bounds=[]
for k in range(2000):
 sample=rng.integers(0,len(uid),size=len(uid)); counts=np.bincount(sample,minlength=len(uid)); c=np.cumsum(counts[g]); tot=c[-1]; bounds.append([float(x[np.searchsorted(c,.95*tot)]),float(x[np.searchsorted(c,.99*tot)])])
bounds=np.asarray(bounds);d.update({'target_mmsi_count':len({str(r['mmsi']) for r in rows}),'neighbor_mmsi_count':len(uid),'neighbor_mmsi_cluster_bootstrap_2000_q95_CI':np.quantile(bounds[:,0],[.025,.975]).tolist(),'neighbor_mmsi_cluster_bootstrap_2000_q99_CI':np.quantile(bounds[:,1],[.025,.975]).tolist(),'cluster_note':'conditional on repeated/constructed base records; does not remove shared-target context and no true labels'})
f.write_text(json.dumps(d,indent=2),encoding='utf8'); print(json.dumps(d,indent=2))""")
assert 'memberids.append' in s and 'calibration_full_cluster' in s
(O:=p.parent/'calibrate_cluster.py').write_text(s,encoding='utf8')
