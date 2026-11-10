from pathlib import Path
import sys,json,torch
P=Path('/mnt/sda/WangZijia/AIS/outputs/formal_20261101/single');sys.path.insert(0,str(P));import train_single as e
out={}
for n in ['tcn','gru','transformer','itransformer','patchtst','mamba','neuralcde','graph']:
 model=e.Model(n);count=lambda m:sum(p.numel() for p in m.parameters());out[n]={'total_registered':count(model),'backbone_including_binary_head':count(model.backbone),'field_adapter':count(model.adapter),'type_head':count(model.typehead),'physical_controller':count(model.ctrl),'scalar_alpha':model.alpha.numel(),'note':'registered parameters; raw C0 bypasses field adapter, type head and controller; single graph has reference node disabled'}
(P/'parameter_inventory.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
