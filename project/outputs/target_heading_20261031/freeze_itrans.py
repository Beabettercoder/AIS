import hashlib,json
from pathlib import Path
r=Path('/mnt/sda/WangZijia/AIS/outputs/target_heading_20261031')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d={'parent_window_protocol_sha256':sha(r/'frozen_protocol_window.json'),'scope':'exploratory fixed source balanced ordering for all 4 itransformer arms','name':'itransformer','seeds':[0,1,2],'epochs':75,'batch':'each epoch shuffle source IDs with default_rng(300000+epoch); group each source four windows, 32 sources per batch, last 18; 64 batches; no label input','reason':'separate 40-dimensional iTransformer experiment showed prior unbalanced runs avoid collapse under same 75 epochs; not independent pre-registration','fixed_last_epoch':True,'source':sha(r/'train_target.py')}
(r/'frozen_protocol_itrans.json').write_text(json.dumps(d,indent=2))
print(sha(r/'frozen_protocol_itrans.json'))
