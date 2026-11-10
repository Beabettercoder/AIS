"""Freeze 48 source/data/checkpoint hashes before any heldout test pass."""
import argparse,json,sys
from pathlib import Path
import torch
from recent_late_provenance import source_files,sha,verify_test_rho
from train_recent_late_bridge import old,fingerprint

METHODS=('tft_gcn','simad','dmrad','carla')
ARMS=('raw','causal_bundle','f0_bundle','f1_bundle')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='/home/ps/AIS_published_detectors_20261107/recent_original_late');p.add_argument('--epochs',type=int,default=25);p.add_argument('--seeds',default='0,1,2');a=p.parse_args()
    root=Path(a.out);manifest_path=root/'frozen_manifest.json'
    if manifest_path.exists():raise FileExistsError('Already frozen: '+str(manifest_path))
    seeds=[int(x) for x in a.seeds.split(',')];tr=old.load('train');va=old.load('val')
    train_hash,val_hash=fingerprint(tr),fingerprint(va)
    gate_path=root/'bridge_gate_lock.json'
    gate_sha=sha(gate_path);gate=json.loads(gate_path.read_text(encoding='utf8'))
    for name,digest in gate['evidence_sha256'].items():
        if sha(name)!=digest:raise ValueError('bridge evidence changed since training: '+name)
    sources={m:source_files(m,include_test=True) for m in METHODS}
    train_sources={m:source_files(m,include_test=False) for m in METHODS};runs={};errors=[]
    for method in METHODS:
      for seed in seeds:
        hashes=[];setting=[]
        for arm in ARMS:
          key=f'{method}/{arm}/seed{seed}';folder=root/'runs'/key
          try:
            last=folder/'last.pt';best=folder/'best_val.pt';protocol=folder/'protocol.json';history=folder/'history.json'
            spec=json.loads(protocol.read_text(encoding='utf8'));hist=json.loads(history.read_text(encoding='utf8'))
            l=torch.load(last,map_location='cpu',weights_only=False);b=torch.load(best,map_location='cpu',weights_only=False)
            if spec['smoke'] or spec['train_count']!=7840 or spec['validation_count']!=1680:raise ValueError('not the full training/validation cohorts')
            if spec['epochs']!=a.epochs or l['epoch']!=a.epochs or len(hist)!=a.epochs or [x['epoch'] for x in hist]!=list(range(1,a.epochs+1)):raise ValueError('incomplete epoch budget')
            if not 1<=b['epoch']<=a.epochs:raise ValueError('invalid best epoch')
            best_hist=max(hist,key=lambda x:x['macro_AP'])
            if b['epoch']!=best_hist['epoch'] or abs(b['metrics']['macro_AP']-best_hist['macro_AP'])>1e-10:raise ValueError('best checkpoint disagrees with validation history')
            if any(x['train_hash']!=train_hash or x['val_hash']!=val_hash for x in (l,b)):raise ValueError('training dataset mismatch')
            trained=spec['sources']
            if trained!=train_sources[method]:raise ValueError('training source hash differs from freeze')
            if spec['method']!=method or spec['arm']!=arm or spec['seed']!=seed:raise ValueError('run identity mismatch')
            if spec['bridge_gate_sha256']!=gate_sha:raise ValueError('bridge gate mismatch')
            hashes.append(spec['init_sha256']);setting.append((spec['epochs'],spec['batch_size'],spec['lr'],spec['native_aux_weight']))
            runs[key]={'last':{'sha256':sha(last),'epoch':l['epoch']},'best_val':{'sha256':sha(best),'epoch':b['epoch']},'protocol_sha256':sha(protocol),'history_sha256':sha(history)}
          except Exception as e:errors.append({'run':key,'error':str(e)})
        if len(hashes)==len(ARMS) and (len(set(hashes))!=1 or len(set(setting))!=1):errors.append({'run':f'{method}/seed{seed}','error':'paired arms do not share initialization/training budget'})
    if errors:
        report=root/'freeze_error.json';report.write_text(json.dumps({'errors':errors},indent=2),encoding='utf8')
        raise RuntimeError(f'Not frozen: {len(errors)} errors; {report}')
    rho_audit=verify_test_rho()
    manifest={'methods':METHODS,'arms':ARMS,'seeds':seeds,'epochs':a.epochs,'train_fingerprint':train_hash,'val_fingerprint':val_hash,'test_rho_provenance':rho_audit,
              'scope':'exploratory late-bridge; no new independent test; fixed final epoch and validation-macro-AP best; evaluate both without test selection',
              'sampling':'historical padded samples and historical mask, but published backbones do not consume time/mask internally',
              'sources':sources,'runs':runs}
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    with manifest_path.open('x',encoding='utf8') as f:json.dump(manifest,f,indent=2)
    print('FROZEN',len(runs),'runs',manifest_path,'sha256',sha(manifest_path))
if __name__=='__main__':main()
