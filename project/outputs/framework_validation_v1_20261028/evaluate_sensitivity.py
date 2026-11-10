"""Fixed-checkpoint validation sensitivity; no fitting, model selection or test.

Run with the shared torch runtime, after all 240 final checkpoints are locked.
The temporary graph wrapper changes only softmax temperature; models.py stays
unchanged. The statistical median context operators have no temperature.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import types
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'sensitivity_results'
DATA=ROOT/'extra_data/individual_sensitivity_val.npz'
PROTOCOL=ROOT/'sensitivity_protocol.json'
ARCHES=('tcn','gru','transformer')
SEEDS=(0,1,2)
TAUS=(.05,.1,.2,.5,1.,2.)
REPRODUCE_ATOL=1e-5


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save_json(path,obj,exclusive=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


@contextmanager
def graph_temperature(model,tau):
    """Temporarily bind audited forward wrappers to both two-node GAT layers.

    Original operation: softmax(mask(leaky_relu(additive_logits))).
    Intervention:      softmax(mask(leaky_relu(additive_logits) / tau)).
    Parameters, projections, graph support, residuals and readout are unchanged.
    """
    import torch
    from torch.nn import functional as F
    assert float(tau)>0 and len(model.graph_layers)==2
    originals=[]
    def forward(layer,nodes,reference_active):
        b,t,_,h=nodes.shape
        transformed=layer.projection(nodes).reshape(b,t,2,layer.heads,layer.head_dim)
        source=(transformed*layer.source_attention).sum(-1)
        dest=(transformed*layer.destination_attention).sum(-1)
        score=F.leaky_relu(source.unsqueeze(3)+dest.unsqueeze(2),negative_slope=.2)/float(tau)
        node_valid=torch.stack((torch.ones_like(reference_active),reference_active),-1)
        adjacency=node_valid.unsqueeze(-1)&node_valid.unsqueeze(-2)
        adjacency[...,1,1]=True
        attention=score.masked_fill(~adjacency.unsqueeze(-1),-torch.inf).softmax(dim=3)
        messages=torch.einsum('btijh,btjhd->btihd',attention,transformed).reshape(b,t,2,h)
        result=layer.norm(nodes+layer.dropout(F.elu(layer.out(messages))))
        return result.masked_fill(~node_valid.unsqueeze(-1),0)
    try:
        for layer in model.graph_layers:
            originals.append((layer,layer.forward));layer.forward=types.MethodType(forward,layer)
        yield
    finally:
        for layer,original in originals:layer.forward=original


def selfcheck():
    import torch
    from models import make_model
    torch.set_num_threads(2);torch.manual_seed(2718)
    model=make_model('graph_temporal',12).eval();x=torch.randn(4,20,12);x[:,:,-1]=1
    mask=torch.ones((4,20),dtype=torch.bool);times=torch.arange(20)[None].expand(4,-1).float()*60
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        ref=model(x,mask,times)['logits']
        with graph_temperature(model,1):same=model(x,mask,times)['logits']
        with graph_temperature(model,.05):changed=model(x,mask,times)['logits']
        restored=model(x,mask,times)['logits']
    assert torch.equal(ref,same) and torch.equal(ref,restored)
    assert not torch.equal(ref,changed)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())
    print(json.dumps({'tau1_exact_original':True,'wrapper_restored':True,'weights_unchanged':True,
                      'tau005_vs1_max_logit_difference':float((changed-ref).abs().max())}),flush=True)


def view_specs(protocol):
    raw=protocol['view_specs']
    specs=[dict(v,name=k) for k,v in raw.items()] if isinstance(raw,dict) else [dict(v) for v in raw]
    family_map={'radius_nm':'radius','e0_mps':'e0','sigma_mps':'sigma','heading_deg':'heading'}
    for spec in specs:
        spec['source_family']=spec['family'];spec['family']=family_map.get(spec['family'],spec['family'])
    return specs


def coverage(data,view):
    mask=data['mask'].astype(bool);x=data[view+'__C3'];valid=(x[...,-1]>0)&mask
    key=view+'__candidate_count'
    count=np.asarray(data[key],float)[mask] if key in data else None
    supported_count=np.asarray(data[key],float)[valid] if key in data else None
    return {'reference_valid_fraction':float(valid.sum()/mask.sum()),
            'candidate_count_mean_valid_target_steps':float(count.mean()) if count is not None else None,
            'candidate_count_median_valid_target_steps':float(np.median(count)) if count is not None else None,
            'candidate_count_max_valid_target_steps':float(count.max()) if count is not None else None,
            'actual_member_mean_when_reference_valid':float(supported_count.mean()) if supported_count is not None and len(supported_count) else None,
            'actual_member_median_when_reference_valid':float(np.median(supported_count)) if supported_count is not None and len(supported_count) else None,
            'valid_target_steps':int(mask.sum()),'reference_valid_steps':int(valid.sum())}


def run(args):
    # These checks precede reading checkpoint predictions or computing AP.
    lock=json.loads((ROOT/'checkpoint_lock.json').read_text())
    core_protocol=json.loads((ROOT/'protocol.json').read_text())
    expected=len(core_protocol['tasks'])*len(core_protocol['backbones'])*len(core_protocol['seeds'])*len(core_protocol['configurations'])
    assert lock['count']==expected==240
    assert lock['code_lock_sha256']==sha(ROOT/'code_lock.json')
    code_lock=json.loads((ROOT/'code_lock.json').read_text())
    for n in ('models.py','train_study.py','protocol.json','data/individual/val.npz'):
        assert sha(ROOT/n)==(code_lock['data_files'][n] if n.startswith('data/') else code_lock['files'][n])
    protocol=json.loads(PROTOCOL.read_text());specs=view_specs(protocol)
    audit_path=ROOT/'sensitivity_audit.json';audit=json.loads(audit_path.read_text())
    assert audit['status']=='complete_validation_only_inference_sensitivity'
    assert audit['output_sha256']==sha(DATA) and audit['protocol_sha256']==sha(PROTOCOL)
    assert len(specs)==21 and len({s['name'] for s in specs})==21
    assert {s['family'] for s in specs}=={'baseline','topk','radius','e0','sigma','heading'}
    with np.load(DATA,allow_pickle=False) as archive:
        for s in specs:assert s['name']+'__C3' in archive.files
    import torch
    from models import make_model
    from train_study import load_npz,normalize,get_mask,times_array,metrics
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    data=load_npz(DATA);core=load_npz(ROOT/'data/individual/val.npz')
    for k in ('window_id','source_id','y','region','mask','time_s'):
        np.testing.assert_array_equal(data[k],core[k])
    np.testing.assert_array_equal(data['baseline__C3'],core['base__C3'])
    defaults={'radius':8.,'e0':2.5,'sigma':2.,'heading':45.}
    for spec in specs:
        if spec['family'] in defaults and float(spec['value'])==defaults[spec['family']]:
            np.testing.assert_array_equal(data[spec['name']+'__C3'],core['base__C3'])
    manifest={'study':'fixed-checkpoint validation inference sensitivity','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'data_split':'val','task':'individual','arm':'C3','weights':'fixed final checkpoints; no retraining',
              'architecture_input_grid':list(ARCHES),'graph_temperature_architecture':'graph_temporal','seeds':list(SEEDS),
              'temperature_values':list(TAUS),'temperature_definition':'both custom target-reference GAT layers: divide additive attention logits after LeakyReLU by tau before masked softmax',
              'temperature_not_applicable_to':['median aggregation','historical CDG branch'],'no_hyperparameter_selection':True,
              'no_test_data_or_scores':True,'normalization':'exact saved training-only checkpoint normalizer; no sensitivity-set fitting',
              'frozen_defaults':{**defaults,'topk':'all eligible members','temperature':1.},
              'source_limit':'candidate archive restricted to original <=8nm and <=45deg; narrower filtering only',
              'data_sha256':sha(DATA),'sensitivity_protocol_sha256':sha(PROTOCOL),'checkpoint_lock_sha256':sha(ROOT/'checkpoint_lock.json'),
              'sensitivity_audit_sha256':sha(audit_path),
              'core_validation_sha256':sha(ROOT/'data/individual/val.npz'),'models_sha256':sha(ROOT/'models.py'),
              'evaluator_sha256':sha(__file__),'device':str(device),'torch':torch.__version__,'view_specs':specs,
              'tau1_reproduction_tolerance_max_abs_score':REPRODUCE_ATOL,'timing_policy':'no claimed efficiency benchmark'}
    manifest_path=OUT/'evaluation_manifest.json'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text())
        for key in ('data_sha256','sensitivity_protocol_sha256','checkpoint_lock_sha256','evaluator_sha256','device'):
            assert existing[key]==manifest[key],key
        manifest=existing
    else:save_json(manifest_path,manifest,exclusive=True)
    rows=[];checks=[]
    mask=torch.tensor(get_mask(core),device=device);times=torch.tensor(times_array(core),device=device)
    def predict(model,x):
        out=[]
        with torch.no_grad():
            for at in range(0,len(x),args.batch_size):
                logits=model(torch.tensor(x[at:at+args.batch_size],device=device),mask[at:at+args.batch_size],times[at:at+args.batch_size])['logits']
                out.append(logits.sigmoid().cpu().numpy())
        scores=np.concatenate(out).astype(np.float64);assert np.isfinite(scores).all();return scores
    def record(model,x,spec,arch,seed,checkpoint_sha,temperature=None):
        if (ROOT/'STOP').exists():raise SystemExit('STOP requested; completed sensitivity artifacts retained')
        folder=OUT/'predictions'/spec['name']/arch/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/'result.json';prediction_path=folder/'validation.npz'
        if result_path.exists():
            row=json.loads(result_path.read_text())
            assert row['checkpoint_sha256']==checkpoint_sha and row['evaluation_manifest_sha256']==sha(manifest_path)
            assert row['prediction_sha256']==sha(prediction_path)
            scores=load_npz(prediction_path)['score']
        else:
            if temperature is None:scores=predict(model,x)
            else:
                with graph_temperature(model,temperature):scores=predict(model,x)
            metric=metrics(data['y'],scores,data['region'])
            cov=coverage(data,'baseline' if spec['family']=='temperature' else spec['name'])
            np.savez_compressed(prediction_path,score=scores,y=data['y'],window_id=data['window_id'],source_id=data['source_id'],region=data['region'])
            row={'family':spec['family'],'view':spec['name'],'value':spec['value'],'architecture':arch,'seed':seed,'task':'individual','split':'val','arm':'C3',
                 'macro_AP':metric['macro_AP'],'pooled_AP':metric['pooled_AP'],'AUROC':metric['AUROC'],'per_region':metric['per_region'],**cov,
                 'checkpoint_sha256':checkpoint_sha,'evaluation_manifest_sha256':sha(manifest_path),'prediction_sha256':sha(prediction_path),
                 'prediction_path':str(prediction_path.relative_to(ROOT)).replace('\\','/')}
            save_json(result_path,row)
        rows.append(row);print('SENSITIVITY_DONE',spec['name'],arch,seed,flush=True)
        return scores
    for arch in ARCHES+('graph_temporal',):
        for seed in SEEDS:
            rel=f'runs/individual/{arch}/seed_{seed}/C3/final.pt';cp=ROOT/rel;checkpoint_sha=sha(cp)
            assert checkpoint_sha==lock['checkpoints'][rel]
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            model=make_model(arch,12,max_len=20,hidden=64).to(device);model.load_state_dict(saved['model']);model.eval()
            core_x,_=normalize(core,'base__C3',saved['normalization'])
            original=predict(model,core_x)
            reference=load_npz(cp.with_name('validation_base.npz'))
            np.testing.assert_array_equal(reference['window_id'],core['window_id'])
            max_core_error=float(np.max(np.abs(original-reference['score'])))
            checks.append({'architecture':arch,'seed':seed,'baseline_vs_saved_validation_max_abs_score':max_core_error})
            save_json(OUT/'baseline_reproduction_checks.json',{'device':str(device),'checks':checks,'tolerance':REPRODUCE_ATOL,
                                                             'all_passed':all(c['baseline_vs_saved_validation_max_abs_score']<=REPRODUCE_ATOL for c in checks)})
            assert max_core_error<=REPRODUCE_ATOL,(arch,seed,max_core_error)
            if arch=='graph_temporal':
                with graph_temperature(model,1):tau1=predict(model,core_x)
                np.testing.assert_array_equal(tau1,original)
                checks[-1]['tau1_vs_original_model_max_abs_score']=float(np.max(np.abs(tau1-original)))
                checks[-1]['tau1_vs_saved_validation_max_abs_score']=float(np.max(np.abs(tau1-reference['score'])))
                for tau in TAUS:
                    spec={'name':'temperature_'+str(tau).replace('.','p'),'family':'temperature','value':tau}
                    record(model,core_x,spec,arch,seed,checkpoint_sha,tau)
            else:
                for spec in specs:
                    x,_=normalize(data,spec['name']+'__C3',saved['normalization'])
                    score=record(model,x,spec,arch,seed,checkpoint_sha)
                    if spec['family']=='baseline':np.testing.assert_array_equal(score,original)
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
    save_json(OUT/'results.json',{'status':'complete','evaluation_manifest_sha256':sha(manifest_path),'rows':rows,'baseline_reproduction_checks':checks,
                                'interpretation':'validation sensitivity of frozen checkpoints; default unchanged; no test-based or validation-based hyperparameter selection'})
    table=ROOT/'tables/table_s5_sensitivity.csv';table.parent.mkdir(exist_ok=True)
    columns=['family','view','value','architecture','seed','macro_AP','pooled_AP','AUROC','reference_valid_fraction','candidate_count_mean_valid_target_steps','candidate_count_median_valid_target_steps','candidate_count_max_valid_target_steps','actual_member_mean_when_reference_valid','actual_member_median_when_reference_valid']
    with table.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in rows)
    print('SENSITIVITY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['selfcheck','evaluate'],default='evaluate');parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=256);args=parser.parse_args()
    (selfcheck if args.mode=='selfcheck' else lambda:run(args))()
