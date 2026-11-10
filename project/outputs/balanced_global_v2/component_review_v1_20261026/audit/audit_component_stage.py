"""Independent train/validation-only audit and complete component summaries.

This module does not import training metrics or selection code. Single-class
strata retain counts and classification rates but have no AP/AUROC estimate.
The reported differences describe validation development pipelines, including
their separate training-only normalizers, and are not held-out test estimates.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

FIELDS = ('window_id', 'base_window_id', 'mmsi', 'region', 'y', 'behavior_type',
          'noise_type', 'noise_scope', 'target_noise_effective',
          'neighbor_noise_effective', 'observation_noise_effective')
BOOL_FIELDS = ('target_noise_effective', 'neighbor_noise_effective', 'observation_noise_effective')
METRICS = ('AP', 'AUROC', 'F1', 'precision', 'recall')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def jwrite(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Unrecognized Boolean value {value!r}')


def threshold_from_counts(y, score):
    """Independent score-sorted F1 maximization, highest threshold on ties."""
    y = np.asarray(y, np.int64)
    score = np.asarray(score, np.float64)
    require(set(y) == {0, 1}, 'Threshold selection requires both classes')
    order = np.argsort(-score, kind='stable')
    ss, yy = score[order], y[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    f1 = 2 * tp / (int(y.sum()) + ends + 1)
    winner = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[0]
    return float(ss[ends[winner]]), float(f1[winner])


def metric_set(y, score, threshold):
    y = np.asarray(y, np.int64)
    score = np.asarray(score, np.float64)
    pred = score >= threshold
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum()); fn = int(((y == 1) & ~pred).sum())
    n = len(y); p = int(y.sum()); both = bool(0 < p < n)
    return dict(samples=n, positive_samples=p, negative_samples=n-p,
                positive_fraction=float(p/n) if n else None,
                AP=float(average_precision_score(y, score)) if both else None,
                AUROC=float(roc_auc_score(y, score)) if both else None,
                F1=float(2*tp/(2*tp+fp+fn)) if n and (2*tp+fp+fn) else (0. if n else None),
                precision=float(tp/(tp+fp)) if tp+fp else (0. if n else None),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                FPR=float(fp/(fp+tn)) if fp+tn else None,
                predicted_positive_rate=float(pred.mean()) if n else None,
                ranking_metric_status='both_classes' if both else ('empty' if n == 0 else 'single_class_undefined'),
                confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn))


def region_metrics(y, score, regions, threshold):
    regions = np.asarray(regions, str)
    per = {r: metric_set(y[regions == r], score[regions == r], threshold) for r in sorted(set(regions))}
    macro, contributors = {}, {}
    for key in METRICS + ('FPR',):
        valid = [v[key] for v in per.values() if v[key] is not None]
        macro[key] = float(np.mean(valid)) if valid else None
        contributors[key] = len(valid)
    return dict(macro=macro, metric_contributors=contributors, per_region=per,
                pooled=metric_set(y, score, threshold))


def load_inputs(data, protocol):
    """Only these two literal split names can be opened."""
    chunks = {'train': [], 'val': []}; proof = []
    features = protocol['feature_names']
    for split in ('train', 'val'):
        for region in protocol['regions']:
            path = data / region / f'{split}.npz'
            with np.load(path, allow_pickle=False) as z:
                record = {k: np.asarray(z[k]) for k in FIELDS}
                for key in BOOL_FIELDS:
                    record[key] = np.asarray([as_bool(v) for v in record[key]], bool)
                record['mask'] = np.asarray(z['mask'], bool)
                record['context_valid'] = np.asarray(z['context_valid'], bool)
                record['features'] = {k: np.asarray(z[k], np.float32) for k in features}
                n = len(record['y']); mask = record['mask']
                require(n == protocol['samples'][f'{split}_per_region'], f'{path}: wrong sample count')
                require(mask.shape == (n, 20), f'{path}: wrong mask')
                require(record['context_valid'].shape == mask.shape, f'{path}: wrong original context-valid shape')
                require(np.all(mask.sum(1) > 0), f'{path}: empty sequence')
                require(np.array_equal(mask, np.arange(20)[None, :] < mask.sum(1)[:, None]), f'{path}: non-prefix mask')
                for key, x in record['features'].items():
                    require(x.shape == (n, 20, 16) and np.isfinite(x).all(), f'{path}: invalid {key}')
                    require(np.all(x[~mask] == 0), f'{path}: nonzero padding {key}')
                require(np.all(record['region'].astype(str) == region), f'{path}: wrong region')
                chunks[split].append(record)
            proof.append(dict(path=str(path), split=split, region=region, sha256=sha(path), samples=n))
    result = {}
    for split, parts in chunks.items():
        result[split] = {k: np.concatenate([p[k] for p in parts]) for k in FIELDS + ('mask','context_valid')}
        result[split]['features'] = {k: np.concatenate([p['features'][k] for p in parts]) for k in features}
        for key in ('window_id', 'base_window_id'):
            require(len(set(result[split][key].astype(str))) == len(result[split]['y']), f'Duplicate {split} {key}')
    for key in ('window_id', 'base_window_id', 'mmsi'):
        require(not(set(result['train'][key].astype(str)) & set(result['val'][key].astype(str))), f'{key} train/val leakage')
    return result, proof


def normalizers(train, features):
    answer = {}
    for feature in features:
        values = train['features'][feature].astype(np.float64)[train['mask']]
        answer[feature] = dict(mean=values.mean(0).astype(np.float32),
                               std=np.maximum(values.std(0), 1e-6).astype(np.float32), valid_steps=len(values))
    return answer


def load_csv_predictions(path, val):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == len(val['y']), f'{path}: prediction row count')
    for key in FIELDS:
        got = np.asarray([r[key] for r in rows])
        expected = val[key]
        if key in BOOL_FIELDS:
            got = np.asarray([as_bool(v) for v in got]); expected = expected.astype(bool)
        elif key == 'y':
            got = got.astype(np.int64); expected = expected.astype(np.int64)
        else:
            got = got.astype(str); expected = expected.astype(str)
        require(np.array_equal(got, expected), f'{path}: {key} mismatch')
    scores = np.asarray([r['score'] for r in rows], np.float64)
    thresholds = np.asarray([r['threshold'] for r in rows], np.float64)
    require(np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1)), f'{path}: invalid scores')
    require(np.all(thresholds == thresholds[0]), f'{path}: varying thresholds')
    if 'prediction' in rows[0]:
        require(np.array_equal([int(r['prediction']) for r in rows], scores >= thresholds[0]), f'{path}: prediction mismatch')
    return scores, float(thresholds[0])


def strata(val, score, threshold):
    outputs = {}
    for key in ('noise_type', 'noise_scope', 'target_noise_effective', 'neighbor_noise_effective', 'observation_noise_effective'):
        values = val[key].astype(str)
        outputs[key] = {}
        for value in sorted(set(values)):
            ii = values == value
            outputs[key][value] = region_metrics(val['y'][ii], score[ii], val['region'][ii], threshold)
    target = val['target_noise_effective'].astype(bool)
    neighbor = val['neighbor_noise_effective'].astype(bool)
    outputs['actual_corruption_joint'] = {}
    for name, ii in [('neither', ~target & ~neighbor), ('target_only', target & ~neighbor),
                     ('neighbors_only', ~target & neighbor), ('both', target & neighbor)]:
        outputs['actual_corruption_joint'][name] = region_metrics(val['y'][ii], score[ii], val['region'][ii], threshold)
    return outputs


def exploratory_support_strata(val,score,threshold):
    """Post-protocol explanatory strata; fixed boundaries, never model selection."""
    mask=val['mask'].astype(bool)
    ratio=(val['context_valid'].astype(bool) & mask).sum(1)/mask.sum(1)
    partitions={'zero_valid_context':ratio==0,
                'positive_below_half':(ratio>0)&(ratio<.5),
                'at_least_half':ratio>=.5}
    require(np.all(sum(x.astype(int) for x in partitions.values())==1),'Support strata must partition every window')
    return dict(scope='Exploratory explanatory stratification added after protocol freeze. Shared original context-valid window fractions, not variant-specific validity. Never used for model selection or predeclared main contrasts.',
                definition='fraction=sum(original context_valid AND target mask)/sum(target mask); bins {0}, (0,0.5), [0.5,1]',
                groups={name:region_metrics(val['y'][ii],score[ii],val['region'][ii],threshold) for name,ii in partitions.items()})


def compare_metrics(actual, recorded, name):
    for metric in METRICS:
        require(math.isclose(actual['macro'][metric], recorded['macro'][metric], rel_tol=0, abs_tol=1e-10), f'{name}: macro {metric}')
    for region, values in actual['per_region'].items():
        for metric in METRICS:
            require(math.isclose(values[metric], recorded['per_region'][region][metric], rel_tol=0, abs_tol=1e-10), f'{name}/{region}: {metric}')


def distribution(values):
    arr = np.asarray(values, float)
    return dict(values=[float(x) for x in arr], mean=float(arr.mean()) if len(arr) else None,
                sample_std=float(arr.std(ddof=1)) if len(arr)>1 else None, n=len(arr))


def state_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous().numpy()
        for value in (name.encode('utf-8'),str(x.dtype).encode('ascii'),json.dumps(list(x.shape),separators=(',',':')).encode('ascii')):
            h.update(value+b'\0')
        h.update(x.tobytes(order='C'))
    return h.hexdigest()


def expected_order_hash(n,seed,epochs):
    rng=np.random.default_rng(seed+100);h=hashlib.sha256();per=[]
    for epoch in range(1,epochs+1):
        payload=np.asarray([epoch],dtype='<i8').tobytes()+np.asarray(rng.permutation(n),dtype='<i8').tobytes()
        h.update(payload);per.append(hashlib.sha256(payload).hexdigest())
    return h.hexdigest(),per


def identifier_hash(data):
    records=[[str(data[k][i]) for k in ('window_id','base_window_id','mmsi','region')]+[int(data['y'][i])] for i in range(len(data['y']))]
    return hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def check_worker_configuration(path,proof,protocol,protocol_hash,training_dir,data):
    config=json.loads(path.read_text())
    require(config['protocol_sha256']==protocol_hash and config['training']==protocol['training'], f'{path}: protocol')
    require(config['regions']==protocol['regions'] and config['features']==protocol['feature_names'],f'{path}: regions/features')
    require(config['driver_sha256']==sha(training_dir/'train_components.py'),f'{path}: driver changed')
    require(config['vendor_sha256']==sha(training_dir/'vendor/pilot_train_frozen.py'),f'{path}: vendor changed')
    expected={(p['region'],p['split']):(p['sha256'],p['samples']) for p in proof}
    actual={(p['region'],p['split']):(p['sha256'],p['samples']) for p in config['sources']}
    require(actual==expected and len(config['sources'])==len(proof),f'{path}: source list')
    for source in config['sources']:
        require(source['split'] in ('train','val') and Path(source['path']).name==source['split']+'.npz',f'{path}: forbidden source split')
    for split in ('train','val'):
        require(config['input_signatures'][split]==identifier_hash(data[split]),f'{path}: input signature')
    with open(path.parent/'input_manifest.csv',newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==sum(len(data[s]['y']) for s in ('train','val')),f'{path}: input manifest length')
    for split in ('train','val'):
        rows_split=[r for r in rows if r['split']==split]
        require(len(rows_split)==len(data[split]['y']),f'{path}: manifest split')
        for i,r in enumerate(rows_split):
            for key in ('window_id','base_window_id','mmsi','region','y'):
                require(str(r[key])==str(data[split][key][i]),f'{path}: manifest row {i}/{key}')
    return sha(path)


def summaries(results, protocol):
    table = []; contrasts = []; interaction = []
    index = {(r['backbone'], r['feature_group'], r['seed']): r for r in results}
    seeds = protocol['training']['seeds']
    for backbone in protocol['backbones']:
        for feature in protocol['feature_names']:
            rr = [index[(backbone, feature, s)] for s in seeds if (backbone, feature, s) in index]
            table.append(dict(backbone=backbone, feature_group=feature, seeds=[r['seed'] for r in rr],
                              **{m: distribution([r['validation']['macro'][m] for r in rr]) for m in METRICS}))
        for contrast in protocol['predeclared_contrasts']:
            paired = [(s, index[(backbone, contrast['from'], s)], index[(backbone, contrast['to'], s)])
                      for s in seeds if (backbone, contrast['from'], s) in index and (backbone, contrast['to'], s) in index]
            contrasts.append(dict(backbone=backbone, **contrast, seeds=[s for s, _, _ in paired],
                                  **{m: distribution([b['validation']['macro'][m]-a['validation']['macro'][m] for _,a,b in paired]) for m in METRICS}))
        vals=[]; used=[]
        for s in seeds:
            names=('legacy_full', 'reliable_gate_target', 'reliable_gate_residual', 'reliable_raw')
            if all((backbone,f,s) in index for f in names):
                m=[index[(backbone,f,s)]['validation']['macro']['AP'] for f in names]
                vals.append(m[0]-m[1]-m[2]+m[3]); used.append(s)
        interaction.append(dict(backbone=backbone, seeds=used, AP_interaction=distribution(vals),
                                formula='AP(legacy_full)-AP(reliable_gate_target)-AP(reliable_gate_residual)+AP(reliable_raw)',
                                scope='Full training pipelines including feature-specific train-only scalers and validation checkpoint selection'))
    return dict(all_configurations=table, paired_contrasts=contrasts, gating_2x2_interaction=interaction)


def stratified_summaries(results):
    groups={}
    for run in results:
        sources=dict(run['validation_strata'])
        sources['exploratory_support_strata']=run['exploratory_support_strata']['groups']
        for stratifier,levels in sources.items():
            for level,metrics in levels.items():
                key=(run['backbone'],run['feature_group'],stratifier,level)
                groups.setdefault(key,[]).append((run['seed'],metrics))
    rows=[]
    for (backbone,feature,stratifier,level),entries in sorted(groups.items()):
        entries=sorted(entries)
        sample_counts={x['pooled']['samples'] for _,x in entries}
        require(len(sample_counts)==1,'Stratum membership changed across seeds')
        row=dict(backbone=backbone,feature_group=feature,stratifier=stratifier,level=level,
                 exploratory=(stratifier=='exploratory_support_strata'),seeds=[s for s,_ in entries],
                 windows=entries[0][1]['pooled']['samples'],positive_windows=entries[0][1]['pooled']['positive_samples'],
                 per_region_counts={r:{k:v[k] for k in ('samples','positive_samples','negative_samples','ranking_metric_status')}
                                    for r,v in entries[0][1]['per_region'].items()})
        for metric in METRICS+('FPR',):
            values=[x['macro'][metric] for _,x in entries if x['macro'][metric] is not None]
            row[metric]=distribution(values)
            row[metric]['contributing_regions_per_seed']=[x['metric_contributors'][metric] for _,x in entries]
        rows.append(row)
    return dict(scope='Complete validation strata for all configurations; no missing-class metric imputation. FPR uses each run own validation-max-F1 threshold, not matched FAR.',
                exploratory_support_scope='Original-context-support groups were added after protocol freeze for explanation only, never for model selection.',rows=rows)


def audit_run(path, backbone, feature, seed, data, scaler, protocol, protocol_hash,config_hash,vendor):
    result = json.loads((path/'result.json').read_text())
    require(result['backbone']==backbone and result['feature_group']==feature and result['seed']==seed, f'{path}: run identity')
    require(result['protocol_sha256']==protocol_hash, f'{path}: protocol changed')
    require(result['configuration_sha256']==config_hash,f'{path}: worker configuration hash')
    require(result['input_channels']==16, f'{path}: wrong input size')
    require(result['epochs_budget']==protocol['training']['epochs'], f'{path}: wrong budget')
    require('test' not in result, f'{path}: unexpected test result')
    for name, expected_hash in result['output_hashes'].items():
        require(Path(name).name==name and 'test' not in name.lower(), f'{path}: unexpected output artifact {name}')
        require(sha(path/name)==expected_hash, f'{path}: output hash changed {name}')
    completion=json.loads((path/'completion.json').read_text())
    require(completion.get('result_sha256')==sha(path/'result.json'), f'{path}: incomplete or changed result')
    with np.load(path/'normalizer.npz', allow_pickle=False) as z:
        for key in ('mean','std'):
            require(np.array_equal(z[key], scaler[key]), f'{path}: normalizer {key} not exact training-only fit')
        require(int(z['valid_steps'])==scaler['valid_steps'], f'{path}: normalizer count')
    score, csv_threshold=load_csv_predictions(path/'val_predictions.csv',data['val'])
    threshold, pooled_f1=threshold_from_counts(data['val']['y'],score)
    require(abs(threshold-result['threshold'])<1e-12 and csv_threshold==threshold, f'{path}: threshold')
    actual=region_metrics(data['val']['y'],score,data['val']['region'],threshold)
    compare_metrics(actual,result['validation'],str(path))
    with np.load(path/'epoch_validation_predictions.npz',allow_pickle=False) as z:
        epoch_scores=np.asarray(z['scores'])
        require(np.array_equal(z['window_id'].astype(str),data['val']['window_id'].astype(str)), f'{path}: epoch IDs')
    epochs=protocol['training']['epochs']
    require(epoch_scores.shape==(epochs,len(score)), f'{path}: epoch matrix shape')
    require(np.isfinite(epoch_scores).all(), f'{path}: nonfinite epoch matrix')
    history=result['history']; require(len(history)==epochs,f'{path}: history length')
    independent_history=[]
    for j,p in enumerate(epoch_scores):
        require(history[j]['epoch']==j+1 and np.isfinite(history[j]['train_loss']),f'{path}: history sequence')
        per=[]
        for region in protocol['regions']:
            ii=data['val']['region'].astype(str)==region
            per.append(average_precision_score(data['val']['y'][ii],p[ii]))
        ap=float(np.mean(per)); independent_history.append(ap)
        require(abs(ap-history[j]['validation_macro_AP'])<1e-10,f'{path}: epoch {j+1} AP')
    best=int(np.argmax(independent_history))+1
    require(result['selected_epoch']==best,f'{path}: selected epoch')
    require(np.allclose(score,epoch_scores[best-1],rtol=0,atol=1e-7),f'{path}: selected scores')
    order_hash,epoch_hashes=expected_order_hash(len(data['train']['y']),seed,epochs)
    require(order_hash==result['batch_order_sha256'] and epoch_hashes==result['epoch_batch_order_sha256'],f'{path}: batch ordering')
    import torch
    torch.manual_seed(seed)
    model=vendor.BackboneClassifier(backbone,16,protocol['training']['hidden'],protocol['training']['dropout'])
    require(state_hash(model.state_dict())==result['initial_state_sha256'],f'{path}: initialized state')
    require(sum(p.numel() for p in model.parameters())==result['parameters'],f'{path}: parameter count')
    checkpoint=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
    require(state_hash(checkpoint['state_dict'])==result['selected_state_sha256'],f'{path}: selected state')
    for key in ('mean','std'):
        require(np.array_equal(checkpoint['normalizer_'+key].numpy(),scaler[key]),f'{path}: checkpoint scaler')
    require(checkpoint['selected_epoch']==best and checkpoint['threshold']==threshold,f'{path}: checkpoint selection metadata')
    require(checkpoint['protocol_sha256']==protocol_hash and checkpoint['configuration_sha256']==config_hash,f'{path}: checkpoint protocol')
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    vx=(data['val']['features'][feature]-scaler['mean'])/scaler['std'];vx[data['val']['mask']==False]=0
    vx=torch.from_numpy(vx);vm=torch.from_numpy(data['val']['mask']);replayed=[]
    with torch.inference_mode():
        for lo in range(0,len(score),protocol['training']['batch_size']):
            hi=lo+protocol['training']['batch_size']
            replayed.append(torch.sigmoid(model(vx[lo:hi],vm[lo:hi])).numpy())
    replayed=np.concatenate(replayed)
    require(np.allclose(replayed,score,rtol=0,atol=1e-7),f'{path}: checkpoint forward replay')
    return dict(backbone=backbone,feature_group=feature,seed=seed,parameters=result['parameters'],
                selected_epoch=best,threshold=threshold,validation=actual,
                initial_state_sha256=result['initial_state_sha256'],batch_order_sha256=result['batch_order_sha256'],
                epoch_batch_order_sha256=result['epoch_batch_order_sha256'],
                protocol_sha256=protocol_hash,configuration_sha256=result['configuration_sha256'],
                result_sha256=sha(path/'result.json'),checkpoint_sha256=sha(path/'checkpoint.pt'),
                val_predictions_sha256=sha(path/'val_predictions.csv'),
                independent_epoch_validation_macro_AP=independent_history,
                checkpoint_replay_max_absolute_error=float(np.max(np.abs(replayed-score))),
                validation_strata=strata(data['val'],score,threshold),
                exploratory_support_strata=exploratory_support_strata(data['val'],score,threshold))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training-code',type=Path,required=True,help='Frozen training folder containing vendor/')
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    require(not args.out.exists(),'Refusing to overwrite audit output')
    protocol=json.loads(args.protocol.read_text()); ph=sha(args.protocol)
    data,proof=load_inputs(args.data,protocol); scales=normalizers(data['train'],protocol['feature_names'])
    spec=importlib.util.spec_from_file_location('audit_backbone_vendor',args.training_code/'vendor/pilot_train_frozen.py')
    vendor=importlib.util.module_from_spec(spec);spec.loader.exec_module(vendor)
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    results=[];missing=[]
    for b in protocol['backbones']:
        worker_config=args.runs/b/'configuration.json'
        if not worker_config.exists() and args.allow_partial:
            missing.extend(str(args.runs/b/f'seed_{s}'/f) for s in protocol['training']['seeds'] for f in protocol['feature_names'])
            continue
        config_hash=check_worker_configuration(worker_config,proof,protocol,ph,args.training_code,data)
        for s in protocol['training']['seeds']:
            for f in protocol['feature_names']:
                path=args.runs/b/f'seed_{s}'/f
                if not (path/'completion.json').exists():
                    missing.append(str(path));continue
                r=audit_run(path,b,f,s,data,scales[f],protocol,ph,config_hash,vendor);results.append(r)
                print('AUDITED',b,s,f,flush=True)
    require(args.allow_partial or not missing,f'Missing {len(missing)} completed runs')
    for b in protocol['backbones']:
        for s in protocol['training']['seeds']:
            rr=[r for r in results if r['backbone']==b and r['seed']==s]
            for key in ('initial_state_sha256','batch_order_sha256','parameters','configuration_sha256'):
                require(len({r[key] for r in rr})<=1,f'{b}/seed{s}: pairing failure {key}')
            require(len({tuple(r['epoch_batch_order_sha256']) for r in rr})<=1,f'{b}/seed{s}: epoch batch-order pairing')
    args.out.mkdir(parents=True)
    summary=summaries(results,protocol)
    summary.update(status='partial' if missing else 'passed',scope='Validation-only component development. All seeds and variants retained; no new test access.',
                   protocol_sha256=ph,completed_runs=len(results),expected_runs=len(protocol['backbones'])*len(protocol['training']['seeds'])*len(protocol['feature_names']),
                   auditor_sha256=sha(Path(__file__)),
                   normalizer_scope='Independent exact training-valid-step mean/std recomputation for all variants.',
                   single_class_policy='AP and AUROC null for single-class strata; report positive prevalence and class counts. No imputation.',
                   variance_scope='Sample standard deviation across training seeds (ddof=1), not an independent-data confidence interval.',
                   missing_runs=missing)
    jwrite(args.out/'independent_component_metric_audit.json',dict(status=summary['status'],source_files=proof,runs=results))
    jwrite(args.out/'component_summary.json',summary)
    jwrite(args.out/'component_stratified_summary.json',stratified_summaries(results))
    rows=[]
    for r in results:
        rows.append({k:r[k] for k in ('backbone','feature_group','seed','parameters','selected_epoch','threshold')}|r['validation']['macro'])
    with open(args.out/'per_seed_validation_metrics.csv','w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]) if rows else ['backbone']);writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=summary['status'],runs=len(results),out=str(args.out))),flush=True)


if __name__=='__main__':
    main()
