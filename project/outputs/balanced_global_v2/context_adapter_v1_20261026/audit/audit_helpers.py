"""Independent metric, masking and tensor identity helpers for adapter audit."""
from pathlib import Path
import csv, hashlib, json, math
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
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Unrecognized Boolean value {value!r}')


def threshold_from_counts(y, score):
    y = np.asarray(y, np.int64); score = np.asarray(score, np.float64)
    require(set(y) == {0, 1}, 'Threshold requires both classes')
    order = np.argsort(-score, kind='stable'); ss, yy = score[order], y[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp = np.cumsum(yy)[ends]
    f1 = 2 * tp / (int(y.sum()) + ends + 1)
    winner = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[0]
    return float(ss[ends[winner]]), float(f1[winner])


def metric_set(y, score, threshold):
    y = np.asarray(y, np.int64); score = np.asarray(score, np.float64)
    pred = score >= threshold
    tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum())
    tn=int(((y==0)&~pred).sum()); fn=int(((y==1)&~pred).sum())
    n=len(y); positive=int(y.sum()); both=bool(0<positive<n)
    return dict(samples=n, positive_samples=positive, negative_samples=n-positive,
                positive_fraction=float(positive/n) if n else None,
                AP=float(average_precision_score(y,score)) if both else None,
                AUROC=float(roc_auc_score(y,score)) if both else None,
                F1=float(2*tp/(2*tp+fp+fn)) if n and 2*tp+fp+fn else (0. if n else None),
                precision=float(tp/(tp+fp)) if tp+fp else (0. if n else None),
                recall=float(tp/(tp+fn)) if tp+fn else None,
                FPR=float(fp/(fp+tn)) if fp+tn else None,
                confusion=dict(tp=tp,fp=fp,tn=tn,fn=fn),
                ranking_metric_status='both_classes' if both else ('empty' if n==0 else 'single_class_undefined'))


def region_metrics(y, score, regions, threshold):
    regions=np.asarray(regions,str)
    per={r:metric_set(y[regions==r],score[regions==r],threshold) for r in sorted(set(regions))}
    macro={}; contributors={}
    for key in METRICS+('FPR',):
        vals=[v[key] for v in per.values() if v[key] is not None]
        macro[key]=float(np.mean(vals)) if vals else None; contributors[key]=len(vals)
    return dict(macro=macro, metric_contributors=contributors, per_region=per, pooled=metric_set(y,score,threshold))


def compare_metrics(actual, recorded, name):
    for metric in METRICS:
        require(math.isclose(actual['macro'][metric],recorded['macro'][metric],rel_tol=0,abs_tol=1e-10),f'{name}: macro {metric}')
    for region, values in actual['per_region'].items():
        for metric in METRICS:
            require(math.isclose(values[metric],recorded['per_region'][region][metric],rel_tol=0,abs_tol=1e-10),f'{name}/{region}: {metric}')


def distribution(values):
    a=np.asarray(values,float)
    return dict(values=a.tolist(),mean=float(a.mean()) if len(a) else None,
                sample_std=float(a.std(ddof=1)) if len(a)>1 else None,n=len(a))


def state_hash(state):
    h=hashlib.sha256()
    for name in sorted(state):
        x=state[name].detach().cpu().contiguous().numpy()
        for item in (name.encode(),str(x.dtype).encode(),json.dumps(list(x.shape),separators=(',',':')).encode()):
            h.update(item+b'\0')
        h.update(x.tobytes(order='C'))
    return h.hexdigest()


def load_csv_predictions(path, data):
    with open(path,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    require(len(rows)==len(data['y']),f'{path}: row count')
    for key in FIELDS:
        got=np.asarray([r[key] for r in rows]); expected=data[key]
        if key in BOOL_FIELDS:
            got=np.asarray([as_bool(v) for v in got]); expected=np.asarray(expected,bool)
        elif key=='y':got=got.astype(np.int64);expected=expected.astype(np.int64)
        else:got=got.astype(str);expected=expected.astype(str)
        require(np.array_equal(got,expected),f'{path}: {key}')
    scores=np.asarray([r['score'] for r in rows],float)
    thresholds=np.asarray([r['threshold'] for r in rows],float)
    require(np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)),f'{path}: scores')
    require(np.all(thresholds==thresholds[0]),f'{path}: varying thresholds')
    require(np.array_equal([int(r['prediction']) for r in rows],scores>=thresholds[0]),f'{path}: predictions')
    return scores,float(thresholds[0])


def strata(data, scores, threshold):
    out={}
    for key in ('noise_type','noise_scope')+BOOL_FIELDS:
        values=data[key].astype(str)
        out[key]={value:region_metrics(data['y'][values==value],scores[values==value],data['region'][values==value],threshold) for value in sorted(set(values))}
    target=data['target_noise_effective'].astype(bool); neighbor=data['neighbor_noise_effective'].astype(bool)
    out['actual_corruption_joint']={name:region_metrics(data['y'][ii],scores[ii],data['region'][ii],threshold)
        for name,ii in [('neither',~target&~neighbor),('target_only',target&~neighbor),('neighbors_only',~target&neighbor),('both',target&neighbor)]}
    ratio=data['common_support'].sum(1)/data['mask'].sum(1)
    bins={'zero_support':ratio==0,'positive_below_half':(ratio>0)&(ratio<.5),'at_least_half':ratio>=.5}
    require(np.all(sum(v.astype(int) for v in bins.values())==1),'Support strata not partition')
    out['common_support']={name:region_metrics(data['y'][ii],scores[ii],data['region'][ii],threshold) for name,ii in bins.items()}
    return out
