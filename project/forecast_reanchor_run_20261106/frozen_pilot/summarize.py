"""Audit and summarize validation-only forecasting predictions.

The file never opens test data or trains models. Run after (or during) the pilot:
    python summarize.py --runs ./runs --out ./summary --bootstrap 2000
Paired differences are candidate minus comparator, so negative error is better.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

MODELS = ('tcn', 'gru', 'transformer', 'itransformer')
ARMS = ('P0', 'P1', 'P2', 'P3')
SEEDS = (0, 1, 2)
VIEWS = ('base', 'target_jump', 'target_drift', 'target_speed', 'neighbor_stress')
ARM_ZH = {'P0': '原始历史', 'P1': '因果门控', 'P2': '门控与普通留一场', 'P3': '门控与可信留一场', 'CV_raw': '原始恒速外推', 'CV_gate': '门控恒速外推'}
VIEW_ZH = {'base': '原始观测', 'target_jump': '目标末点位置跳变', 'target_drift': '目标末段位置漂移', 'target_speed': '目标末段速度尖峰', 'neighbor_stress': '邻船速度与位置污染'}
PAIRS = (('P1', 'P0'), ('P2', 'P1'), ('P3', 'P2'), ('P3', 'P0'))
SUBSETS = ('all', 'observed_maneuver')


def scalar(z, key, default=None):
    if key not in z:
        return default
    value = np.asarray(z[key])
    if value.size != 1:
        raise ValueError(f'{key} must be scalar, got {value.shape}')
    return value.reshape(-1)[0].item()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(x) for x in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    tmp = Path(str(path) + '.tmp')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        if fields:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
    tmp.replace(path)


def load_prediction(path):
    with np.load(path, allow_pickle=False) as z:
        required = ('pred_xy', 'future_xy', 'query_s', 'mmsi', 'region',
                    'window_id', 'maneuver', 'split', 'model', 'arm', 'seed')
        missing = [k for k in required if k not in z]
        if missing:
            raise ValueError(f'{path}: missing fields {missing}')
        if str(scalar(z, 'split')) != 'val':
            raise ValueError(f'{path}: only validation predictions may be summarized')
        pred = np.asarray(z['pred_xy'], np.float64)
        truth = np.asarray(z['future_xy'], np.float64)
        query = np.asarray(z['query_s'], np.float64)
        if pred.ndim != 3 or pred.shape[-1] != 2 or pred.shape != truth.shape:
            raise ValueError(f'{path}: expected paired [N,F,2] positions')
        n, steps, _ = pred.shape
        if n == 0 or steps == 0 or query.shape != (n, steps):
            raise ValueError(f'{path}: empty/mismatched predictions and query times')
        if not (np.isfinite(pred).all() and np.isfinite(truth).all()
                and np.isfinite(query).all() and (query > 0).all()
                and (np.diff(query, axis=1) > 0).all()):
            raise ValueError(f'{path}: invalid coordinates or nonpositive/unordered times')
        mid = np.asarray(z['mmsi']).astype(str)
        region = np.asarray(z['region']).astype(str)
        ids = np.asarray(z['window_id']).astype(str)
        maneuver = np.asarray(z['maneuver'], bool)
        if any(x.shape != (n,) for x in (mid, region, ids, maneuver)):
            raise ValueError(f'{path}: metadata must contain one value per window')
        if len(set(ids)) != n:
            raise ValueError(f'{path}: window_id is not unique')
        errors = np.linalg.norm(pred - truth, axis=-1)
        ade = errors.mean(axis=1)
        fde = errors[:, -1]
        for key, computed in [('point_error', errors), ('ADE', ade), ('FDE', fde)]:
            if key in z:
                stored = np.asarray(z[key], np.float64)
                if stored.shape != computed.shape or not np.allclose(
                        stored, computed, rtol=1e-7, atol=1e-6):
                    raise ValueError(f'{path}: saved {key} differs from coordinate recomputation')
        model = str(scalar(z, 'model'))
        arm = str(scalar(z, 'arm'))
        seed = int(scalar(z, 'seed'))
        view = str(scalar(z, 'view', path.stem[len('predictions_'):]))
    order = np.argsort(ids, kind='stable')
    result = dict(path=str(path.resolve()), file_sha256=sha256(path), model=model,
                  arm=arm, seed=seed, view=view, split='val',
                  mmsi=mid[order], region=region[order], window_id=ids[order],
                  maneuver=maneuver[order], query_s=query[order],
                  future_xy=truth[order], ADE=ade[order], FDE=fde[order],
                  prediction_sha256=hashlib.sha256(pred[order].tobytes()).hexdigest())
    status_path = path.parent / 'result.json'
    result['completion_marker'] = False
    result['budget_matches_protocol'] = False
    result['expected_epochs'] = None
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding='utf-8'))
        result['completion_marker'] = (status.get('status') == 'complete'
            and status.get('completed_epochs') == status.get('expected_epochs'))
        result['expected_epochs'] = status.get('expected_epochs')
        result['budget_matches_protocol'] = (model.startswith('CV') or
            result['expected_epochs'] == (75 if model == 'itransformer' else 25))
    return result


def aligned(a, b):
    for key in ('window_id', 'mmsi', 'region', 'maneuver', 'query_s', 'future_xy'):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f'Paired data mismatch in {key}: {a["path"]} vs {b["path"]}')


def subset_mask(record, subset):
    return (np.ones(len(record['window_id']), bool) if subset == 'all'
            else record['maneuver'])


def metrics(record, mask):
    n = int(mask.sum())
    if not n:
        return {'windows': 0, 'target_mmsi': 0, 'regions': 0,
                'ADE_m': None, 'FDE_m': None, 'region_macro_ADE_m': None,
                'region_macro_FDE_m': None}
    regions = np.unique(record['region'][mask])
    out = dict(windows=n, target_mmsi=len(np.unique(record['mmsi'][mask])),
               regions=len(regions))
    for key in ('ADE', 'FDE'):
        out[key + '_m'] = float(record[key][mask].mean())
        out['region_macro_' + key + '_m'] = float(np.mean([
            record[key][mask & (record['region'] == reg)].mean() for reg in regions]))
    return out


def seed_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def time_description(record, mask):
    if not mask.any():
        return dict(first_query_s_min=None, first_query_s_median=None,
                    final_horizon_s_min=None, final_horizon_s_median=None,
                    final_horizon_s_p90=None, final_horizon_s_max=None)
    q = record['query_s'][mask]
    end = q[:, -1]
    return dict(first_query_s_min=float(q[:, 0].min()),
                first_query_s_median=float(np.median(q[:, 0])),
                final_horizon_s_min=float(end.min()),
                final_horizon_s_median=float(np.median(end)),
                final_horizon_s_p90=float(np.quantile(end, .9)),
                final_horizon_s_max=float(end.max()))


def paired_bootstrap(candidates, comparators, subset, draws, rng_seed):
    """Resample global target MMSIs, retaining every window of each sampled ship.

    Differences are first averaged over the matched fixed trained seeds. The same
    sampled ship multiplicities apply to all arms and seeds. For macro estimates,
    a replicate lacking any original region is undefined and is not used; the
    valid replicate count is always reported instead of silently dropping regions.
    """
    reference = candidates[0]
    for r in candidates + comparators:
        aligned(reference, r)
    mask = subset_mask(reference, subset)
    n = int(mask.sum())
    if not n:
        return []
    mid = reference['mmsi'][mask]
    reg = reference['region'][mask]
    unique_mid, cluster = np.unique(mid, return_inverse=True)
    regions = np.unique(reg)
    m = len(unique_mid)
    count = np.bincount(cluster, minlength=m).astype(float)
    diff = np.stack([np.mean(np.stack([c[key][mask] - b[key][mask]
                                     for c, b in zip(candidates, comparators)]), axis=0)
                     for key in ('ADE', 'FDE')])
    sums = np.stack([np.bincount(cluster, weights=x, minlength=m) for x in diff])
    reg_counts = np.stack([np.bincount(cluster, weights=(reg == region).astype(float),
                                      minlength=m) for region in regions])
    reg_sums = np.stack([np.stack([np.bincount(cluster,
                      weights=x * (reg == region), minlength=m) for region in regions])
                      for x in diff])
    rng = np.random.default_rng(rng_seed)
    boot = np.empty((draws, 2), dtype=float)
    macro = np.full((draws, 2), np.nan)
    for start in range(0, draws, 64):
        end = min(start + 64, draws)
        sampled = rng.integers(0, m, size=(end - start, m))
        weights = np.zeros((end - start, m), float)
        for k, row in enumerate(sampled):
            weights[k] = np.bincount(row, minlength=m)
        denominator = weights @ count
        boot[start:end] = (weights @ sums.T) / denominator[:, None]
        region_denominator = weights @ reg_counts.T
        valid = np.all(region_denominator > 0, axis=1)
        for metric in range(2):
            numerators = weights @ reg_sums[metric].T
            macro[start:end, metric][valid] = np.mean(
                numerators[valid] / region_denominator[valid], axis=1)
    output = []
    for index, key in enumerate(('ADE', 'FDE')):
        for aggregation, value, values in [
            ('pooled', float(diff[index].mean()), boot[:, index]),
            ('region_macro', float(np.mean([diff[index][reg == r].mean() for r in regions])),
             macro[:, index])]:
            valid_values = values[np.isfinite(values)]
            identifiable = m >= 2 and len(valid_values) >= 2
            low, high = (np.quantile(valid_values, [.025, .975]) if identifiable
                         else (np.nan, np.nan))
            output.append(dict(metric=key + '_m', aggregation=aggregation,
                               delta=value, ci95_low=float(low), ci95_high=float(high),
                               bootstrap_requested=draws,
                               bootstrap_valid=int(len(valid_values)),
                               target_mmsi=m, windows=n, regions=len(regions),
                               interval_status='available' if identifiable else 'insufficient_clusters'))
    return output


def format_value(x):
    return '—' if x is None or not math.isfinite(float(x)) else f'{float(x):.2f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=2000)
    ap.add_argument('--bootstrap-seed', type=int, default=20261105)
    args = ap.parse_args()
    if args.bootstrap < 0:
        ap.error('--bootstrap must be >= 0 (0 skips intervals)')
    files = sorted(args.runs.rglob('predictions_*.npz'))
    if not files:
        raise SystemExit(f'No saved validation predictions under {args.runs}. No results generated.')
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for path in tqdm(files, desc='检查已保存验证预测', unit='文件'):
        records.append(load_prediction(path))
    reference = records[0]
    # A single pilot uses the same validation IDs, labels and query times in all views.
    for r in tqdm(records[1:], desc='核验配对来源与未来参考', unit='文件'):
        aligned(reference, r)
    index = {}
    duplicate_baselines = []
    for r in records:
        key = (r['model'], r['arm'], r['seed'], r['view'])
        if key in index:
            prior = index[key]
            if (r['model'].startswith('CV') and
                    r['prediction_sha256'] == prior['prediction_sha256']):
                aligned(prior, r)
                duplicate_baselines.append({'kept':prior['path'], 'duplicate':r['path']})
                continue
            raise ValueError(f'Duplicate prediction key {key}; refusing arbitrary version selection')
        index[key] = r
    all_prediction_files = records
    records = list(index.values())
    observed_views = sorted({r['view'] for r in records})
    unknown_views = set(observed_views) - set(VIEWS)
    if unknown_views:
        raise ValueError(f'Unknown views: {sorted(unknown_views)}')
    views = list(VIEWS)
    expected = {(m, a, s) for m in MODELS for a in ARMS for s in SEEDS}
    completed = set()
    for key in expected:
        present = [index.get((*key, v)) for v in views]
        if all(r is not None and r['completion_marker'] and r['budget_matches_protocol'] for r in present):
            completed.add(key)
    observed_runs = {(r['model'], r['arm'], r['seed']) for r in records if r['model'] in MODELS}
    status = 'complete' if len(completed) == len(expected) else 'partial'
    seed_rows, region_rows, horizon_rows = [], [], []
    for r in tqdm(records, desc='计算各模型各视图误差', unit='文件'):
        base = {k: r[k] for k in ('model', 'arm', 'seed', 'view')}
        for subset in SUBSETS:
            mask = subset_mask(r, subset)
            row = {**base, 'subset': subset, **metrics(r, mask), **time_description(r, mask)}
            seed_rows.append(row)
        for region in sorted(np.unique(r['region'])):
            region_rows.append({**base, 'region': str(region), **metrics(r, r['region'] == region)})
        end = r['query_s'][:, -1]
        for label, mask in [('<=300s', end <= 300), ('300-600s', (end > 300) & (end <= 600)),
                            ('>600s', end > 600)]:
            horizon_rows.append({**base, 'horizon_stratum': label, **metrics(r, mask),
                                 **time_description(r, mask)})
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row['model'], row['arm'], row['view'], row['subset'])].append(row)
    summary_rows = []
    for key, rows in sorted(grouped.items()):
        baseline = key[0].startswith('CV')
        row = dict(zip(('model', 'arm', 'view', 'subset'), key))
        row.update(n_seeds=len(rows), seeds=','.join(str(r['seed']) for r in sorted(rows, key=lambda x: x['seed'])),
                   seed_status='deterministic_baseline' if baseline else
                   ('three_seeds' if {r['seed'] for r in rows} == set(SEEDS) else 'partial_seeds'))
        for name in ('windows', 'target_mmsi', 'regions', 'first_query_s_min', 'first_query_s_median',
                     'final_horizon_s_min', 'final_horizon_s_median', 'final_horizon_s_p90',
                     'final_horizon_s_max'):
            row[name] = rows[0][name]
        for metric in ('ADE_m', 'FDE_m', 'region_macro_ADE_m', 'region_macro_FDE_m'):
            values = [r[metric] for r in rows if r[metric] is not None]
            row[metric + '_mean'] = float(np.mean(values)) if values else None
            row[metric + '_seed_sd'] = seed_std(values) if values and not baseline else None
        summary_rows.append(row)
    pair_rows, pair_coverage = [], []
    work = [(model, view, cand, base, subset) for model in MODELS for view in views
            for cand, base in PAIRS for subset in SUBSETS]
    for model, view, cand, base, subset in tqdm(work, desc='目标船配对 Bootstrap', unit='对照'):
        seeds = [s for s in SEEDS if (model, cand, s, view) in index and (model, base, s, view) in index]
        coverage = dict(model=model, view=view, contrast=f'{cand}-{base}', subset=subset,
                        n_seeds=len(seeds), seeds=','.join(map(str, seeds)),
                        status='three_seeds' if len(seeds) == 3 else 'partial_seeds')
        pair_coverage.append(coverage)
        if not seeds:
            continue
        cand_rows = [index[(model, cand, s, view)] for s in seeds]
        base_rows = [index[(model, base, s, view)] for s in seeds]
        values = paired_bootstrap(cand_rows, base_rows, subset, args.bootstrap,
                                  args.bootstrap_seed)
        for value in values:
            pair_rows.append({**coverage, **value})
    audit = dict(status=status, split='val', expected_training_runs=len(expected),
                 completed_training_runs=len(completed), observed_training_runs=len(observed_runs),
                 prediction_files=len(all_prediction_files), unique_prediction_files=len(records),
                 views=views, observed_views=observed_views, duplicate_identical_baselines=duplicate_baselines, missing_runs=[
                     dict(model=m, arm=a, seed=s) for m, a, s in sorted(expected - completed)],
                 bootstrap_seed=args.bootstrap_seed, bootstrap_draws=args.bootstrap,
                 definition='Candidate minus comparator error in metres; negative means improvement.',
                 pair_coverage=pair_coverage, time_horizon=time_description(reference, np.ones(len(reference['window_id']), bool)),
                 files=[{k: r[k] for k in ('path', 'file_sha256', 'model', 'arm', 'seed', 'view', 'completion_marker', 'budget_matches_protocol', 'expected_epochs')}
                        for r in all_prediction_files], summary=summary_rows, paired=pair_rows,
                 notes=[
                     'Observed maneuver is a rule-defined evaluation stratum, not an incident label.',
                     'ADE is mean pointwise Euclidean error; FDE is final requested-time error in a common metric frame.',
                     'Target-MMSI cluster intervals condition on the saved trained seeds and retain every window of sampled ships.',
                     'Macro bootstrap replicates missing any original region are undefined and omitted; valid replicate counts are reported.',
                     'Intervals are pointwise and do not account for all shared-background dependence or multiplicity.',
                     'Only validation predictions are read; incomplete pilots remain explicitly partial.',
                 ])
    write_csv(args.out / 'per_seed.csv', seed_rows)
    write_csv(args.out / 'per_region.csv', region_rows)
    write_csv(args.out / 'per_horizon.csv', horizon_rows)
    write_csv(args.out / 'summary.csv', summary_rows)
    write_csv(args.out / 'paired_bootstrap.csv', pair_rows)
    write_json(args.out / 'summary.json', audit)
    lines = ['# 航迹预测验证集汇总', '',
             f'状态：**{status}**；完成训练任务 **{len(completed)}/{len(expected)}**。',
             '单位均为米，越低越好；差值＝候选配置－对照配置，负值表示误差降低。',
             '每项同时给出合并窗口结果与区域等权结果；缺失模型或种子不填零。',
             '完整小试要求五种验证视图、四个骨干各三种子四组配置，TCN/GRU/Transformer 25轮、iTransformer 75轮；少量流程测试不会计为完整小试。', '',
             '|模型|配置|观测视图|样本分层|种子数|ADE均值|FDE均值|区域等权ADE|区域等权FDE|',
             '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary_rows:
        lines.append('|'+ '|'.join([r['model'], ARM_ZH.get(r['arm'], r['arm']), VIEW_ZH.get(r['view'], r['view']), ('全部窗口' if r['subset']=='all' else '观测机动窗口'), str(r['n_seeds']),
             *(format_value(r[x]) for x in ('ADE_m_mean', 'FDE_m_mean',
                                            'region_macro_ADE_m_mean', 'region_macro_FDE_m_mean'))])+'|')
    lines.extend(['', '## 配对差值与逐项95%区间', '',
                  '|模型|视图|对照|分层|种子数|聚合|指标|差值|区间下限|区间上限|有效重采样次数|',
                  '|---|---|---|---|---:|---|---|---:|---:|---:|---:|'])
    for r in pair_rows:
        lines.append('|'+ '|'.join([r['model'],VIEW_ZH.get(r['view'],r['view']),r['contrast'],('全部窗口' if r['subset']=='all' else '观测机动窗口'),str(r['n_seeds']),
             r['aggregation'],r['metric'],format_value(r['delta']),format_value(r['ci95_low']),
             format_value(r['ci95_high']),str(r['bootstrap_valid'])])+'|')
    tm = audit['time_horizon']
    lines.extend(['', '## 预测时间跨度', '',
                  '以下为每窗口最后一个查询时间相对历史末点的秒数，而不是固定步数换算：',
                  f"最小 {format_value(tm['final_horizon_s_min'])} 秒；中位 {format_value(tm['final_horizon_s_median'])} 秒；P90 {format_value(tm['final_horizon_s_p90'])} 秒；最大 {format_value(tm['final_horizon_s_max'])} 秒。",
                  '', '逐窗口时间段分层见 per_horizon.csv；逐区域结果见 per_region.csv；逐种子结果与波动见 per_seed.csv / summary.csv。',
                  '目标船聚类区间针对已训练的固定模型；区域等权区间中缺失某区域的重采样无定义，已单独报告有效次数。',
                  '观测机动是按速度/方向变化规则划分，不是人工确认的真实异常。没有读取测试数据。'])
    (args.out / '航迹预测验证集汇总.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'SUMMARY {status}: {len(completed)}/{len(expected)} completed training runs; {args.out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
