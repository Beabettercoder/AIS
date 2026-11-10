"""Render current 11,200-window dataset table and Fig. 2, without result metrics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
ORDER = ('houston', 'ny', 'la', 'puget', 'great_belt', 'haugesund', 'kattegat', 'shanghai')
SPLITS = ('train', 'val', 'test')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def date_range(days):
    days = sorted(days)
    return days[0] if len(days) == 1 else days[0] + ' to ' + days[-1]


def collect(stage, registry_path):
    import pyarrow.parquet as pq
    manifest, audit = load(stage / 'data_manifest.json'), load(stage / 'data_audit.json')
    registry = load(registry_path)['regions']
    inputs = {str(stage / 'data_manifest.json'): sha(stage / 'data_manifest.json'),
              str(stage / 'data_audit.json'): sha(stage / 'data_audit.json'), str(registry_path): sha(registry_path)}
    datasets = {}
    for task in ('individual', 'relation'):
        datasets[task] = {}
        for split in SPLITS:
            info = manifest['tasks'][task][split]
            path = stage / info['path']
            assert sha(path) == info['sha256']
            inputs[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as z:
                keys = ['region', 'mmsi', 'source_id', 'window_id', 'y', 'mask', 'time_s', 'length']
                keys += ['common_valid'] if task == 'individual' else ['base__C1']
                data = {k: z[k].copy() for k in keys}
            assert len(data['y']) == info['counts']['windows']
            assert len(set(data['mmsi'])) == info['counts']['unique_target_mmsi']
            assert len(set(data['source_id'])) == info['counts']['sources']
            assert int(data['mask'].sum()) == info['counts']['valid_steps']
            datasets[task][split] = data
    panel_a, entries = [], []
    for key in ORDER:
        source = registry[key]
        canonical = Path(source['canonical_file'])
        assert sha(canonical) == source['sha256'], canonical
        inputs[str(canonical)] = source['sha256']
        rawrows = pq.ParquetFile(canonical).metadata.num_rows
        pieces, ships, selected_times = [], set(), []
        for split in SPLITS:
            data = datasets['individual'][split]
            pick = data['region'] == key
            targets = set(data['mmsi'][pick])
            ships.update(targets)
            stamps = data['time_s'][pick][data['mask'][pick]]
            selected_times.extend(stamps.tolist())
            valid = int(data['mask'][pick].sum())
            common = int(data['common_valid'][pick].sum())
            known = audit['tasks']['individual'][split]['regions'][key]
            assert int(pick.sum()) == known['windows'] and common == known['common_steps']
            pieces.append({'split': split, 'windows': int(pick.sum()), 'source_windows': len(set(data['source_id'][pick])),
                           'target_mmsi': len(targets), 'positives': int(data['y'][pick].sum()),
                           'target_observations': valid, 'common_context_steps': common,
                           'common_context_percent': 100 * common / valid})
        assert sum(p['windows'] for p in pieces) == 1400 and len(ships) == 260
        observed_days = sorted({datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d') for t in selected_times})
        registered_days = sorted(source['coverage_bbox_by_utc_day'])
        assert set(observed_days) <= set(registered_days), (key, observed_days, registered_days)
        relation_counts = {}
        for split in SPLITS:
            data = datasets['relation'][split]
            pick = data['region'] == key
            relation_counts[split] = {'sources': len(set(data['source_id'][pick])), 'variants': int(pick.sum()),
                                     'target_mmsi': len(set(data['mmsi'][pick]))}
        row = {'region': key, 'display_name': source['display_name'], 'continent': source['continent'],
               'provider': source['provider'], 'source_url': source['source_url'],
               'source_registry_utc_dates': registered_days, 'selected_target_utc_dates': observed_days,
               'source_canonical_file': str(canonical), 'source_canonical_sha256': source['sha256'],
               'canonical_source_report_rows': rawrows, 'core_bbox_wgs84': source['core_bbox'],
               'target_mmsi_all_splits': len(ships), 'windows_all_splits': 1400,
               'selected_target_observations': sum(p['target_observations'] for p in pieces),
               'splits': pieces, 'relation_by_split': relation_counts}
        panel_a.append(row)
        entries.append({'key': key, 'display_name': source['display_name'], 'continent': source['continent'],
                        'core_bbox': source['core_bbox'], 'windows': 1400, 'vessels': len(ships),
                        'relation_sources': sum(v['sources'] for v in relation_counts.values())})
    assert len(panel_a) == 8 and sum(p['windows_all_splits'] for p in panel_a) == 11200
    totals = {}
    for task in ('individual', 'relation'):
        allships = set().union(*(set(d['mmsi']) for d in datasets[task].values()))
        allsources = set().union(*(set(d['source_id']) for d in datasets[task].values()))
        totals[task] = {'target_mmsi': len(allships), 'sources': len(allsources),
                        'windows': sum(len(d['y']) for d in datasets[task].values()),
                        'target_observations': sum(int(d['mask'].sum()) for d in datasets[task].values()),
                        'positives': sum(int(d['y'].sum()) for d in datasets[task].values())}
    assert totals['individual']['target_mmsi'] == 2065
    panel_b = []
    for split in SPLITS:
        data = datasets['relation'][split]
        support = data['base__C1'][..., 6] > 0
        counts = audit['tasks']['relation'][split]['counts']
        panel_b.append({'split': split, **counts,
                        'common_context_steps': int(support.sum()),
                        'common_context_percent': 100 * support.sum() / data['mask'].sum(),
                        'background_mmsi': audit['tasks']['relation']['isolation']['counts'][split]['background_mmsi'],
                        'background_reports': audit['tasks']['relation']['isolation']['counts'][split]['background_reports']})
    table = {'title': 'Table 1. Data sources and evaluation splits', 'version': 'framework_validation_v1_20261028',
             'panel_A_individual_sources': panel_a, 'panel_B_relation_sources': panel_b,
             'panel_B_relation_overlap': audit['tasks']['relation']['isolation']['pairs'],
             'individual_overlap_audit_separate': audit['tasks']['individual']['isolation']['pairs'],
             'panel_C_confirmation': {'status': 'not_built', 'source_dates': None, 'regions': None, 'target_mmsi': None,
                                      'sources': None, 'labels': None, 'isolation_audit': 'pending collection and verification',
                                      'note': 'Existing test sources are previously exposed development evaluation; no fresh independent confirmation set exists in this stage.'},
             'totals': totals, 'inputs_sha256': inputs,
             'notes': ['Provider/date labels come from the source registry; selected UTC dates are verified from actual target time_s.',
                       'Canonical-source row counts describe source files, not the selected benchmark or unique global reports.',
                       'Selected target observations count valid target time steps; relation counts include four variants per source.',
                       'Regional individual target count is 260; global distinct MMSI is 2065 rather than the regional sum 2080.',
                       'Individual positive fraction 40%; relation four-cell XOR positive fraction 50%; unchanged source labels.',
                       'No-extra-injection target windows are not asserted manually verified normal.',
                       'Common individual support retains the historical affine-valid mask and its reliability/geometry selection.',
                       'Relation eligibility is offline whole-window; target isolation does not eliminate shared background observations.']}
    return table, entries, inputs


def markdown_table(table):
    lines = ['# Table 1. Data sources and evaluation splits', '',
             'Current 11,200-window benchmark. Counts come from the frozen framework data files, not the earlier 21,600-window source pool.', '',
             '## A. Individual-trajectory benchmark', '',
             '| Region | Provider | Source dates (UTC) | Source-file AIS reports | Selected target observations | Target MMSI (train/val/test; total) | Windows (train/val/test) | Positive fraction |',
             '|---|---|---|---:|---:|---|---|---:|']
    for r in table['panel_A_individual_sources']:
        ship = '/'.join(str(s['target_mmsi']) for s in r['splits']) + '; ' + str(r['target_mmsi_all_splits'])
        windows = '/'.join(str(s['windows']) for s in r['splits'])
        lines.append(f"| {r['display_name']} | [{r['provider']}]({r['source_url']}) | {date_range(r['source_registry_utc_dates'])} | {r['canonical_source_report_rows']:,} | {r['selected_target_observations']:,} | {ship} | {windows} | 40% |")
    total = table['totals']['individual']
    lines += [f"| Global distinct / total | Eight study regions | Registry dates above | Not pooled as unique reports | {total['target_observations']:,} | 1444/310/311; {total['target_mmsi']:,} | 7840/1680/1680 | 40% |", '',
              'Source-file report counts describe the canonical observation files. The selected benchmark contains exactly 1,400 windows in each region. The sum of regional target MMSI counts is 2,080; cross-region identity reuse gives 2,065 distinct targets globally. Source dates above describe registered availability; the JSON separately records dates actually present in selected target observations.', '',
              '| Region | Common context support: train | val | test |', '|---|---:|---:|---:|']
    for r in table['panel_A_individual_sources']:
        lines.append('| ' + r['display_name'] + ' | ' + ' | '.join(f"{s['common_context_percent']:.2f}%" for s in r['splits']) + ' |')
    lines += ['', 'Support is common-valid time steps divided by valid target time steps, conditional on the historical affine-valid support. C1, C2 and C3 use the same support. The raw source windows are not manually verified normal examples.', '',
              '## B. Group-relative task and background reuse', '',
              '| Split | Source windows | Target MMSI | Four-cell variants | Positive fraction | Context support | Background MMSI | Background reports |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table['panel_B_relation_sources']:
        lines.append(f"| {r['split']} | {r['sources']:,} | {r['unique_target_mmsi']:,} | {r['windows']:,} | 50% | {r['common_context_steps']:,}/{r['valid_steps']:,} ({r['common_context_percent']:.2f}%) | {r['background_mmsi']:,} | {r['background_reports']:,} |")
    lines += ['', '| Region | Relation sources (train/val/test) | Target MMSI (train/val/test) |', '|---|---|---|']
    for r in table['panel_A_individual_sources']:
        sources = '/'.join(str(r['relation_by_split'][s]['sources']) for s in SPLITS)
        ships = '/'.join(str(r['relation_by_split'][s]['target_mmsi']) for s in SPLITS)
        lines.append(f"| {r['display_name']} | {sources} | {ships} |")
    lines += ['', '| Split pair | Shared target MMSI | Shared source windows | Shared background MMSI | Shared background (MMSI, time) reports |',
              '|---|---:|---:|---:|---:|']
    for pair, r in table['panel_B_relation_overlap'].items():
        lines.append(f"| {pair.replace('_', '-')} | {r['target_mmsi']} | {r['source_id']} | {r['background_mmsi']:,} | {r['background_reports']:,} |")
    lines += ['', 'The relation task retains all four siblings in the same split and verifies target-only and neighbor-only marginal label balance. Cross-split target-versus-background identity intersections are zero in this relation dataset. Background identities and reports still repeat across splits. Relation membership uses offline full-window eligibility; high context coverage does not establish online or spatially unrestricted coverage.', '',
              'Individual-task background reuse is a separate audit: train-val 2,498 shared background MMSI / 32,215 reports; train-test 2,515 / 30,753; val-test 2,134 / 10,185. Individual train targets can appear as later-split background (849 in validation, 854 in test); 176 validation targets appear in test background. Do not describe all observations as independent.', '',
              '## C. Independent confirmation set', '',
              '| Status | New dates / regions | Sources and targets | Labels | Isolation audit |',
              '|---|---|---|---|---|',
              '| Not yet built | Pending | Pending | Pending | Pending collection and verification |', '',
              'Existing test sources were previously exposed during development. This table does not rename them as a fresh blind confirmation set. All labels and splits remain those of the frozen core data.', '',
              'Reproducibility: source registry, canonical files, frozen arrays and their SHA256 values are recorded in `Table1_dataset.json`.']
    return '\n'.join(lines) + '\n'


def render(stage, table, entries, map_path, fonts_path, land_path):
    spec = importlib.util.spec_from_file_location('approved_ais_map_geometry', map_path)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    plt = previous.plt
    fontproof = previous.register_fonts(fonts_path)
    land = list(previous.polygons(load(land_path)))
    figdir = stage / 'figures'
    figdir.mkdir(exist_ok=True)
    sources = {str(map_path): sha(map_path), str(land_path): sha(land_path), str(Path(__file__).resolve()): sha(__file__)}
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        sources[str(fonts_path / name)] = sha(fonts_path / name)
    outputs, rendered_texts = {}, []

    def save(fig, prefix):
        fig.canvas.draw()
        alltexts = [obj.get_text() for obj in fig.findobj() if hasattr(obj, 'get_text') and obj.get_text()]
        rendered_texts.extend(alltexts)
        assert not any('2,700' in txt or '21,600' in txt for txt in alltexts)
        for ext in ('png', 'svg'):
            path = figdir / (prefix + '.' + ext)
            fig.savefig(path, dpi=450, facecolor='white')
            outputs[str(path.relative_to(stage))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        plt.close(fig)

    fig = plt.figure(figsize=(180 / 25.4, 172 / 25.4), facecolor='white')
    title = fig.text(.055, .975, 'Geographic coverage and data support', ha='left', va='top', fontsize=12.5)
    subtitle = fig.text(.055, .942, '8 regions | 11,200 windows | 2,065 distinct target vessels', va='top', fontsize=8.8, color='#4A6574')
    mapax = fig.add_axes([.025, .59, .95, .32])
    previous.world(mapax, land, entries, standalone=False)
    fig.text(.055, .909, '(a) Current benchmark distribution', fontsize=9.5, va='top')
    gs = fig.add_gridspec(1, 2, left=.19, right=.975, bottom=.13, top=.48, width_ratios=[1.03, 1], wspace=.30)
    ax_count, ax_support = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    y = np.arange(8)
    targets = np.array([e['vessels'] for e in entries])
    relation = np.array([e['relation_sources'] for e in entries])
    ax_count.barh(y - .16, targets, height=.28, color='#67727B', label='Individual target vessels')
    ax_count.barh(y + .16, relation, height=.28, color='#16877A', label='Relation source windows')
    for yy, values in [(y - .16, targets), (y + .16, relation)]:
        for yp, value in zip(yy, values):
            ax_count.text(value + 13, yp, f'{value:,}', va='center', fontsize=7)
    ax_count.set_yticks(y, [e['display_name'].replace('–', '-') for e in entries], fontsize=8)
    ax_count.set_xlim(0, 1160)
    ax_count.set_xticks([0, 500, 1000])
    ax_count.set_xlabel('Count across train / val / test', fontsize=8)
    ax_count.set_title('(b) Region-level counts', loc='left', fontsize=9.2, pad=33)
    ax_count.legend(loc='lower left', bbox_to_anchor=(-.03, 1.005), fontsize=7.2, frameon=False, handlelength=1.2,
                    borderaxespad=0, labelspacing=.25)
    palette = {'train': '#9B9FA3', 'val': '#2563A6', 'test': '#16877A'}
    shifts = [-.21, 0, .21]
    plotted_support = {}
    for split, shift in zip(SPLITS, shifts):
        values = np.array([next(s for s in row['splits'] if s['split'] == split)['common_context_percent']
                           for row in table['panel_A_individual_sources']])
        plotted_support[split] = values.tolist()
        ax_support.barh(y + shift, values, height=.19, color=palette[split], label=split.capitalize())
    ax_support.set_yticks(y, [''] * 8)
    ax_support.set_xlim(0, 100)
    ax_support.set_xticks([0, 25, 50, 75, 100])
    ax_support.set_xlabel('Common context support (%)', fontsize=8)
    ax_support.set_title('(c) Individual-task support', loc='left', fontsize=9.2, pad=33)
    ax_support.legend(loc='lower left', bbox_to_anchor=(-.025, 1.005), ncol=3, fontsize=7.2, frameon=False,
                      handlelength=.9, columnspacing=.65, borderaxespad=0)
    for ax in (ax_count, ax_support):
        ax.set_ylim(7.6, -.6)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#DEE5E9', lw=.45)
        ax.tick_params(axis='x', labelsize=7.5, length=2.5, color='#7C929F')
        ax.tick_params(axis='y', length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#B6CCD7')
    footer = fig.text(.055, .048, 'Support: common-valid / valid target time steps; the historical affine-valid support is retained.', fontsize=7.7, color='#4A6574')
    fig.text(.055, .026, 'Regional vessel counts may share identities. Robinson projection; land: Natural Earth (110m).', fontsize=7.7, color='#4A6574')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not title.get_window_extent(renderer).overlaps(subtitle.get_window_extent(renderer))
    assert not ax_count.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
    assert not ax_support.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
    save(fig, 'fig2_geography_support')
    standalone = plt.figure(figsize=(180 / 25.4, 88 / 25.4), facecolor='white')
    standalone.text(.045, .97, 'Current 11,200-window benchmark: eight study regions', va='top', fontsize=11)
    ax = standalone.add_axes([.015, .015, .97, .90])
    previous.world(ax, land, entries, standalone=True)
    save(standalone, 'fig2_world_overview')
    expected_continents = {'North America': 5600, 'Europe': 4200, 'Asia': 1400}
    actual = {c: sum(e['windows'] for e in entries if e['continent'] == c) for c in expected_continents}
    assert actual == expected_continents
    for count in actual.values():
        assert any(f'{count:,} windows' in text for text in rendered_texts)
    caption = ('Fig. 2. Geographic coverage and data support for the current eight-region, 11,200-window benchmark. '
               '(a) Region centers are the midpoints of the source-registry study bounding boxes, rendered with the Robinson projection. '
               'North America, Europe and Asia contribute 5,600, 4,200 and 1,400 windows, respectively. '
               '(b) Region-wise distinct individual-task target vessels and relation-task source windows, summed over the three splits. '
               'The individual regional vessel counts include cross-region identities; the global distinct count is 2,065. '
               '(c) Common context support divided by valid target time steps for train, validation and test. '
               'All context configurations share the retained historical affine-valid support; coverage does not represent unrestricted neighbor availability. '
               'Relation source counts have their own task and sampling definition and are not added to the 11,200 individual windows. '
               'Source providers and UTC dates appear in Table 1. Land polygons: Natural Earth 110m.\n')
    (figdir / 'fig2_caption.md').write_text(caption, encoding='utf8')
    meta = {'status': 'rendered_from_verified_current_dataset', 'figure_width_mm': 180, 'figure_height_mm': 172,
            'core_files_changed': False, 'projection': '+proj=robin +lon_0=0 +R=6371008.8',
            'fonts': fontproof, 'regions': entries, 'continent_windows': actual,
            'individual_windows': 11200, 'individual_global_target_mmsi': 2065,
            'common_context_percent': plotted_support, 'figure_source_sha256': sources,
            'dataset_source_sha256': table['inputs_sha256'], 'table_sha256': sha(stage / 'tables/Table1_dataset.json'),
            'outputs': outputs, 'rendered_texts': rendered_texts, 'caption': caption,
            'assertions': {'exact_eight_regions': len(entries) == 8, 'no_oresund': all(e['key'] != 'oresund' for e in entries),
                           'exact_1400_per_region': all(e['windows'] == 1400 for e in entries),
                           'old_2700_and_21600_labels_absent': True, 'providers_dates_and_source_hashes_verified': True}}
    dump(figdir / 'fig2_metadata.json', meta)
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--registry', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/release_20261026/source_registry.json'))
    ap.add_argument('--map-renderer', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/visuals/v2/render_map_v2.py'))
    ap.add_argument('--fonts', type=Path, default=Path('/mnt/sda/WangZijia/AIS/outputs/balanced_global_v2/visuals/v2/fonts_runtime'))
    ap.add_argument('--land', type=Path, default=Path('/mnt/sda/WangZijia/AIS/data/geo/ne_110m_land.geojson'))
    args = ap.parse_args()
    stage = args.stage.resolve()
    assert stage.name == 'framework_validation_v1_20261028'
    tabledir = stage / 'tables'
    tabledir.mkdir(exist_ok=True)
    lock = load(stage / 'code_lock.json')
    locked = {str(stage / key): value for key, value in {**lock['files'], **lock['data_files']}.items()}
    for path, checksum in locked.items():
        assert sha(path) == checksum
    table, entries, sourcefiles = collect(stage, args.registry)
    table['renderer_sha256'] = sha(__file__)
    dump(tabledir / 'Table1_dataset.json', table)
    text = markdown_table(table)
    assert '2700/2700' not in text and '7840/1680/1680' in text
    (tabledir / 'Table1_dataset.md').write_text(text, encoding='utf8')
    figure_sources = render(stage, table, entries, args.map_renderer, args.fonts, args.land)
    for path, checksum in {**locked, **sourcefiles, **figure_sources}.items():
        assert sha(path) == checksum, ('source_changed', path)
    print(json.dumps({'status': 'complete', 'individual': table['totals']['individual'],
                      'relation': table['totals']['relation'], 'outputs': ['tables/Table1_dataset.md', 'tables/Table1_dataset.json',
                        'figures/fig2_geography_support.png', 'figures/fig2_geography_support.svg',
                        'figures/fig2_world_overview.png', 'figures/fig2_world_overview.svg']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
