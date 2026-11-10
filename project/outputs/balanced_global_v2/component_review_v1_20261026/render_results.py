"""Render only recorded validation results and paired, actual seed differences."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

LABELS = {
    'raw16': 'Raw trajectory', 'raw_meta': 'Raw + reliability metadata',
    'gated_meta': 'Gated trajectory + metadata', 'mean_meta': 'Ordinary mean context',
    'affine_meta': 'Uniform affine context', 'reliable_raw': 'Reliable context + raw trajectory',
    'reliable_gate_target': 'Reliable context + target gate',
    'reliable_gate_residual': 'Reliable context + residual gate',
    'legacy_full': 'Reliable context + both gates',
    'reliable_raw_fallback': 'Reliable context + constant fallback',
}
BACKBONE_LABELS = {'tcn': 'TCN', 'gru': 'GRU', 'transformer': 'Transformer'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    protocol = json.loads((stage / 'protocol.json').read_text())
    for path in args.fonts.glob('times*.ttf'):
        font_manager.fontManager.addfont(str(path))
    font_path = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5,
                         'axes.titlesize': 10, 'svg.fonttype': 'path', 'axes.unicode_minus': False})
    features, backbones, seeds = protocol['feature_names'], protocol['backbones'], protocol['training']['seeds']
    results, inputs = {}, {}
    for b in backbones:
        for s in seeds:
            for f in features:
                path = stage / 'runs' / b / f'seed_{s}' / f / 'result.json'
                result = json.loads(path.read_text())
                assert result['status'] == 'completed_validation_only'
                assert len(result['history']) == protocol['training']['epochs']
                assert result['protocol_sha256'] == digest(stage / 'protocol.json')
                results[b, s, f] = result
                inputs[str(path.relative_to(stage))] = digest(path)
    dest = stage / 'figures'
    dest.mkdir(exist_ok=False)
    values = np.array([[[results[b, s, f]['validation']['macro']['AP'] for s in seeds]
                        for b in backbones] for f in features])
    mean, sd = values.mean(-1), values.std(-1, ddof=1)
    fig, ax = plt.subplots(figsize=(180 / 25.4, 118 / 25.4))
    cmap = LinearSegmentedColormap.from_list('review', ['#F4F8F7', '#A7CBC3', '#0F6C5C'])
    im = ax.imshow(mean, cmap=cmap, aspect='auto', vmin=min(.65, float(mean.min())), vmax=1)
    ax.set_xticks(range(len(backbones)), [BACKBONE_LABELS[b] for b in backbones])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(features)), [LABELS[f] for f in features])
    for i in range(len(features)):
        for j in range(len(backbones)):
            ax.text(j, i, f'{mean[i,j]:.3f}  +/- {sd[i,j]:.3f}', ha='center', va='center', fontsize=8.3,
                    color='white' if mean[i, j] > .88 else '#17342E')
    ax.set_title('Validation AP (mean +/- SD; three seeds)', loc='left', pad=27)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(backbones)), minor=True)
    ax.set_yticks(np.arange(-.5, len(features)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(dest / f'validation_component_matrix.{ext}', dpi=450)
    plt.close(fig)

    comparisons = [
        ('raw16', 'raw_meta', 'Add reliability metadata'),
        ('raw_meta', 'gated_meta', 'Gate target without context'),
        ('affine_meta', 'reliable_raw', 'Use reliable neighborhood'),
        ('reliable_raw', 'reliable_gate_target', 'Gate target with context'),
        ('reliable_raw', 'reliable_gate_residual', 'Gate field residual'),
        ('reliable_raw', 'reliable_raw_fallback', 'Use constant-field fallback'),
    ]
    deltas = np.array([[[100 * (results[b, s, to]['validation']['macro']['AP'] -
                               results[b, s, fr]['validation']['macro']['AP']) for s in seeds]
                        for fr, to, _ in comparisons] for b in backbones])
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 91 / 25.4), sharex=True, sharey=True)
    colors = ['#247C9C', '#C17D3B', '#6A5D9D']
    limit = max(1., float(np.max(np.abs(deltas))) * 1.25)
    for ib, (ax, b, color) in enumerate(zip(axes, backbones, colors)):
        y = np.arange(len(comparisons))
        for k in range(len(seeds)):
            ax.scatter(deltas[ib, :, k], y + (k - 1) * .11, color=color, alpha=.42, s=11, zorder=3)
        ax.errorbar(deltas[ib].mean(-1), y, xerr=deltas[ib].std(-1, ddof=1), fmt='o',
                    color=color, ecolor=color, markersize=4, capsize=3, linewidth=1.1, zorder=4)
        ax.axvline(0, color='#73817D', linewidth=.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(BACKBONE_LABELS[b])
        ax.set_xlabel('Change in validation AP (pp)')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', color='#D8E1DE', alpha=.7)
    axes[0].set_yticks(np.arange(len(comparisons)), [row[2] for row in comparisons])
    axes[0].invert_yaxis()
    fig.suptitle('Paired component effects: individual seeds, mean and sample SD', x=.015, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for ext in ['png', 'svg']:
        fig.savefig(dest / f'paired_component_effects.{ext}', dpi=450)
    plt.close(fig)

    selected = ['raw16', 'raw_meta', 'reliable_raw', 'legacy_full', 'reliable_raw_fallback']
    colors = ['#7C8990', '#D19B45', '#0F6C5C', '#A65B58', '#665994']
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 87 / 25.4), sharey=True)
    for ax, b in zip(axes, backbones):
        for f, color in zip(selected, colors):
            history = np.array([[h['validation_macro_AP'] for h in results[b, s, f]['history']] for s in seeds])
            x = np.arange(1, history.shape[1] + 1)
            m, st = history.mean(0), history.std(0, ddof=1)
            ax.plot(x, m, color=color, linewidth=1.1, label=LABELS[f])
            ax.fill_between(x, m - st, m + st, color=color, alpha=.1, linewidth=0)
        ax.set_title(BACKBONE_LABELS[b])
        ax.set_xlabel('Epoch')
        ax.set_xticks([1, 8, 16, 24])
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation macro AP')
    fig.suptitle('Validation learning curves: fixed 24-epoch budget', x=.08, ha='left', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', frameon=False, fontsize=7.3)
    fig.tight_layout(rect=(0, .22, 1, .94))
    for ext in ['png', 'svg']:
        fig.savefig(dest / f'validation_learning_curves.{ext}', dpi=450)
    plt.close(fig)
    provenance = dict(scope='Validation-only; all values are actual saved predictions/metrics from three seeds',
                      protocol_sha256=digest(stage / 'protocol.json'), font='Times New Roman', font_file=font_path,
                      input_result_sha256=inputs, features=features, backbones=backbones, seeds=seeds,
                      macro_ap_by_feature_backbone_seed=values.tolist(), comparisons=comparisons,
                      paired_delta_pp_by_backbone_comparison_seed=deltas.tolist(),
                      output_sha256={p.name: digest(p) for p in dest.iterdir()})
    (dest / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='rendered', figures=str(dest), runs=len(results))))


if __name__ == '__main__':
    main()
