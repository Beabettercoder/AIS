"""Deterministic Fig. 1 / Fig. 8 support, separate from the frozen training code.

CPU-only stages:
  AIS .venv Python: --mode select, --mode framework, --mode render
  shared torch Python: --mode infer (requires all final checkpoints locked)
No validation/test score is read when selecting cases. No GPU is requested.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import sys
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
SELECTION = ROOT / 'case_selection.json'
ARCHITECTURES = ('tcn', 'gru', 'transformer')
ARMS = ('C0', 'C3')
SEED = 0
KNOT = 1852 / 3600


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_json(path, data, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def source_root():
    return Path(json.loads((ROOT / 'data_sources.json').read_text())['source_root'])


def source_file(task, region, window_id):
    if task == 'individual':
        return source_root() / 'labeling/labeled_v1/features_scene' / region / 'test.npz'
    return source_root() / 'relation_dual_pressure_v2_20261027/cases/relaxed' / region / 'test' / (window_id.split(':case')[0] + '_base.npz')


def select_cases():
    if SELECTION.exists():
        selection = json.loads(SELECTION.read_text())
        for task, h in selection['test_data_sha256'].items():
            assert sha(ROOT / 'data' / task / 'test.npz') == h
        print('SELECTION_ALREADY_FROZEN', sha(SELECTION), flush=True)
        return
    import pandas as pd
    datasets = {task: load_npz(ROOT / 'data' / task / 'test.npz') for task in ('individual', 'relation')}
    source_manifest = json.loads((ROOT / 'data_sources.json').read_text())['sha256']
    meta = {}
    meta_files = {}
    d = datasets['individual']
    for region in sorted(set(d['region'].astype(str))):
        path = source_root() / 'labeling/labeled_v1' / (region + '_label_manifest.parquet')
        meta_files[str(path)] = sha(path)
        frame = pd.read_parquet(path, columns=['window_id', 'onset_utc_s', 'start_utc_s', 'end_utc_s', 'prefix_obs', 'post_obs'])
        for row in frame.to_dict('records'):
            meta[(region, str(row['window_id']))] = row
    rules = [
        ('coherent_group', 'relation', 'Coherent group slowdown', 'case3 = target/cohort factors 1/1; label0; no effective observation noise'),
        ('target_teleport', 'individual', 'Target position jump', 'teleport; target_only; effective target noise; behavior none; label0'),
        ('early_turn_back', 'individual', 'Early turn-back', 'turn_back; label1; no effective observation noise; onset in first half of elapsed window'),
        ('deviation', 'individual', 'Individual deviation', 'deviation; label1; no effective observation noise'),
    ]
    selected = []
    for key, task, title, rule in rules:
        d = datasets[task]
        support = (d['base__C3'][..., -1] > 0) & d['mask']
        eligible = (d['length'] >= 12) & (support.sum(1) >= 4)
        if key == 'coherent_group':
            eligible &= (d['case'] == 3) & (d['y'] == 0) & ~d['observation_noise_effective']
        elif key == 'target_teleport':
            eligible &= (d['noise_type'] == 'teleport') & (d['noise_scope'] == 'target_only') & d['target_noise_effective'] & (d['behavior_type'] == 'none') & (d['y'] == 0)
        else:
            eligible &= (d['behavior_type'] == ('turn_back' if key == 'early_turn_back' else 'deviation')) & (d['y'] == 1) & ~d['observation_noise_effective']
            if key == 'early_turn_back':
                fractions = np.asarray([(meta[(str(r), str(w))]['onset_utc_s'] - meta[(str(r), str(w))]['start_utc_s']) / (meta[(str(r), str(w))]['end_utc_s'] - meta[(str(r), str(w))]['start_utc_s']) for r, w in zip(d['region'], d['window_id'])])
                eligible &= np.isfinite(fractions) & (fractions >= 0) & (fractions <= .5)
        ordered = sorted(np.flatnonzero(eligible), key=lambda i: (str(d['window_id'][i]), str(d['region'][i])))
        excluded = []
        choice = None
        for i in ordered:
            region, window = str(d['region'][i]), str(d['window_id'][i])
            path = source_file(task, region, window)
            if not path.exists():
                excluded.append({'window_id': window, 'reason': 'source coordinate archive absent'})
                continue
            raw = load_npz(path)
            required = ['target_pos', 'pre_noise_pos', 'window_id'] if task == 'individual' else ['target_pos', 'pre_noise_target_pos', 'neighbor_pos', 'meta_target_factor', 'meta_cohort_factor']
            if not all(k in raw for k in required):
                excluded.append({'window_id': window, 'reason': 'required recorded coordinates absent'})
                continue
            if task == 'individual':
                loc = np.flatnonzero(raw['window_id'].astype(str) == window)
                assert len(loc) == 1
                source_index = int(loc[0])
                assert np.all(np.isfinite(raw['target_pos'][source_index, raw['mask'][source_index]]))
            else:
                source_index = int(d['case'][i])
                assert raw['meta_target_factor'][source_index] == 1 and raw['meta_cohort_factor'][source_index] == 1 and raw['y'][source_index] == 0
            assert sha(path) == source_manifest[str(path)]
            choice = {'case_key': key, 'title': title, 'task': task, 'split': 'test', 'view': 'base', 'index': int(i),
                      'window_id': window, 'source_id': str(d['source_id'][i]), 'region': region, 'mmsi': str(d['mmsi'][i]),
                      'source_mmsi': str(d.get('source_mmsi', d['mmsi'])[i]), 'label': int(d['y'][i]),
                      'length': int(d['length'][i]), 'context_valid_steps': int(support[i].sum()),
                      'rule': rule, 'eligible_count': len(ordered), 'availability_exclusions_before_selection': excluded,
                      'source_archive': str(path), 'source_archive_sha256': sha(path), 'source_row_or_case': source_index,
                      'selection_rank_among_metadata_eligible': len(excluded) + 1,
                      'coordinate_interpretation': 'recorded positions minus one common first-target origin; no cumulative sum',
                      'prefix_scope': 'offline prefix diagnostic conditional on full-window cohort' if task == 'relation' else 'observed-prefix input diagnostic'}
            if task == 'individual':
                row = meta[(region, window)]
                choice['recorded_behavior_onset_s'] = float(row['onset_utc_s']) if key != 'target_teleport' else None
                if key == 'early_turn_back':
                    choice['behavior_onset_elapsed_fraction'] = float((row['onset_utc_s'] - row['start_utc_s']) / (row['end_utc_s'] - row['start_utc_s']))
            break
        if choice is None:
            raise RuntimeError(f'No eligible recorded case for {key}; keep figure pending instead of inventing a substitute')
        selected.append(choice)
    selection = {'status': 'frozen_before_reading_model_scores', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'protocol': {'selection': 'first lexicographic window_id, then region, satisfying metadata and recorded-source availability',
                              'common_eligibility': 'test/base; at least 12 valid target observations and 4 context-valid observations',
                              'architectures': list(ARCHITECTURES), 'arms': list(ARMS), 'seed': SEED,
                              'prefix_lengths': 'all observed prefixes of length 2 through final valid length',
                              'threshold': 'each model/config fixed full-window base-validation 1% nominal FAR threshold',
                              'prefix_plot': 'sigmoid(logit) minus own frozen threshold; threshold=0 line; not prefix-calibrated FAR',
                              'model_scores_used_for_selection': False, 'failure_cases_retained': True,
                              'reference_semantics': 'pre-observation-noise trajectory; may already include controlled behavior; not external positioning ground truth',
                              'relation_semantics': 'full-window eligibility/cohort retained; no online warning claim'},
                 'model_inventory_sha256': sha(ROOT / 'model_inventory.json'), 'code_lock_sha256': sha(ROOT / 'code_lock.json'),
                 'test_data_sha256': {task: sha(ROOT / 'data' / task / 'test.npz') for task in datasets},
                 'source_metadata_sha256': meta_files, 'cases': selected}
    save_json(SELECTION, selection, exclusive=True)
    print('SELECTION_FROZEN', sha(SELECTION), [(c['case_key'], c['window_id']) for c in selected], flush=True)


def configure_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_paths = [FIG / 'fig1_framework.font_regular.ttf', FIG / 'fig1_framework.font_bold.ttf']
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'axes.labelsize': 10.5,
                         'axes.titlesize': 12, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.facecolor': 'white', 'figure.facecolor': 'white', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e2e5e7', 'grid.linewidth': .55})
    return plt


def save_figure(fig, stem):
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{stem}.{suffix}', dpi=220, bbox_inches='tight', facecolor='white')


def render_framework():
    plt = configure_plot()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    inventory = json.loads((ROOT / 'model_inventory.json').read_text())
    names = [m['name'] for m in inventory['models']]
    assert len(names) == 8 and len(set(names)) == 8
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    def box(x, y, w, h, text, color='#f6f8fa', edge='#708090', size=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009,rounding_size=0.012', linewidth=1.05, facecolor=color, edgecolor=edge))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=size, linespacing=1.45)
    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13, linewidth=1.25, color='#465562', linestyle='--' if dashed else '-'))
    ax.text(.5, .96, 'Reliability-aware context modeling across distinct trajectory encoders', ha='center', fontsize=18)
    box(.025, .69, .19, .16, 'Observed trajectories\nTarget and neighbors\nValid time mask')
    box(.265, .69, .18, .16, 'Kinematic compatibility\nObserved position vs.\nreported velocity')
    box(.495, .665, .235, .21, 'Shared reference operator\nFixed members and support\nC1 mean / C2 median\nC3 reliability-weighted median', '#eaf6f3', '#338a7b')
    box(.785, .665, .19, .21, 'Common input interface\nTarget | Reference\nResidual / delta | Support', '#eaf6f3', '#338a7b', 11.3)
    for x1, x2 in ((.215,.265),(.443,.495),(.73,.785)): arrow((x1,.77),(x2,.77))
    ax.plot([.12,.12,.88,.88],[.69,.61,.61,.665], '--', color='#666e77', lw=1.05)
    ax.text(.42,.622,'C0: target only, context slots masked', ha='center', fontsize=10.5, color='#4d5359')
    box(.30,.245,.675,.30,'', '#fbfcfd','#adb7bf')
    ax.text(.635,.515,'Eight classification adaptations with a common output head',ha='center',fontsize=12.5)
    labels = {'Neural-CDE':'Neural CDE\nRK4 control solver','GraphTemporal':'Graph temporal*\nTwo-node GAT + TCN','iTransformer':'iTransformer\nVariable tokens','PatchTST':'PatchTST\nChannel-wise patches','Mamba':'Mamba\nSelective SSM','Transformer':'Transformer\nTime tokens','TCN':'TCN\nDilated convolution','GRU':'GRU\nGated recurrence'}
    for j, name in enumerate(names):
        x=.322+(j%4)*.164; y=.37 if j<4 else .26
        box(x,y,.142,.085,labels[name], '#ffffff','#b4bdc5',10.7)
    arrow((.88,.665),(.88,.55))
    box(.025,.285,.22,.18,'C4: optional FSC-v1 training\nFeature-statistics perturbation\nBCE + binary JS consistency\nInference input remains C3','#fff5e8','#c4924d',10.7)
    arrow((.245,.375),(.294,.375),True)
    box(.48,.085,.32,.085,'Common classification head → anomaly score','#eff3f6','#718391',12)
    arrow((.64,.245),(.64,.178))
    ax.text(.5,.027,'* Custom target–aggregate-reference graph; C0 disables reference-node edges. All eight encoders are explicit task adaptations.',ha='center',fontsize=10.3)
    save_figure(fig,'fig1_framework'); plt.close(fig)
    save_json(FIG/'fig1_framework.json', {'inventory_sha256':sha(ROOT/'model_inventory.json'),'encoder_names':names,'encoder_count':len(names),'script_sha256':sha(__file__),'claim':'Architecture schematic only; no performance values'})
    print('FIG1_RENDERED',flush=True)


def infer_prefixes(device_name='cpu'):
    # Gate before importing torch, loading any checkpoint or reading scores.
    selection = json.loads(SELECTION.read_text())
    checkpoint_lock = json.loads((ROOT/'checkpoint_lock.json').read_text())
    protocol = json.loads((ROOT/'protocol.json').read_text())
    expected = len(protocol['tasks'])*len(protocol['backbones'])*len(protocol['seeds'])*len(protocol['configurations'])
    assert checkpoint_lock['count'] == expected == 240
    assert checkpoint_lock['code_lock_sha256'] == sha(ROOT/'code_lock.json') == selection['code_lock_sha256']
    assert selection['created_utc'] < checkpoint_lock['created_utc']
    code_lock=json.loads((ROOT/'code_lock.json').read_text())
    for name in ('models.py','train_study.py','protocol.json'):
        assert sha(ROOT/name)==code_lock['files'][name]
    import torch
    from models import make_model
    from train_study import normalize, times_array
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device=torch.device(device_name)
    datasets = {task: load_npz(ROOT/'data'/task/'test.npz') for task in protocol['tasks']}
    for task in datasets: assert sha(ROOT/'data'/task/'test.npz') == selection['test_data_sha256'][task]
    results = {'selection_sha256':sha(SELECTION),'checkpoint_lock_sha256':sha(ROOT/'checkpoint_lock.json'),'device':str(device),'seed':SEED,
               'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'precision':'float32; TF32 disabled',
               'prefix_scope':selection['protocol']['relation_semantics'],'series':[],'script_sha256':sha(__file__)}
    for case in selection['cases']:
        data = datasets[case['task']]; i=case['index']; n=len(data['y'])
        assert str(data['window_id'][i]) == case['window_id']
        sample = {k:v[i:i+1] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v for k,v in data.items()}
        for arch in ARCHITECTURES:
            for arm in ARMS:
                relative = f'runs/{case["task"]}/{arch}/seed_{SEED}/{arm}/final.pt'
                path=ROOT/relative; assert sha(path)==checkpoint_lock['checkpoints'][relative]
                saved=torch.load(path,map_location='cpu',weights_only=False)
                model=make_model(arch,data['base__C0'].shape[-1],max_len=20,hidden=64).to(device);model.load_state_dict(saved['model']);model.eval()
                x,_=normalize(sample,'base__'+arm,saved['normalization']);times=times_array(sample)
                lengths=list(range(2,case['length']+1));scores=[]
                with torch.no_grad():
                    for length in lengths:
                        pred=model(torch.tensor(x[:,:length],device=device),torch.tensor(sample['mask'][:,:length],device=device),torch.tensor(times[:,:length],device=device))['logits'].sigmoid().item()
                        assert np.isfinite(pred);scores.append(pred)
                threshold=float(saved['thresholds']['0.01']['threshold'])
                core_prediction=load_npz(path.with_name('test_base.npz'))
                assert str(core_prediction['window_id'][i])==case['window_id']
                endpoint_error=abs(scores[-1]-float(core_prediction['score'][i]))
                if device.type=='cuda':assert endpoint_error<=1e-5,(case['case_key'],arch,arm,endpoint_error)
                results['series'].append({'case_key':case['case_key'],'window_id':case['window_id'],'architecture':arch,'arm':arm,
                                          'checkpoint':relative,'checkpoint_sha256':sha(path),'prefix_lengths':lengths,
                                          'elapsed_minutes':(times[0,np.asarray(lengths)-1]/60).tolist(),'scores':scores,
                                          'validation_full_window_threshold_1pct':threshold,'threshold_margins':(np.asarray(scores)-threshold).tolist()})
                results['series'][-1]['full_window_endpoint_vs_saved_core_test_abs_error']=endpoint_error
                print('PREFIX_DONE',case['case_key'],arch,arm,flush=True)
    cpu_path=FIG/'fig8_cases.cpu.prefixes.json'
    if device.type=='cuda' and cpu_path.exists():
        previous=json.loads(cpu_path.read_text());assert previous['selection_sha256']==results['selection_sha256']
        key=lambda r:(r['case_key'],r['architecture'],r['arm'])
        before={key(r):r for r in previous['series']}
        results['cpu_to_cuda_replay']={'cpu_prefixes_sha256':sha(cpu_path),'max_abs_prefix_score_difference':max(float(np.max(np.abs(np.asarray(r['scores'])-before[key(r)]['scores']))) for r in results['series']),
                                      'same_frozen_case_selection':True,'same_frozen_checkpoints':True}
    save_json(FIG/'fig8_cases.prefixes.json',results)


def case_geometry(case, data):
    raw=load_npz(Path(case['source_archive']));assert sha(case['source_archive'])==case['source_archive_sha256']
    i=case['index'];k=case['source_row_or_case'];n=case['length'];x=data['base__C3'][i,:n]
    t=data['time_s'][i,:n];time_minutes=(t-t[0])/60
    if case['task']=='individual':
        observed=raw['target_pos'][k,:n];reference=raw['pre_noise_pos'][k,:n]
        reliability=raw['self_reliability'][k,:n].astype(float)
        target_velocity=x[:,2:4];background_velocity=x[:,7:9]
        speed=np.linalg.norm(target_velocity,axis=1);background=np.linalg.norm(background_velocity,axis=1);residual=np.linalg.norm(x[:,9:11],axis=1)
        neighbors=[]
        if case['case_key']=='target_teleport':
            changed=np.flatnonzero(np.linalg.norm(observed-reference,axis=1)>1e-6)
            event=float(time_minutes[changed[0]]) if len(changed) else None
        else:event=(case['recorded_behavior_onset_s']-t[0])/60
        reliability_label='Target compatibility'
    else:
        from data_builder import absolute_weights
        observed=raw['target_pos'][k];reference=raw['pre_noise_target_pos'][k]
        reliability,_=absolute_weights(raw['target_t'][k],observed,raw['target_sog_kn'][k],raw['target_cog_deg'][k])
        speed=x[:,0]*10*KNOT;background=x[:,4]*10*KNOT;residual=np.abs(speed-background)
        neighbors=[raw['neighbor_pos'][k,int(a):int(b)] for a,b in zip(raw['member_offsets'][:-1],raw['member_offsets'][1:])]
        event=None;target_velocity=None;background_velocity=None
        reliability_label='Target compatibility'
    origin=observed[0].copy()
    return {'time_minutes':time_minutes,'observed':(observed-origin)/1000,'reference':(reference-origin)/1000,
            'neighbors':[(p-origin)/1000 for p in neighbors],'speed':speed,'background_speed':background,'residual':residual,
            'reliability':reliability,'support':x[:,-1]>0,'event_minutes':event,'background_velocity':background_velocity,
            'reliability_label':reliability_label}


def render_cases():
    selection=json.loads(SELECTION.read_text());pred=json.loads((FIG/'fig8_cases.prefixes.json').read_text())
    assert pred['selection_sha256']==sha(SELECTION)
    assert pred['checkpoint_lock_sha256']==sha(ROOT/'checkpoint_lock.json')
    plt=configure_plot()
    from matplotlib.lines import Line2D
    datasets={task:load_npz(ROOT/'data'/task/'test.npz') for task in ('individual','relation')}
    fig=plt.figure(figsize=(15.8,11.6));grid=fig.add_gridspec(3,4,height_ratios=[1.1,1.05,1],left=.055,right=.985,bottom=.23,top=.84,hspace=.43,wspace=.29)
    archcolors={'tcn':'#356c9b','gru':'#8e5a97','transformer':'#238576'}
    coordinates=[]
    for col,case in enumerate(selection['cases']):
        g=case_geometry(case,datasets[case['task']]);time=g['time_minutes'];valid=g['support'];coordinates.append(g)
        ax=fig.add_subplot(grid[0,col]);obs,ref=g['observed'],g['reference']
        for neighbor in g['neighbors']:ax.plot(neighbor[:,0],neighbor[:,1],color='#a9b0b6',lw=.8,alpha=.65)
        ax.plot(ref[:,0],ref[:,1],color='#787f86',lw=2.4,ls='--',label='Pre-noise reference')
        ax.plot(obs[:,0],obs[:,1],color='#263746',lw=1.1,marker='o',ms=2.8,label='Observed target')
        ax.scatter(obs[0,0],obs[0,1],marker='s',s=22,color='#263746',zorder=4)
        if g['background_velocity'] is not None:
            ix=np.flatnonzero(valid)[::max(1,int(valid.sum()/5))];v=g['background_velocity'][ix]*60/1000
            ax.quiver(obs[ix,0],obs[ix,1],v[:,0],v[:,1],angles='xy',scale_units='xy',scale=1,color='#279184',width=.007)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Relative east (km)')
        if col==0:ax.set_ylabel('Relative north (km)')
        ax.set_title(f'({chr(97+col)}) {case["title"]}\n{case["region"].replace("_"," ")}; y = {case["label"]}',pad=8)
        ax.text(.02,.02,case['window_id'][:12],transform=ax.transAxes,fontsize=8.7,color='#515b64')
        middle=grid[1,col].subgridspec(2,1,height_ratios=[3.1,1],hspace=.12)
        ax=fig.add_subplot(middle[0]);ax.plot(time,g['speed'],color='#424a52',lw=1.3,label='Reported target speed')
        ax.plot(time,np.where(valid,g['background_speed'],np.nan),color='#238576',lw=1.3,label='C3 reference speed')
        ax.plot(time,np.where(valid,g['residual'],np.nan),color='#b97635',lw=1.15,ls='--',label='Residual magnitude')
        if col==0:ax.set_ylabel('Speed / residual (m/s)')
        ax.tick_params(labelbottom=False)
        if g['event_minutes'] is not None:ax.axvline(g['event_minutes'],color='#89939b',ls=':',lw=1)
        ax2=fig.add_subplot(middle[1],sharex=ax);ax2.plot(time,g['reliability'],color='#7b5c92',lw=1.1)
        ax2.step(time,valid.astype(float),where='post',color='#238576',lw=.9,ls='--');ax2.set_ylim(-.1,1.1);ax2.set_yticks([0,1]);ax2.set_xlabel('Observed elapsed time (min)')
        if col==0:ax2.set_ylabel('Weight',fontsize=9)
        ax=fig.add_subplot(grid[2,col]);ax.axhline(0,color='#535b63',lw=1,ls=':')
        for item in pred['series']:
            if item['case_key']!=case['case_key']:continue
            ax.plot(item['elapsed_minutes'],item['threshold_margins'],color=archcolors[item['architecture']],ls='--' if item['arm']=='C0' else '-',lw=1.25,marker=None)
        if g['event_minutes'] is not None:ax.axvline(g['event_minutes'],color='#89939b',ls=':',lw=1)
        ax.set_ylim(-1,1);ax.set_xlabel('Observed prefix duration (min)')
        if col==0:ax.set_ylabel('Score − validation threshold')
        if case['task']=='relation':ax.set_title('Offline cohort prefix diagnostic',fontsize=10.5)
    fig.suptitle('Fixed-source trajectory, context and observed-prefix responses',fontsize=18,y=.965)
    top_handles=[Line2D([],[],color='#263746',marker='o',lw=1,label='Observed target'),Line2D([],[],color='#787f86',ls='--',lw=2,label='Pre-observation-noise reference'),Line2D([],[],color='#a9b0b6',lw=1,label='Recorded cohort tracks (a)'),Line2D([],[],color='#279184',marker='>',lw=0,label='C3 reference velocity × 60 s (b–d)')]
    fig.legend(handles=top_handles,loc='upper center',bbox_to_anchor=(.52,.925),ncol=4,frameon=False,fontsize=9.5)
    handles=[Line2D([],[],color=c,lw=1.5,label=a.upper() if a!='transformer' else 'Transformer') for a,c in archcolors.items()]
    handles += [Line2D([],[],color='#4a5158',ls='--',label='C0'),Line2D([],[],color='#4a5158',ls='-',label='C3'),Line2D([],[],color='#7b5c92',label='Compatibility weight'),Line2D([],[],color='#238576',ls='--',label='Reference valid')]
    handles += [Line2D([],[],color='#424a52',label='Reported target speed'),Line2D([],[],color='#238576',label='C3 reference speed'),Line2D([],[],color='#b97635',ls='--',label='Residual magnitude')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.115),ncol=5,frameon=False,fontsize=9.5)
    fig.text(.055,.078,'Seed 0; cases chosen by fixed metadata rules and sorted window IDs before scores. Prefix predictions use only the available input prefix.',fontsize=9.8)
    fig.text(.055,.051,'Margins use each model’s frozen full-window validation 1% FAR threshold; prefix FAR is not calibrated. Relation membership uses the full window (offline diagnostic).',fontsize=9.8)
    fig.text(.055,.024,'Pre-noise references may contain the controlled behavior. Residuals: vector norm in (b–d), absolute speed difference in (a). Dotted vertical lines mark recorded events.',fontsize=9.8)
    save_figure(fig,'fig8_cases');plt.close(fig)
    save_json(FIG/'fig8_cases.json',{'selection_sha256':sha(SELECTION),'prefix_predictions_sha256':sha(FIG/'fig8_cases.prefixes.json'),'script_sha256':sha(__file__),'case_keys':[c['case_key'] for c in selection['cases']],'no_coordinate_cumsum':True,'no_score_based_case_selection':True})
    def serializable(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,list):return [serializable(v) for v in value]
        if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
        return value
    save_json(FIG/'fig8_cases.geometry.json',{'selection_sha256':sha(SELECTION),'units':{'coordinates':'km relative to first observed target point','speed_and_residual':'m/s','time':'elapsed minutes','background_velocity':'m/s'},
                                            'cases':[{'metadata':case,'plotted_values':serializable(g)} for case,g in zip(selection['cases'],coordinates)]})
    print('FIG8_RENDERED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['select','framework','infer','render']);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    FIG.mkdir(exist_ok=True)
    {'select':select_cases,'framework':render_framework,'infer':lambda:infer_prefixes(args.device),'render':render_cases}[args.mode]()
