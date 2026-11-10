"""Same frozen cases, primary individual field arms plus a separate scalar contrast.

prepare: freeze display protocol without scores. infer: requires field72 + core240
checkpoint locks. render: AIS .venv matplotlib. Prefer the original CUDA path for
final inference because the CPU Transformer has a documented numerical difference.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import numpy as np
from render_cases import sha,load_npz,save_json,configure_plot,case_geometry,save_figure

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'
FIELD=ROOT/'field_extension'
DISPLAY_PROTOCOL=FIG/'fig8_kinematic_loo_mainline.protocol.json'
ARCHES=('tcn','gru','transformer')
FIELD_ARMS=('F0','F1','F2')
RELATION_ARMS=('C2','C3')
SEED=0


def prepare():
    original=json.loads((ROOT/'case_selection.json').read_text())
    expected={'target_teleport':'027c946512ca833a72091141','early_turn_back':'11c0882ffb5b7ffb4775009f','deviation':'00cb0766297afe488e6feec8',
              'coherent_group':'00c96e7349a9b90190927f5d:case3'}
    assert {c['case_key']:c['window_id'] for c in original['cases']}==expected
    if DISPLAY_PROTOCOL.exists():
        prior=json.loads(DISPLAY_PROTOCOL.read_text());assert prior['original_selection_sha256']==sha(ROOT/'case_selection.json')
        print('FIELD_CASE_PROTOCOL_ALREADY_FROZEN',sha(DISPLAY_PROTOCOL),flush=True);return
    protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'fixed_before_field_checkpoint_lock_and_case_inference',
              'original_selection_sha256':sha(ROOT/'case_selection.json'),'original_selection_created_utc':original['created_utc'],
              'field_protocol_sha256':sha(ROOT/'field_protocol.json'),'cases':original['cases'],'architectures':list(ARCHES),'seed':SEED,
              'individual_arms':list(FIELD_ARMS),'relation_arms':list(RELATION_ARMS),
              'scientific_scope':{'individual':'kinematic compatibility plus leave-target-out local-affine field; F2 is the optional causal target package',
                                  'relation':'independent scalar-context C2/C3 contrast; offline full-window cohort; not a local-affine field case'},
              'field_arm_meanings':{'F0':'distance-only target-excluded local-affine field','F1':'distance times absolute-compatibility target-excluded local-affine field','F2':'same F1 field plus audited causal target package'},
              'prefixes':'all available observed prefixes of length2..T; fixed training-only normalization',
              'thresholds':'own frozen full-window validation 1% nominal FAR; no prefix FAR calibration',
              'coordinates':'raw and pre-noise positions are recorded coordinates; F2 slots0:2 already cumulative relative positions; no cumsum',
              'case_changes_permitted':False,'score_based_selection':False,'retain_failures_and_low_support':True,
              'weights_required':'all72 field and all240 core final checkpoints locked before inference',
              'curve_legends':'individual F0/F1/F2 and relation scalar C2/C3 kept separate'}
    save_json(DISPLAY_PROTOCOL,protocol,exclusive=True);print('FIELD_CASE_PROTOCOL_FROZEN',sha(DISPLAY_PROTOCOL),flush=True)


def checked_inputs():
    protocol=json.loads(DISPLAY_PROTOCOL.read_text())
    assert protocol['original_selection_sha256']==sha(ROOT/'case_selection.json')
    assert protocol['field_protocol_sha256']==sha(ROOT/'field_protocol.json')
    core_lock=json.loads((ROOT/'checkpoint_lock.json').read_text());field_lock=json.loads((FIELD/'checkpoint_lock.json').read_text())
    assert core_lock['count']==240 and field_lock['count']==72
    assert field_lock['field_code_lock_sha256']==sha(FIELD/'code_lock.json')
    # The cases were selected before any scores; the display protocol may be
    # registered after training finishes. Do not conflate those two dates.
    assert protocol['original_selection_created_utc']<field_lock['created_utc']
    if protocol['created_utc']>=field_lock['created_utc']:
        assert protocol['chronology']['display_registration_after_field_checkpoint_lock'] is True
        assert protocol['chronology']['field_predictions_read_before_display_registration'] is False
    frozen=json.loads((FIELD/'code_lock.json').read_text())
    for name,h in frozen['parent_files'].items():assert sha(ROOT/name)==h,name
    for name,h in frozen['field_files'].items():assert sha(ROOT/name)==h,name
    data={task:load_npz(ROOT/'data'/task/'test.npz') for task in ('individual','relation')}
    extension=load_npz(ROOT/'extra_data/field_test.npz')
    for key in ('window_id','mask','time_s','y'):np.testing.assert_array_equal(extension[key],data['individual'][key])
    for arm in FIELD_ARMS:data['individual']['base__'+arm]=extension['base__'+arm]
    return protocol,core_lock,field_lock,data


def infer(device_name):
    protocol,core_lock,field_lock,data=checked_inputs()
    import torch
    from models import make_model
    from train_study import normalize,times_array
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    device=torch.device(device_name)
    result={'display_protocol_sha256':sha(DISPLAY_PROTOCOL),'original_selection_sha256':protocol['original_selection_sha256'],
            'field_checkpoint_lock_sha256':sha(FIELD/'checkpoint_lock.json'),'core_checkpoint_lock_sha256':sha(ROOT/'checkpoint_lock.json'),
            'device':str(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'precision':'FP32, TF32 disabled','seed':SEED,'series':[],
            'individual_field_arms':list(FIELD_ARMS),'independent_relation_scalar_arms':list(RELATION_ARMS),'script_sha256':sha(__file__)}
    for case in protocol['cases']:
        d=data[case['task']];i=case['index'];n=len(d['y']);assert str(d['window_id'][i])==case['window_id']
        sample={k:v[i:i+1] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v for k,v in d.items()}
        is_field=case['task']=='individual';parent=FIELD if is_field else ROOT;lock=field_lock if is_field else core_lock
        for arch in ARCHES:
            for arm in FIELD_ARMS if is_field else RELATION_ARMS:
                relative=f'runs/{case["task"]}/{arch}/seed_{SEED}/{arm}/final.pt';cp=parent/relative
                assert sha(cp)==lock['checkpoints'][relative]
                saved=torch.load(cp,map_location='cpu',weights_only=False)
                model=make_model(arch,d['base__'+arm].shape[-1],max_len=20,hidden=64).to(device);model.load_state_dict(saved['model']);model.eval()
                x,_=normalize(sample,'base__'+arm,saved['normalization']);times=times_array(sample);lengths=list(range(2,case['length']+1));scores=[]
                with torch.no_grad():
                    for length in lengths:
                        output=model(torch.tensor(x[:,:length],device=device),torch.tensor(sample['mask'][:,:length],device=device),torch.tensor(times[:,:length],device=device))['logits']
                        value=output.sigmoid().item();assert np.isfinite(value);scores.append(value)
                threshold=float(saved['thresholds']['0.01']['threshold']);endpoint=None
                if cp.with_name('test_base.npz').exists():
                    ref=load_npz(cp.with_name('test_base.npz'));assert str(ref['window_id'][i])==case['window_id']
                    endpoint=abs(scores[-1]-float(ref['score'][i]))
                    if device.type=='cuda':assert endpoint<=1e-5,(case['case_key'],arch,arm,endpoint)
                result['series'].append({'case_key':case['case_key'],'window_id':case['window_id'],'experiment_family':'individual_loo_field' if is_field else 'independent_relation_scalar',
                                         'architecture':arch,'arm':arm,'checkpoint':str(cp.relative_to(ROOT)).replace('\\','/'),'checkpoint_sha256':sha(cp),
                                         'prefix_lengths':lengths,'elapsed_minutes':(times[0,np.asarray(lengths)-1]/60).tolist(),'scores':scores,
                                         'full_window_validation_1pct_threshold':threshold,'threshold_margins':(np.asarray(scores)-threshold).tolist(),
                                         'full_window_endpoint_vs_saved_test_abs_error':endpoint,
                                         'endpoint_reference_status':'checked' if endpoint is not None else 'official full-window export not yet available'})
                print('FIELD_PREFIX_DONE',case['case_key'],arch,arm,flush=True)
    assert len(result['series'])==33
    save_json(FIG/'fig8_kinematic_loo_mainline.prefixes.json',result)


def render():
    protocol,_,_,data=checked_inputs();prediction_path=FIG/'fig8_kinematic_loo_mainline.prefixes.json';pred=json.loads(prediction_path.read_text())
    assert pred['display_protocol_sha256']==sha(DISPLAY_PROTOCOL)
    assert pred['field_checkpoint_lock_sha256']==sha(FIELD/'checkpoint_lock.json')
    known_errors=[s['full_window_endpoint_vs_saved_test_abs_error'] for s in pred['series'] if s['full_window_endpoint_vs_saved_test_abs_error'] is not None]
    assert not known_errors or max(known_errors)<=1e-5,'Final figure needs numerical agreement with official full-window predictions'
    plt=configure_plot()
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(16.3,12.2));grid=fig.add_gridspec(3,4,height_ratios=[1.1,1.05,1],left=.057,right=.985,bottom=.27,top=.83,hspace=.44,wspace=.29)
    colors={'tcn':'#356c9b','gru':'#8e5a97','transformer':'#238576'};field_styles={'F0':':','F1':'-','F2':'--'};relation_styles={'C2':'--','C3':'-'};geometry=[]
    for col,case in enumerate(protocol['cases']):
        d=data[case['task']];g=case_geometry(case,d);i=case['index'];n=case['length'];time=g['time_minutes'];is_field=case['task']=='individual'
        if is_field:
            f0,f1,f2=[d['base__'+arm][i,:n] for arm in FIELD_ARMS]
            np.testing.assert_array_equal(f1[:,7:9],f2[:,7:9]);np.testing.assert_array_equal(f0[:,-1],f1[:,-1]);np.testing.assert_array_equal(f1[:,-1],f2[:,-1])
            g['support']=f1[:,-1]>0;g['field_F0_background_mps']=f0[:,7:9];g['field_F1_background_mps']=f1[:,7:9]
            g['field_F1_residual_mps']=f1[:,9:11];g['field_F2_residual_mps']=f2[:,9:11];g['causal_F2_position_km']=f2[:,:2]/1000
        geometry.append({'case_metadata':case,'plotted_values':g})
        ax=fig.add_subplot(grid[0,col]);obs,ref=g['observed'],g['reference']
        for neighbor in g['neighbors']:ax.plot(neighbor[:,0],neighbor[:,1],color='#b2b8bd',lw=.8,alpha=.65)
        ax.plot(ref[:,0],ref[:,1],color='#80868c',lw=2.2,ls='--');ax.plot(obs[:,0],obs[:,1],color='#293b49',lw=1.15,marker='o',ms=2.5)
        ax.scatter(obs[0,0],obs[0,1],marker='s',s=20,color='#293b49',zorder=4)
        if is_field:
            gate=g['causal_F2_position_km'];ax.plot(gate[:,0],gate[:,1],color='#be8848',lw=.95,ls=':',alpha=.9)
            ix=np.flatnonzero(g['support'])[::max(1,int(g['support'].sum()/5))];v=g['field_F1_background_mps'][ix]*60/1000
            ax.quiver(obs[ix,0],obs[ix,1],v[:,0],v[:,1],angles='xy',scale_units='xy',scale=1,color='#238576',width=.006)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Relative east (km)')
        if col==0:ax.set_ylabel('Relative north (km)')
        scope='Scalar relation contrast' if not is_field else 'LOO local-affine field'
        ax.set_title(f'({chr(97+col)}) {case["title"]}\n{scope}; y = {case["label"]}',fontsize=11.4,pad=9)
        ax.text(.02,.02,case['window_id'][:12],transform=ax.transAxes,fontsize=8.3,color='#59626b')
        mid=grid[1,col].subgridspec(2,1,height_ratios=[3.1,1],hspace=.13)
        ax=fig.add_subplot(mid[0]);valid=g['support'];ax.plot(time,g['speed'],color='#414a52',lw=1.2)
        if is_field:
            for arm,vel,color,style in [('F0',g['field_F0_background_mps'],'#8797a0',':'),('F1',g['field_F1_background_mps'],'#238576','-')]:
                ax.plot(time,np.where(valid,np.linalg.norm(vel,axis=1),np.nan),color=color,ls=style,lw=1.25)
            ax.plot(time,np.where(valid,np.linalg.norm(g['field_F1_residual_mps'],axis=1),np.nan),color='#b97635',ls='--',lw=1.15)
        else:
            c2,c3=d['base__C2'][i,:n],d['base__C3'][i,:n]
            ax.plot(time,c2[:,4]*10*(1852/3600),color='#8797a0',ls=':',lw=1.25)
            ax.plot(time,c3[:,4]*10*(1852/3600),color='#238576',lw=1.25)
            ax.plot(time,np.abs(g['speed']-c3[:,4]*10*(1852/3600)),color='#b97635',ls='--',lw=1.15)
        if col==0:ax.set_ylabel('Speed / residual (m/s)')
        ax.tick_params(labelbottom=False)
        if g['event_minutes'] is not None:ax.axvline(g['event_minutes'],color='#89939b',ls=':',lw=.9)
        small=fig.add_subplot(mid[1],sharex=ax);small.plot(time,g['reliability'],color='#7b5c92',lw=1)
        small.step(time,valid.astype(float),where='post',color='#238576',ls='--',lw=.9);small.set_ylim(-.1,1.1);small.set_yticks([0,1]);small.set_xlabel('Elapsed time (min)')
        if col==0:small.set_ylabel('Weight',fontsize=9)
        ax=fig.add_subplot(grid[2,col]);ax.axhline(0,color='#535b63',lw=.9,ls=':')
        for row in pred['series']:
            if row['case_key']!=case['case_key']:continue
            styles=field_styles if is_field else relation_styles
            ax.plot(row['elapsed_minutes'],row['threshold_margins'],color=colors[row['architecture']],ls=styles[row['arm']],lw=1.2)
        if g['event_minutes'] is not None:ax.axvline(g['event_minutes'],color='#89939b',ls=':',lw=.9)
        ax.set_ylim(-1,1);ax.set_xlabel('Observed prefix duration (min)')
        if col==0:ax.set_ylabel('Score − own validation threshold');ax.set_title('Offline cohort prefix diagnostic',fontsize=10.3)
    fig.suptitle('Kinematic / leave-target-out field responses on the original fixed cases',fontsize=18,y=.965)
    top=[Line2D([],[],color='#293b49',marker='o',label='Observed target'),Line2D([],[],color='#80868c',ls='--',label='Pre-noise reference'),
         Line2D([],[],color='#be8848',ls=':',label='Optional F2 target path (b–d)'),Line2D([],[],color='#238576',marker='>',ls='',label='F1 field velocity × 60 s (b–d)')]
    fig.legend(handles=top,ncol=4,loc='upper center',bbox_to_anchor=(.52,.925),frameon=False,fontsize=9.3)
    curves=[Line2D([],[],color=colors[a],lw=1.5,label={'tcn':'TCN','gru':'GRU','transformer':'Transformer'}[a]) for a in ARCHES]
    fig.legend(handles=curves,ncol=3,loc='lower center',bbox_to_anchor=(.52,.192),frameon=False,fontsize=10)
    field_legend=[Line2D([],[],color='#434b53',ls=field_styles[a],label={'F0':'F0: distance-only field','F1':'F1: reliable field','F2':'F2: reliable field + target package'}[a]) for a in FIELD_ARMS]
    relation_legend=[Line2D([],[],color='#434b53',ls=relation_styles[a],label={'C2':'C2: scalar median','C3':'C3: reliable scalar median'}[a]) for a in RELATION_ARMS]
    fig.legend(handles=relation_legend,ncol=1,title='Independent relation column (a)',loc='lower center',bbox_to_anchor=(.22,.116),frameon=False,fontsize=9.3,title_fontsize=10)
    fig.legend(handles=field_legend,ncol=3,title='Individual field columns (b–d)',loc='lower center',bbox_to_anchor=(.68,.135),frameon=False,fontsize=9.3,title_fontsize=10)
    fig.text(.057,.094,'Field middle rows (b–d): target speed (dark gray), F0 field (gray dotted), F1 field (green), F1 residual magnitude (orange dashed).',fontsize=9.3)
    fig.text(.057,.076,'Relation middle row (a): target speed, scalar C2 reference (gray dotted), scalar C3 reference (green), absolute scalar residual (orange dashed).',fontsize=9.3)
    fig.text(.057,.058,'Small strip: target compatibility weight (purple) and shared reference validity (green dashed). F2 keeps the original observed target as its field query.',fontsize=9.3)
    fig.text(.057,.040,'Same seed-0 models and original metadata-selected cases; no score-based replacements. Relation membership uses the full window; its prefixes are offline diagnostics.',fontsize=9.3)
    fig.text(.057,.022,'Margins use frozen full-window validation 1% nominal FAR thresholds; prefix FAR is not calibrated. Pre-noise reference is not external positioning ground truth.',fontsize=9.3)
    save_figure(fig,'fig8_kinematic_loo_mainline');plt.close(fig)
    def serializable(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,list):return [serializable(x) for x in v]
        if isinstance(v,dict):return {k:serializable(x) for k,x in v.items()}
        return v
    save_json(FIG/'fig8_kinematic_loo_mainline.geometry.json',{'protocol_sha256':sha(DISPLAY_PROTOCOL),'coordinates':'km, single origin subtraction only','velocity':'m/s','cases':serializable(geometry)})
    save_json(FIG/'fig8_kinematic_loo_mainline.json',{'protocol_sha256':sha(DISPLAY_PROTOCOL),'prefix_predictions_sha256':sha(prediction_path),'renderer_sha256':sha(__file__),
                                                'field_and_scalar_legends_separate':True,'same_original_cases':True,'series_count':len(pred['series'])})
    print('FIELD_CASE_FIGURE_RENDERED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','infer','render'],required=True);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    {'prepare':prepare,'infer':lambda:infer(args.device),'render':render}[args.mode]()
