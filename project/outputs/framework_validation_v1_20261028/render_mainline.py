"""Research schematic for the audited kinematic / leave-target-out field mainline.

Reads only implementation protocols/inventory. No model scores or weights.
Preserves the earlier median-reference Figure 1 under its original filename.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures'


def main():
    protocol=json.loads((ROOT/'field_protocol.json').read_text())
    fit=protocol['fixed_fit']
    assert fit['R_nm']==8 and fit['distance_sigma_nm']==4 and fit['ridge_lambda']==.001
    assert fit['ridge_penalty']=='diag(0,1,1); intercept unpenalized'
    assert 'original observed target position' in fit['design']
    assert 'target identity exclusion' in fit['normalization']
    assert 'same field' in protocol['target_gate']['background_reference']
    inventory=json.loads((ROOT/'model_inventory.json').read_text())
    names=[m['name'] for m in inventory['models']]
    assert len(names)==8 and len(set(names))==8
    FIG.mkdir(exist_ok=True)
    for name in ('fig1_framework.font_regular.ttf','fig1_framework.font_bold.ttf'):
        font_manager.fontManager.addfont(str(FIG/name))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(16.2,8.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,face='#f7f9fa',edge='#7c8a95',size=11.7,dashed=False):
        patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.007,rounding_size=0.012',linewidth=1.2,
                            facecolor=face,edgecolor=edge,linestyle='--' if dashed else '-')
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.45)
    def arrow(a,b,dashed=False,color='#4c5b65'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.25,color=color,
                                    linestyle='--' if dashed else '-'))
    ax.text(.5,.97,'Kinematic compatibility and leave-target-out local-affine context',ha='center',fontsize=19)
    ax.text(.5,.925,'A shared background field and individual residual, followed by any registered trajectory encoder',ha='center',fontsize=12.5,color='#4c5962')
    top_y=.705;top_h=.17
    box(.018,top_y,.152,top_h,'Observed reports\nPosition, velocity, time\nTarget + neighbor tracks\nValid observations',size=11.4)
    box(.205,top_y,.168,top_h,'Kinematic consistency\nPosition/report mismatch e (m/s)\nAbsolute weight w_abs\nInvalid history → weight 0',face='#edf6f4',edge='#2d8c7d',size=11.3)
    box(.410,top_y,.157,top_h,'Leave target i out\nExclude target identity\nbefore normalization\nSame remaining members',face='#edf6f4',edge='#2d8c7d',size=11.2)
    box(.604,top_y,.168,top_h,'Local-affine fit\nF0: distance weight\nF1: distance × w_abs\nWeighted ridge regression',face='#e6f3ef',edge='#2d8c7d',size=11.2)
    box(.812,top_y,.17,top_h,'Background and residual\nQuery at observed p_i\nV_−i = fitted intercept\nr_i = v_i − V_−i',face='#e6f3ef',edge='#2d8c7d',size=11.3)
    for end,start in ((.170,.205),(.373,.410),(.567,.604),(.772,.812)):
        arrow((end,top_y+top_h/2),(start,top_y+top_h/2))
    box(.018,.43,.255,.19,'F2: optional causal target package\nCausal position handling +\nlow-weight speed/course holding\nSame F1 field, query and support',face='#fff7eb',edge='#bd8d4b',size=11.4,dashed=True)
    arrow((.092,.705),(.092,.625),True,'#a7804d')
    box(.330,.435,.355,.18,'Fixed field definition\nX = [1, dx/R, dy/R], relative to observed target\nB = solve(X^T W X + λ diag(0,1,1), X^T W V)\nR = 8 nm; distance scale = 4 nm; λ = 0.001',face='#fbfcfc',edge='#adb7be',size=11.1)
    box(.735,.435,.247,.18,'Fixed 12-slot interface\nTarget 7 | Background 2\nResidual 2 | Shared support 1\nF0/F1 raw target; F2 causal package',face='#edf6f4',edge='#2d8c7d',size=11.1)
    arrow((.898,.705),(.898,.622))
    ax.plot([.273,.300,.300,.718,.718],[.465,.465,.368,.368,.500],color='#a7804d',lw=1.25,ls='--')
    arrow((.718,.500),(.731,.500),True,'#a7804d')
    ax.text(.49,.376,'Optional target input only',ha='center',fontsize=9.6,color='#9a7545')
    ax.text(.503,.401,'Weights normalized after target exclusion; unpenalized intercept',ha='center',fontsize=10.3,color='#52616b')
    box(.260,.135,.722,.205,'',face='#fbfcfd',edge='#adb7bf')
    ax.text(.62,.316,'Eight distinct classification adaptations · common classifier head',ha='center',fontsize=12)
    labels={'TCN':'TCN','GRU':'GRU','Transformer':'Transformer','iTransformer':'iTransformer',
            'PatchTST':'PatchTST','Mamba':'Mamba / selective SSM','Neural-CDE':'Neural CDE / RK4','GraphTemporal':'Two-node GAT + TCN*'}
    for j,name in enumerate(names):
        box(.280+(j%4)*.175,.233 if j<4 else .155,.153,.052,labels[name],face='white',edge='#b4bdc5',size=10.7)
    arrow((.857,.435),(.857,.346))
    ax.text(.018,.303,'Absolute compatibility weight',ha='left',fontsize=11.3)
    ax.text(.018,.270,'w_abs = exp(−max(e − 2.5, 0)^2 / 8)',ha='left',fontsize=10.8)
    ax.text(.018,.229,'Distance weight',ha='left',fontsize=11.3)
    ax.text(.018,.196,'w_dist = exp(−0.5 × (d / 7408 m)^2)',ha='left',fontsize=10.5)
    ax.text(.5,.088,'Shared support: inherited validity ∩ both field fits valid; at least 4 neighbors; unregularized weighted-Gram condition ≤ 10^8.',ha='center',fontsize=10.6)
    ax.text(.5,.048,'* Custom target–aggregate-reference graph. Mean, median and weighted-median references are separate comparison branches.',ha='center',fontsize=10.5)
    for ext in ('png','svg','pdf'):
        fig.savefig(FIG/f'fig1_kinematic_loo_mainline.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    metadata={'field_protocol_sha256':hashlib.sha256((ROOT/'field_protocol.json').read_bytes()).hexdigest(),
              'model_inventory_sha256':hashlib.sha256((ROOT/'model_inventory.json').read_bytes()).hexdigest(),
              'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'encoder_names':names,
              'target_excluded_before_weight_normalization':True,'query_reference':'original observed target position',
              'target_gate':'optional F2 package; same F1 field and query','score_data_read':False,
              'earlier_median_figure_preserved':True}
    (FIG/'fig1_kinematic_loo_mainline.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('MAINLINE_DIAGRAM_RENDERED',flush=True)


if __name__=='__main__':main()
