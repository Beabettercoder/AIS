"""Fill forecast tables from completed predictions, repair figure reference, compile."""
from __future__ import annotations
import argparse,re,shutil,subprocess,os
from pathlib import Path
from common import *

METHOD=r'''
\subsection{Forecasting-specific causal re-anchoring}
The forecasting check uses a separate position readout while retaining the detection models unchanged.
Let $\widehat{\Delta\mathbf p}_t^{\mathrm{old}}$ be the increment returned by the original causal gate.
The locked forecasting candidate updates the historical position as
\begin{equation}
 \widehat{\mathbf p}_t=\begin{cases}
 \mathbf p_t^{\mathrm{obs}}, & \text{valid interval and }w_t\ge 0.7,\\
 \widehat{\mathbf p}_{t-1}+\widehat{\Delta\mathbf p}_t^{\mathrm{old}},&\text{otherwise}.
 \end{cases}\label{eq:forecast-reanchor}
\end{equation}
The reset uses the currently visible observation, including any imposed observation error.
Reported-velocity handling and the leave-one-out field are unchanged. High increment compatibility
does not certify an absolute position: a gradual or persistent offset can remain compatible.
This candidate was explicitly fixed after development analysis; no threshold search is performed
on the forecasting test data.
For forecasting, the nine target channels are relative east/north displacement, east/north reported
velocity, speed in m/s, sine/cosine of course, elapsed time and interval duration. The protected
speed cap is 14.404444 m/s, fitted only on the forecasting training histories. The eight field
channels contain background velocity, target-minus-background velocity, divergence, curl,
support and estimation mode. Unlike the individual classification affine-only interface, this
forecasting interface retains a zero-order distance or distance--reliability weighted velocity
estimate when fewer than four neighbours or degenerate geometry prevent an affine fit.
Support is $\min(N_+/4,1)$ for the positive-weight contributing reports in each estimator.
The two field variants share observed-position neighbourhood selection and differ in reliability weighting.
Given historical encoding $\mathbf z$ and a requested positive time offset $\tau$, the head predicts
\begin{equation}
 \widehat{\mathbf p}(t_{\rm last}+\tau)=\widehat{\mathbf p}_{\rm last}
 +\mathbf v^{\rm gate}_{\rm last}\tau+1000\,h_\theta(\mathbf z,\tau/600).
\end{equation}
The final correction layer is zero-initialized. Raw inputs use their own observed anchor and velocity.
All arms use the same registered architecture, frozen original training standardization and
SmoothL1 loss on coordinates divided by 1000, with transition parameter 0.1.
'''

MECHANISM=r'''
\subsubsection{Historical offset persistence and re-anchoring}
An audit of the 1,200 forecasting validation windows found 92 changed prediction origins:
80 were affected only by earlier historical increments, five only by the final increment,
and seven by both. Seven windows containing a cold-start fallback accounted for 90.6\%
of the pooled origin-only ADE increase. One Shanghai window contained a 13.43-km observed
displacement over 178 seconds in its first interval; its retained offset accounted for 61.4\%
of that increase. Base therefore denotes unmodified observations, not verified error-free tracks.
With fixed reported-velocity processing, re-anchoring reduced constant-velocity (CV) validation
ADE from 326.62 to 309.08 m (raw CV: 308.62 m), and from 459.74 to 434.46 m in the
758-window observed-manoeuvre stratum. Under endpoint jumps and speed spikes, gated CV ADE
changed from 438.89 to 422.46 m and from 1,859.23 to 1,840.29 m, respectively.
Under gradual drift, ADE changed from 729.37 to 736.25 m, with paired difference interval
$[-18.52,22.89]$ m. The candidate did not meet the original one-percent drift continuation
criterion; the subsequent experimental protocol explicitly accepted this development tradeoff
and locked the rule before retraining and forecasting test evaluation. A paired comparison of
drift-minus-Base degradation increased by 24.41 m, with interval $[21.95,26.95]$ m.
Thus removing persistent historical offsets and rejecting slow drift are distinct requirements.
'''

def table(summary,split):
    labels={'base':'Base observations','target_jump':'Target endpoint jump','target_drift':'Target gradual drift',
            'target_speed':'Target speed spike','neighbor_stress':'Neighbour speed and position corruption'}
    rows=summary['summary'];get=lambda m,a,v,sub='all':next(x for x in rows if (x['model'],x['arm'],x['view'],x['subset'])==(m,a,v,sub))
    lines=[r'\begingroup\small\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{lrrrr}',
        '\\caption{Forecasting '+split+r' errors in metres (ADE / FDE), averaged over three fixed seeds.}\label{tab:forecast-'+split+r'}\\',
        r'\toprule Backbone & Raw & Re-anchored gate & Ordinary field & Reliable field\\\midrule\endfirsthead',
        r'\toprule Backbone & Raw & Re-anchored gate & Ordinary field & Reliable field\\\midrule\endhead']
    for view,label in labels.items():
        lines.append(r'\multicolumn{5}{l}{\textit{'+label+r'}}\\')
        for m,name in [('tcn','TCN'),('gru','GRU'),('transformer','Transformer'),('itransformer','iTransformer')]:
            cells=[]
            for arm in ['P0','P1','P2','P3']:
                r=get(m,arm,view);cells.append(f"{r['ADE_m_mean']:.2f} / {r['FDE_m_mean']:.2f}")
            lines.append(name+' & '+' & '.join(cells)+r'\\')
        cv=get('CV_raw','CV_raw',view);cg=get('CV_gate','CV_gate',view)
        lines.append(r'\multicolumn{5}{l}{\footnotesize CV raw: '+f"{cv['ADE_m_mean']:.2f} / {cv['FDE_m_mean']:.2f}"+
            r'; CV re-anchored: '+f"{cg['ADE_m_mean']:.2f} / {cg['FDE_m_mean']:.2f}"+r'}\\\midrule')
    lines.extend([r'\bottomrule\end{longtable}',r'\noindent\footnotesize All arms share the frozen original training-only normalization. Raw weights are reused after exact input checks. Pointwise target-MMSI intervals, seed deviations and observed-manoeuvre strata are provided in the accompanying result files.',r'\endgroup'])
    return '\n'.join(lines)

def audit_pdf(paper,name):
    import fitz
    pdf=paper/(name+'.pdf');doc=fitz.open(pdf);text='\n'.join(p.get_text() for p in doc)
    if '??' in text:raise RuntimeError('Unresolved reference marker in compiled PDF')
    if '\ufffd' in text:raise RuntimeError('Replacement characters in compiled PDF')
    aux=(paper/(name+'.aux')).read_text(errors='replace') if (paper/(name+'.aux')).exists() else ''
    if r'\newlabel{fig:coverage}{{1}' not in aux:raise RuntimeError('Figure 1 label did not resolve to 1')
    qa=paper/'qa';qa.mkdir(exist_ok=True);outside=[];pages=[]
    for i,p in enumerate(doc):
        for word in p.get_text('words'):
            rect=fitz.Rect(word[:4])*p.rotation_matrix
            if rect.x0<-1 or rect.y0<-1 or rect.x1>p.rect.width+1 or rect.y1>p.rect.height+1:outside.append([i+1,word[:5]])
        if any(s in p.get_text() for s in ['Forecasting','re-anchoring','query-time','Figure 1']):pages.append(i)
    for i in sorted(set(pages)):
        doc[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(qa/f'page_{i+1:02d}.png')
    if outside:raise RuntimeError(f'Text outside PDF page: {outside[:3]}')
    save(qa/'pdf_audit.json',dict(pdf_sha256=sha(pdf),pages=len(doc),unresolved_reference_marker=False,
        coverage_label='1',outside_page_words=outside,rendered_review_pages=[i+1 for i in sorted(set(pages))],
        review_note='Automatic checks and rendered pages provided; inspect page PNGs for visual quality.'))

def compile_pdf(paper,name='main_v3'):
    compiler=os.environ.get('AIS_TECTONIC') or str(ROOT/'tools/tectonic')
    if not Path(compiler).is_file():
        compiler=shutil.which('tectonic')
    if not compiler:raise FileNotFoundError('Tectonic not found. Set AIS_TECTONIC to its executable.')
    for attempt in range(2):
        cmd=[compiler,'--keep-logs','--keep-intermediates',name+'.tex']
        with (paper/f'compile_{attempt+1}.log').open('w') as log:subprocess.run(cmd,cwd=paper,stdout=log,stderr=subprocess.STDOUT,check=True)
    log=(paper/(name+'.log')).read_text(errors='replace')
    if re.search(r'undefined references|Reference .* undefined|Citation .* undefined',log,re.I):raise RuntimeError('Unresolved LaTeX references; see log')
    audit_pdf(paper,name)

def update(work,preview=False,compile=True):
    work=Path(work);paper=work/('paper_preview' if preview else 'paper');paper.mkdir(parents=True,exist_ok=True)
    source=ROOT/'paper_template'
    for p in source.rglob('*'):
        if p.is_file():
            dest=paper/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    main=(paper/'main_v3.tex').read_text(encoding='utf-8')
    methods=(paper/'sections_v3/methods.tex');methods.write_text(methods.read_text(encoding='utf-8')+METHOD,encoding='utf-8')
    exp=paper/'sections_v3/experiments.tex';text=exp.read_text(encoding='utf-8')
    # Keep figure numbers automatic; make the coverage cross-reference explicitly linked and auditable.
    text=text.replace(r'Figure~\ref{fig:coverage}',r'\hyperref[fig:coverage]{Figure~\ref*{fig:coverage}}')
    assert text.count(r'\label{fig:coverage}')==1
    marker=r'\subsection{Individual benchmark across eight backbones}'
    assert marker in text;text=text.replace(marker,r'\input{sections_v3/forecasting_addendum}'+'\n\n'+marker)
    exp.write_text(text,encoding='utf-8')
    body=MECHANISM
    if preview:
        body+=r'\paragraph{Execution status.} This preview contains completed CV diagnostics only. The locked 36-run retraining and the post-freeze forecasting test have not been executed in this preview; their tables will be generated from saved predictions by the runnable pipeline.'
        main=main.replace(r'\maketitle',r'\maketitle\noindent\fbox{\parbox{.95\linewidth}{Forecasting integration preview: neural retraining and test tables pending execution.}}\par\medskip')
    else:
        assert_frozen(work);assert read(work/'test_complete.json')['status']=='complete'
        val=read(work/'reports/val/summary.json');test=read(work/'reports/test/summary.json')
        assert val['status']==test['status']=='complete';tm=read(work/'data/test_manifest.json')
        body+=r'''
\subsubsection{Query-time forecasting with observed future positions}
Forecasting uses 12 historical reports and eight subsequent unmodified AIS positions as observation references.
History-only inputs include actual time intervals; future report timestamps specify the requested prediction times.
Train/validation contain 5,600/1,200 windows. Test windows are constructed only after all final checkpoints,
normalization statistics and the evaluation protocol are frozen. Target identities are disjoint across these
forecasting sets; the parent AIS sources have appeared in earlier detection studies and are not a newly
collected external dataset. Every neighbourhood lookup precedes its historical query time.
Four forecasting adaptations use Raw, re-anchored gate, gate plus ordinary field, and gate plus reliable field.
Raw weights are reused after byte-exact input and normalization checks; the other 36 models are retrained.
Training uses Base observations, a shared SmoothL1 loss and the frozen original training normalization,
with 25 epochs for TCN/GRU/Transformer and 75 for iTransformer. The forecasting heads learn a correction
to constant-velocity extrapolation; the classification training bundle is not used.
ADE averages the eight Euclidean position errors; FDE evaluates the final requested time.
Prediction origins are obtained from each model's visible historical observations, never from clean future anchors.
'''
        body+=f"\nThe forecasting test contains {tm['windows']:,} windows from {tm['target_mmsi']} targets; its median final prediction horizon is {tm['horizon_seconds_quantiles'][1]:.0f} seconds.\n"
        body+=r'\input{tables_v3/forecast_val}'+'\n'+r'\input{tables_v3/forecast_test}'+'\n'
        rows=test['summary']
        def avg(arm,view):return sum(r['ADE_m_mean'] for r in rows if r['arm']==arm and r['view']==view and r['subset']=='all')/4
        body+='The equal-backbone mean test ADE for Raw versus the complete reliable-field configuration is '
        body+=f"{avg('P0','base'):.2f} versus {avg('P3','base'):.2f} m on Base, "
        body+=f"{avg('P0','target_jump'):.2f} versus {avg('P3','target_jump'):.2f} m under endpoint jumps, and "
        body+=f"{avg('P0','target_speed'):.2f} versus {avg('P3','target_speed'):.2f} m under speed spikes. "
        body+='These complete-configuration contrasts combine gating and context; the reliable-minus-ordinary contrasts isolate the context weighting change. All views, including gradual drift, are retained in the tables.\n'
        for split,summary in [('val',val),('test',test)]:(paper/f'tables_v3/forecast_{split}.tex').write_text(table(summary,split),encoding='utf-8')
        main=main.replace('Two event-associated trajectories illustrate',
            'A separate, frozen four-backbone forecasting evaluation uses subsequent AIS observations to assess position error under historical-input corruption. Two event-associated trajectories illustrate')
        save(paper/'forecast_result_sources.json',dict(validation_sha256=sha(work/'reports/val/summary.json'),test_sha256=sha(work/'reports/test/summary.json'),freeze_sha256=sha(work/'checkpoint_freeze.json')))
    (paper/'sections_v3/forecasting_addendum.tex').write_text(body,encoding='utf-8')
    (paper/'main_v3.tex').write_text(main,encoding='utf-8')
    if compile:compile_pdf(paper)
    return paper

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'));p.add_argument('--preview',action='store_true');p.add_argument('--no-compile',action='store_true')
    a=p.parse_args();print(update(a.work,a.preview,not a.no_compile))
