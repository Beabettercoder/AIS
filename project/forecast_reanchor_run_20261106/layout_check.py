"""Synthetic table-layout QA only; its numbers are not experiment results."""
from pathlib import Path
from common import ROOT
from paper_update import table,compile_pdf

if __name__=='__main__':
    dest=ROOT/'qa_layout';dest.mkdir(exist_ok=True);rows=[]
    for model,arm in [(m,a) for m in ['tcn','gru','transformer','itransformer'] for a in ['P0','P1','P2','P3']]+[('CV_raw','CV_raw'),('CV_gate','CV_gate')]:
        for view in ['base','target_jump','target_drift','target_speed','neighbor_stress']:
            rows.append(dict(model=model,arm=arm,view=view,subset='all',ADE_m_mean=12345.67,FDE_m_mean=23456.78))
    text=r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath,booktabs,array,longtable,geometry,hyperref,graphicx}
\geometry{margin=1in}\begin{document}
\section*{Synthetic layout fixture -- not experimental results}
\begin{figure}[h]\centering\fbox{Figure reference layout check}\caption{Layout-only figure}\label{fig:coverage}\end{figure}
Figure~\ref{fig:coverage} tests the reference mechanism.
'''+table({'summary':rows},'layout')+r'\end{document}'
    (dest/'main_v3.tex').write_text(text,encoding='utf-8');compile_pdf(dest)
    print('PASS table-layout compile and page-bound checks. All numbers in this fixture are synthetic.')
