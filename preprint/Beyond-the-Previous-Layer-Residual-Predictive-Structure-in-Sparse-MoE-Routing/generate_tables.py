"""Render frozen results as LaTeX; no model loading, fitting, or experiments."""
from pathlib import Path
import hashlib, json, re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = json.loads((HERE / 'source_manifest.json').read_text())
for item in MANIFEST['sources']:
    path = ROOT / item['path']
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item['sha256']:
        raise SystemExit(f"Frozen source changed: {item['path']}")

def get(path):
    return json.loads((ROOT / path).read_text())

e = get('code/experiments/03_eipc/artifacts/EIPC_P0/results.json')
p = get('code/experiments/05_epd/artifacts/EPD_P0/results.json')
r = get('code/experiments/06_rmo/artifacts/results.json')
c = get('code/experiments/07_rmc/artifacts/results.json')
n = get('code/experiments/08_nhd/artifacts/results.json')
epd_report = (ROOT / 'code/experiments/05_epd/artifacts/EPD_P0/RESULTS.md').read_text()
match = re.search(r'ID_PATH\+CONTENT \+(\d+\.\d+)', epd_report)
if not match:
    raise SystemExit('Missing EPD post-hoc path-plus-content result in frozen report.')
path_plus_content = float(match.group(1))
def fmt(v):
    return f'{v:.5f}'
def row(*v):
    return ' & '.join(str(x) for x in v) + r' \\' + '\n'

values = {'EpdPathContent':path_plus_content, 'EipcEightA':e['targets']['8']['A'], 'EipcTwelveA':e['targets']['12']['A'],
          'EipcMeanA':e['mean_A'], 'EpdId':p['main']['R2_id'], 'EpdContent':p['main']['R2_content'],
          'EpdFirst':p['layerwise']['R2_id_layer']['1'], 'EpdLast':p['layerwise']['R2_id_layer']['11'],
          'RmoOne':r['r2_test']['R2_1'], 'RmoEleven':r['r2_test']['R2_11'],
          'RmoStepEleven':r['stepwise_gains']['S_11'],
          'RmcTwelveDelta':c['per_target']['12']['Delta4'], 'RmcTwentyDelta':c['per_target']['20']['Delta4']}
for target, name in [('12','Twelve'), ('20','Twenty')]:
    a=n['per_target'][target]['part_a']; b=n['per_target'][target]['part_b']
    for key, raw in [('A',a['A']),('B',a['B']),('Matched',a['B_matched']),('Residual',b['residual_R2']),('Permutation',b['permuted_residual_R2'])]:
        values['Nhd'+name+key]=raw
(HERE/'numbers.tex').write_text('% Generated from the pinned result files.\n'+''.join('\\expandafter\\def\\csname val'+k+'\\endcsname{'+fmt(v)+'}\n' for k,v in values.items()))

t=r'''\begin{table*}[t]
\centering\small
\caption{\textbf{Where the predictive signal comes from.} Held-out router-logit $R^2$ in OLMoE. EIPC retains selected contributions in identity slots; EPD separates selection identity from contribution content. EIPC and EPD use different samples and compression budgets, so comparisons are within each panel. The EPD single-layer curve uses 16 PCA components per representation.}
\label{tab:provenance}
\begin{minipage}[t]{0.49\textwidth}
\vspace{0pt}
\centering
\begin{tabular}{lrrr}
\toprule
\multicolumn{4}{l}{\textbf{EIPC: provenance (64 components)}}\\
Target & Fused & Identity & Shuffled\\
\midrule
'''
for k in ['8','12']:
    d=e['targets'][k];t+=row('L'+k,fmt(d['R2_fused']),fmt(d['R2_identity']),fmt(d['R2_shuffled']))
t+=r'''\midrule
Target & Id.$-$Fused & Id.$-$Shuf. & \\
'''
for k in ['8','12']:
    d=e['targets'][k];t+=row('L'+k,fmt(d['A']),fmt(d['B']),'')
t+=row('Mean',fmt(e['mean_A']),fmt(e['mean_B']),'')
t+=r'''\bottomrule
\end{tabular}
\par\medskip
\begin{tabular}{lr}
\toprule
\multicolumn{2}{l}{\textbf{EPD: L12 (32 components)}}\\
Representation & $R^2$\\
\midrule
'''
for label,key in [('Fused','R2_fused'),('Selection path','R2_id'),('Content by router rank','R2_content'),('Full provenance','R2_full')]:t+=row(label,fmt(p['main'][key]))
t+=row('Path + content (post-hoc)',fmt(path_plus_content))
t+=r'''\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.47\textwidth}
\vspace{0pt}
\centering
\begin{tabular}{rrr}
\toprule
\multicolumn{3}{l}{\textbf{EPD: single-layer prediction of L12}}\\
Source layer & Selection path & Content\\
\midrule
'''
for k in map(str,range(1,12)):
    t+=row(k,fmt(p['layerwise']['R2_id_layer'][k]),fmt(p['layerwise']['R2_content_layer'][k]))
t+=r'\midrule'+'\n'+row('Mean',fmt(p['layerwise']['summary']['mean_id']),fmt(p['layerwise']['summary']['mean_content']))
t+=r'''\bottomrule
\end{tabular}
\end{minipage}
\end{table*}
'''
(HERE/'table_provenance.tex').write_text(t)

t=r'''\begin{table*}[t]
\centering\small
\caption{\textbf{History improves prediction beyond the previous layer.} Top: held-out $R^2$ for nested history windows; $k$ includes the immediately preceding layer. Middle: OLMoE gains over $k=1$ and between consecutive windows. Bottom: JetMoE gains, with 95\% paired bootstrap intervals for the preregistered $k=4$ comparison (10,000 resamples). Dashes denote unavailable windows.}
\label{tab:history}
\begin{tabular}{lrrrrr}
\toprule
Model / target & $k=1$ & $k=2$ & $k=4$ & $k=8$ & $k=11$\\
\midrule
'''
t+=row('OLMoE / L12',*[fmt(r['r2_test'][f'R2_{k}']) for k in [1,2,4,8,11]])
for k in ['12','20']:t+=row('JetMoE / L'+k,*[fmt(c['per_target'][k]['R2'][f'k{h}']) for h in [1,2,4,8]],'--')
t+=r'\midrule'+'\n'
t+=row('OLMoE cumulative gain','--',*[fmt(r['cumulative_gains'][f'G_{k}']) for k in [2,4,8,11]])
t+=row('OLMoE stepwise gain','--',*[fmt(r['stepwise_gains'][f'S_{k}']) for k in [2,4,8,11]])
t+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{lrrrr}
\toprule
JetMoE target & $\Delta_2$ & $\Delta_4$ & $\Delta_8$ & 95\% CI for $\Delta_4$\\
\midrule
'''
for k in ['12','20']:
    d=c['per_target'][k];ci=d['bootstrap_Delta4']
    t+=row('L'+k,fmt(d['Delta2']),fmt(d['Delta4']),fmt(d['Delta8']),'['+fmt(ci['ci_low'])+', '+fmt(ci['ci_high'])+']')
t+=r'''\bottomrule
\end{tabular}
\end{table*}
'''
(HERE/'table_history.tex').write_text(t)

t=r'''\begin{table*}[t]
\centering\small
\caption{\textbf{Nonlinear decoding preserves the historical advantage (NHD).} Raw-state ridge scores and MLP scores for all three initialization seeds. $A$ is the recent-only MLP gain over recent-only ridge; $B$ is the history MLP gain over the recent-only MLP; $B_{\mathrm{matched}}$ uses the parameter-matched recent-only MLP. Residual scores predict $\epsilon_Y$ from $\epsilon_H$ and use a different target from the joint prediction scores. The permutation control shuffles TEST history residuals with the fitted residual probe held fixed.}
\label{tab:nonlinear}
\begin{tabular}{llrrrrr}
\toprule
Target & Decoder / input & Parameters & Seed 42 & Seed 123 & Seed 2026 & Mean / fixed\\
\midrule
'''
for k in ['12','20']:
    a=n['per_target'][k]['part_a']
    for label,key in [('Ridge / $R$','LINEAR_RECENT'),('Ridge / $[H;R]$','LINEAR_HISTORY')]:
        t+=row('L'+k,label,'--','--','--','--',fmt(a['linear'][key]))
    for label,key,params in [('MLP / $R$','MLP_RECENT',552),('Matched MLP / $R$','MLP_RECENT_MATCHED',1317),('MLP / $[H;R]$','MLP_HISTORY',1320)]:
        d=a['nonlinear'][key];t+=row('L'+k,label,f'{params:,}',*[fmt(d[s]) for s in ['42','123','2026','mean']])
    if k=='12':t+=r'\midrule'+'\n'
t+=r'''\bottomrule
\end{tabular}
\par\smallskip
\begin{tabular}{lrrrrrr}
\toprule
Target & Linear gain & $A$ & $B$ & $B_{\mathrm{matched}}$ & Residual $R^2$ & Permuted $R^2$\\
\midrule
'''
for k in ['12','20']:
    a=n['per_target'][k]['part_a'];b=n['per_target'][k]['part_b']
    t+=row('L'+k,fmt(a['linear']['linear_gain']),fmt(a['A']),fmt(a['B']),fmt(a['B_matched']),fmt(b['residual_R2']),fmt(b['permuted_residual_R2']))
t+=r'''\bottomrule
\end{tabular}
\end{table*}
'''
(HERE/'table_nonlinear.tex').write_text(t)
print('Rendered 3 result tables and numerical macros from hash-verified frozen sources.')
