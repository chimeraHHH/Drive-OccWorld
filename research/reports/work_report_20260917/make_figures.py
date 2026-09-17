"""Render report figures from the bundled, source-identified real aggregates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / 'data.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': '#293b4a', 'text.color': '#293b4a',
    'axes.titleweight': 'bold', 'pdf.fonttype': 42,
    'savefig.facecolor': 'white'})
C = {'O':'#164d6b','M0_fp32':'#8398a8','native1':'#b09a75',
     'T':'#cf763c','J':'#9775af','D':'#459b8f'}

def save(fig, name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf', bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png', bbox_inches='tight', dpi=170)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 2.7), layout='constrained')
for model, label in [('M0_fp32','M0 (matched precision)'),('native1','C (continued control)'),('O','O (retained reference)')]:
    rows = [r for r in d['full5119'][model] if r['horizon_s'] > 0]
    ax.plot([r['horizon_s'] for r in rows], [r['GMO_percent'] for r in rows],
        '-o', color=C[model], lw=2.1, ms=5, label=label)
ax.set(xlabel='Future horizon (s)', ylabel='GMO IoU (%)', xticks=[.5,1,1.5,2], ylim=(12,16.2))
ax.grid(axis='y',alpha=.16); ax.legend(frameon=False,fontsize=8,loc='lower left')
save(fig,'full5119_horizons')

fig, axes = plt.subplots(1,2,figsize=(7.2,3.0),layout='constrained')
for arm in ['O','T','J','D']:
    if arm not in d['dev200']: continue
    v=d['dev200'][arm]
    axes[0].plot([.5,1,1.5,2],v['horizon_GMO_percent'][1:],'-o',color=C[arm],lw=1.8,ms=4,label=arm)
axes[0].plot([.5,1,1.5,2],d['dev200']['T']['own_current_persistence']['horizon_GMO_percent'],
    '--',color='#666666',lw=1.4,label='T/J/D current persistence')
axes[0].set(xlabel='Future horizon (s)',ylabel='GMO IoU (%)',xticks=[.5,1,1.5,2],title='(a) Occupancy: dev200')
axes[0].legend(frameon=False,fontsize=7,loc='lower left')
arms=['T','J'] + (['D'] if 'D' in d['dev_physical'] else [])
groups=['moving','stationary']; x=np.arange(2); width=.17
cv=[next(r['CRN_CV_EPE_XYZ_m'] for r in d['dev_physical']['T'] if r['endpoint_index']==4 and r['group']==g) for g in groups]
for i,arm in enumerate(['CRN-CV']+arms):
    vals=cv if arm=='CRN-CV' else [next(r['candidate_EPE_XYZ_m'] for r in d['dev_physical'][arm] if r['endpoint_index']==4 and r['group']==g) for g in groups]
    bars=axes[1].bar(x+(i-(len(arms)/2))*width,vals,width,color='#8398a8' if arm=='CRN-CV' else C[arm],label=arm)
    axes[1].bar_label(bars,fmt='%.2f',fontsize=7,padding=2)
axes[1].set(xticks=x,xticklabels=['Moving\n(n=1,226)','Stationary\n(n=2,136)'],ylabel='XYZ endpoint error (m)',title='(b) Physical proxy: +2 s',ylim=(0,6.3))
axes[1].legend(frameon=False,fontsize=7,ncol=2)
for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'dev200_comparison')

fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
f=d['T_train512_fit']
for ax,v,population in [(axes[0],f,'Train512 / 256 scenes'),(axes[1],d['dev200']['T'],'Dev200 / 100 scenes')]:
    y=v['future_GMO_percent'] if ax is axes[0] else v['horizon_GMO_percent'][1:]
    p=v['persistence']['future_GMO_percent'] if ax is axes[0] else v['own_current_persistence']['horizon_GMO_percent']
    ax.plot([.5,1,1.5,2],y,'-o',color=C['T'],ms=4,label='T forecast')
    ax.plot([.5,1,1.5,2],p,'--s',color='#738c9d',ms=4,label='Own current persistence')
    ax.set(xlabel='Future horizon (s)',ylabel='GMO IoU (%)',title=population,xticks=[.5,1,1.5,2],ylim=(6.7,14.6))
    ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=7,loc='lower left')
save(fig,'T_fit_diagnosis')
print('Rendered three figures from data.json; no synthetic values.')
