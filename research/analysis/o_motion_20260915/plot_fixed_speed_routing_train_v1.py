"""Render all fixed speed policies; no selection or refitting."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N = Path(__file__).resolve().parent
SOURCE = N/'fixed_speed_routing_train_analysis_v1.json'
SOURCE_SHA = 'c8ad1b370758f970ec3ae73b4830126d8a31fc35cfe6ce5321f860033ba6dcf3'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
data = json.loads(SOURCE.read_text())
thresholds = data['rules']['thresholds_mps']
labels = ['0', '0.1', '0.5', '1', '2', '5', '10', '∞']


def rows(group):
    return [next(r for r in data['policies'] if r['horizon_seconds'] == 2. and
                 r['group'] == group and r['threshold_mps'] == t) for t in thresholds]


stationary, moving, all_rows = rows('stationary'), rows('moving'), rows('all')
static_cost = np.array([r['metrics']['xy']['minus_CV_m'] for r in stationary])
moving_gain = -np.array([r['metrics']['xy']['minus_CV_m'] for r in moving])
used = 100*np.array([r['selected_fraction_of_uncovered_weight'] for r in all_rows])
gain_retained = 100*moving_gain/moving_gain[0]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11, 'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':'#263849','text.color':'#263849',
    'xtick.color':'#526574','ytick.color':'#526574','pdf.fonttype':42})
fig, (a,b) = plt.subplots(1,2,figsize=(12.4,5.2))
fig.subplots_adjust(left=.075,right=.97,bottom=.21,top=.73,wspace=.28)
fig.suptitle('Speed magnitude is already a strong routing comparator',x=.075,y=.965,ha='left',fontsize=17,weight='bold')
fig.text(.075,.888,'Actual train512 / 256 D-seen scenes · 2 s XY EPE · all eight fixed policies',fontsize=11,color='#596B79')
a.plot(100*static_cost,moving_gain,color='#176C86',lw=1.8,zorder=2)
a.scatter(100*static_cost,moving_gain,s=48,color='#176C86',edgecolor='white',lw=.8,zorder=3)
offsets=[(-19,-20),(4,9),(4,9),(5,9),(7,-10),(9,0),(9,0),(9,5)]
for i,(label,offset) in enumerate(zip(labels,offsets)):
    a.annotate('τ='+label,(100*static_cost[i],moving_gain[i]),xytext=offset,textcoords='offset points',fontsize=9)
a.axvline(0,color='#8B989F',lw=.8,ls='--');a.axhline(0,color='#8B989F',lw=.8,ls='--')
a.set(xlabel='Stationary EPE increase over CV (cm)',ylabel='Moving EPE reduction from CV (m)',
      xlim=(-.15,4.05),ylim=(-.025,.47))
a.set_title('A   Motion benefit and stationary cost',loc='left',fontsize=12,weight='bold',pad=18)
a.grid(alpha=.16)
x=np.arange(8)
b.plot(x,gain_retained,'o-',color='#176C86',lw=1.8,label='Moving benefit retained')
b.plot(x,used,'s--',color='#C07742',lw=1.6,label='Uncovered weight using D')
b.set_xticks(x,labels);b.set(xlabel='Predicted LSQ speed threshold τ (m/s)',ylabel='Percent of original reference',ylim=(-4,110))
b.set_title('B   Benefit retained as D use decreases',loc='left',fontsize=12,weight='bold',pad=18)
b.legend(frameon=False,fontsize=9.5,loc='upper right');b.grid(axis='y',alpha=.16)
fig.text(.075,.085,'Covered points always retain CRN-CV; uncovered points use D only when predicted speed ≥ τ.',fontsize=9.5)
fig.text(.075,.043,'GT-defined diagnostic queries · no threshold selected · no held-out or occupancy evaluation · no confidence intervals',fontsize=9,color='#687B88')
out=N/'figures';out.mkdir(exist_ok=True)
for suffix in ('png','pdf'):
    fig.savefig(out/('fixed_speed_routing_train_v1.'+suffix),dpi=210,facecolor='white')
plt.close(fig)
(out/'fixed_speed_routing_train_v1.json').write_text(json.dumps(dict(source_sha256=SOURCE_SHA,
    plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),thresholds_mps=thresholds,
    stationary_cost_m=static_cost.tolist(),moving_gain_m=moving_gain.tolist(),D_use_percent=used.tolist(),
    moving_gain_retained_percent=gain_retained.tolist(),synthetic_or_mock=False),indent=2)+'\n')
print(out/'fixed_speed_routing_train_v1.png')
