"""Render publication figures from frozen configuration, saved scores and seeded traces."""

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from describe_fixed_policy import ROOT, EVIDENCE, SCENARIOS, read_rows, summarize
sys.path.insert(0, str(ROOT / 'benchmark'))
from data import make_batch, plan_devices, corrupt

OUT = ROOT / 'manuscript' / 'figures'
CONFIG = json.loads((ROOT / 'benchmark' / 'protocol.json').read_text(encoding='utf-8'))
RESULT = json.loads((EVIDENCE / 'results.json').read_text(encoding='utf-8'))
INK, MUTED, LIGHT = '#182932', '#526872', '#DDE6E8'
TEAL, BLUE, RED, GRAY = '#106F68', '#355E9B', '#A03F35', '#626E74'
LABELS = ('Clean', 'Brief dropout', 'Sustained dropout')
SCENARIO_COLORS = (TEAL, BLUE, RED)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10.5, 'axes.titleweight':'bold',
    'axes.labelcolor':INK, 'text.color':INK, 'axes.edgecolor':LIGHT,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.axisbelow':True,
    'xtick.color':MUTED, 'ytick.color':MUTED, 'grid.color':LIGHT, 'grid.linewidth':.7})


def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=220, bbox_inches='tight', facecolor='white', pad_inches=.12)
    plt.close(fig)


def trace():
    cases = [x for x in RESULT['failure_analysis']['examples'] if x['label'] == 1]
    assert len(cases) == 2
    assert cases[0]['device'] == cases[1]['device']
    seed = int(cases[0]['device'].split('-')[0][1:])
    batch = make_batch(CONFIG, seed, list(plan_devices(CONFIG, seed)), 'test')
    rows = {(r['window_id'], r['scenario']): r for r in read_rows() if r['model'] == 'cnn'}
    _, mask = corrupt(batch, CONFIG, 'sustained')
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 4.4), sharex=True, sharey=True, layout='constrained')
    for row, case in enumerate(cases):
        j = batch.ids.index(case['window_id'])
        for k, scenario in enumerate(('clean','sustained')):
            ax = axes[row,k]
            sample = rows[case['window_id'],scenario]
            assert int(batch.labels[j]) == int(sample['label']) == 1
            observed = np.ones_like(mask[j]) if k == 0 else mask[j]
            for ch,color in enumerate((TEAL, BLUE)):
                ax.plot(np.arange(64), np.where(observed[ch],batch.signal[j,ch],np.nan),color=color,lw=1.25,label=f'Channel {ch+1}')
            if k:
                missing = np.flatnonzero(~observed[int(sample['channel_missing'])])
                ax.axvspan(missing[0]-.5, missing[-1]+.5, color=RED, alpha=.09)
                ax.text(.98,.05, f'Channel {int(sample["channel_missing"])+1} missing', ha='right', transform=ax.transAxes,fontsize=8.5,color=RED)
            status = 'Deferred' if sample['retained'] == 'False' else ('Issued event' if float(sample['probability'])>=.5 else 'Issued no event')
            ax.set_title(f'{case["window_id"][-4:]} | {scenario.title()}   P(event) {float(sample["probability"]):.3f}\n{status}',loc='left',fontsize=10)
            ax.grid(axis='y')
    axes[0,0].legend(loc='upper right', frameon=False, fontsize=8.5, ncol=2)
    fig.supxlabel('Sample index within a 64-sample window'); fig.supylabel('Synthetic signal (arbitrary units)')
    save(fig,'paired_traces')


def comparison():
    fig, axes = plt.subplots(1,3, figsize=(10.3,3.7),width_ratios=(1,1,1.1),layout='constrained')
    x=np.arange(3)
    for ax,key,title,ylim in zip(axes[:2],('average_precision','brier'),('A  Average precision','B  Brier score'),((0,1),(0,.26))):
        for model,color,style,label in [('cnn',TEAL,'o','1D CNN'),('logistic',BLUE,'s','Logistic'),('prevalence',GRAY,'^','Training prior')]:
            ax.plot(x,[RESULT['metrics'][model][s][key] for s in SCENARIOS],marker=style,color=color,lw=1.9,label=label)
        ax.set(xticks=x,xticklabels=('Clean','Brief','Sustained'),ylim=ylim)
        ax.set_title(title,loc='left',fontsize=11); ax.grid(axis='y')
    axes[0].legend(loc='upper right',frameon=False,fontsize=8.5)
    ax=axes[2]
    for i,(scenario,color) in enumerate(zip(SCENARIOS,SCENARIO_COLORS)):
        item=RESULT['bootstrap']['comparisons'][f'cnn_minus_logistic_{scenario}']['average_precision']
        p=item['estimate']; low,high=item['percentile_95']
        ax.errorbar(p,2-i,xerr=[[p-low],[high-p]],fmt='o',color=color,capsize=4,linewidth=1.8)
    ax.axvline(0,c=GRAY,ls='--',lw=1)
    ax.set(yticks=[2,1,0],yticklabels=['Clean','Brief','Sustained'],ylim=(-.5,2.5),xlim=(-.025,.31),xlabel='Difference in average precision')
    ax.set_title('C  CNN minus logistic',loc='left',fontsize=11); ax.grid(axis='x')
    save(fig,'model_comparison')


def risk_and_support():
    with (EVIDENCE/'risk_coverage.csv').open(encoding='utf-8',newline='') as f:
        curve=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.15),layout='constrained')
    for scenario,color,label in zip(SCENARIOS,SCENARIO_COLORS,LABELS):
        rows=[r for r in curve if r['scenario']==scenario and r['risk']]
        axes[0].plot([float(r['coverage']) for r in rows],[float(r['risk']) for r in rows],marker='.',color=color,label=label)
        m=RESULT['metrics']['cnn'][scenario]
        axes[0].scatter(m['coverage'],m['risk'],s=75,c=color,edgecolors='black',linewidths=.75,zorder=4)
    axes[0].set(xlabel='Windows receiving an automated label',ylabel='Error among issued labels',xlim=(0,1.025),ylim=(0,.32))
    axes[0].xaxis.set_major_formatter(PercentFormatter(1)); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc='upper left',frameon=False,fontsize=8.5)
    axes[0].set_title('A  Risk and coverage',loc='left'); axes[0].grid()
    axes[0].text(.035,.07,'Outlined dots: frozen policy',transform=axes[0].transAxes,fontsize=8.5)
    ax=axes[1]
    bins=RESULT['metrics']['cnn']['sustained']['calibration_bins']
    ax.plot([0,1],[0,1],'--',color=GRAY,lw=1,label='Perfect agreement')
    ax.plot([b['mean_probability'] for b in bins],[b['event_rate'] for b in bins],color=RED,lw=1.2)
    for b in bins:
        ax.scatter(b['mean_probability'],b['event_rate'],s=11+.25*b['n'],color=RED,alpha=.8,edgecolors='white',zorder=3)
    ax.set(xlabel='Mean calibrated event probability',ylabel='Observed event fraction',xlim=(0,1),ylim=(0,1))
    ax.set_title('B  Sustained dropout calibration',loc='left'); ax.grid(); ax.legend(frameon=False,fontsize=8.5,loc='upper left')
    ax.text(.98,.05,'Marker area reflects bin support',ha='right',transform=ax.transAxes,fontsize=8.5)
    save(fig,'risk_and_calibration')


def policy_figure():
    m=summarize(read_rows())['sustained']
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.9),layout='constrained')
    ax=axes[0]; rows=((1369,431),(1164,274,205,157))
    assert rows[0]==(m['windows']-m['all_issued_wrong'],m['all_issued_wrong'])
    assert rows[1]==tuple(m[k] for k in ('issued_correct','issued_wrong','deferred_correct_if_issued','deferred_wrong_if_issued'))
    colors=(TEAL,RED,'#BCE1D9','#F1C4B9'); categories=('Correct issued','Wrong issued','Deferred: would be right','Deferred: would be wrong')
    for y,counts in zip((1,0),rows):
        left=0
        for j,n in enumerate(counts):
            ax.barh(y,n,left=left,color=colors[j],height=.47,label=categories[j] if y==0 else None)
            ax.text(left+n/2,y,f'{n:,}',color='white' if j<2 else INK,ha='center',va='center',fontsize=9,weight='bold')
            left+=n
    ax.set(yticks=[1,0],yticklabels=['Issue all','Frozen rule'],xlim=(0,1800),xticks=[0,600,1200,1800],xlabel='Windows',ylim=(-.9,1.45))
    ax.set_title('A  Label outcomes',loc='left'); ax.grid(axis='x')
    ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,.035),fontsize=7.6)
    ax=axes[1]; x=np.linspace(0, .75, 101); n=m['windows']; fixed=(m['issued_wrong']+x*m['deferred'])/n
    baseline=m['error_if_all_issued']; cross=m['break_even_cost_per_deferral']
    ax.plot(x,fixed,c=TEAL,lw=2,label='Frozen rule'); ax.axhline(baseline,c=RED,lw=1.8,label='Issue all')
    ax.scatter([.18,cross],[m['cost_at_original_weight'],baseline],c=INK,zorder=4,s=27)
    ax.annotate('Original 0.18',(.18,m['cost_at_original_weight']),xytext=(.06,.225),arrowprops={'arrowstyle':'-','color':MUTED},fontsize=8.5)
    ax.annotate('Break-even 0.434',(cross,baseline),xytext=(.45,.198),arrowprops={'arrowstyle':'-','color':MUTED},fontsize=8.5)
    ax.set(xlabel='Assumed cost per deferral',ylabel='Assumed cost per window',xlim=(0,.75),ylim=(.13,.33))
    ax.set_title('B  Cost sensitivity',loc='left'); ax.grid(); ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    save(fig,'decision_tradeoff')


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    trace(); comparison(); risk_and_support(); policy_figure()
