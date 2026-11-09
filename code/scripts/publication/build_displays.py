"""Generate CAS-sized manuscript figures and tables from archived experiment outputs."""
from pathlib import Path
import argparse
import sys,json,hashlib
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-root', type=Path, required=True,
                    help='Directory containing archived experiment evidence (not bundled).')
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/publication',
                    help='Directory for generated figures, tables, and provenance.')
args = parser.parse_args()
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

EVIDENCE = args.evidence_root.resolve()
PAPER = args.output.resolve()
PAPER.mkdir(parents=True, exist_ok=True)
for folder in ('tables', 'figs'):
    (PAPER / folder).mkdir(exist_ok=True)
sys.path.insert(0,str(HERE))
from orx_figstyle import use_style,PALETTE
use_style()
plt.rcParams.update({'font.family':'DejaVu Sans','font.sans-serif':['DejaVu Sans'],'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8.5})
# Supplied CAS class: paper width192mm, horizontal margins13.7mm each.
WIDTH=(192-2*13.7)/25.4
M=EVIDENCE/'multisegment'
D=EVIDENCE/'highspeed'
q=pd.read_csv(M/'summary.csv')
p=q.query("role=='test' and regime=='gap4'")
r={m:p.query('model==@m').set_index('k') for m in p.model.unique()}
models=[('cnn','CNN',PALETTE['blue'],'o'),('transformer','Transformer',PALETTE['orange'],'s'),('rf','Random forest',PALETTE['green'],'^')]
assert json.loads((M/'checkpoint_audit.json').read_text())['status']=='PASS'
outputs=[]
def save(fig,name):
 for ext in ['pdf','svg','png']:
  out=PAPER/'figs'/f'{name}.{ext}';fig.savefig(out,dpi=180,facecolor='white');outputs.append(out)
 plt.close(fig)
def table(name,caption,label,cols,headers,body,note=''):
 lines=['% Generated from archived results by revision_20261018_camera/build_displays.py.',r'\begin{table}[pos=tbp]',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\centering',r'\begin{tabular}{'+cols+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
 lines += [' & '.join(row)+r' \\' if isinstance(row,list) else row for row in body]
 lines += [r'\bottomrule',r'\end{tabular}']
 if note: lines += [r'\par\smallskip\begin{minipage}{\linewidth}\footnotesize '+note+r'\end{minipage}']
 lines += [r'\end{table}']
 out=PAPER/'tables'/f'{name}.tex';out.write_text('\n'.join(lines)+'\n');outputs.append(out)
pct=lambda x:f'{100*x:.2f}'
pm=lambda x,sd:f'${100*x:.2f}\\pm{100*sd:.2f}$'
counts=pd.read_csv(D/'counts.csv').groupby(['campaign','label']).windows.sum().unstack(fill_value=0)
roles={'Vicolungo':'Per-frame control','ASta':'Per-frame control','Casale':'Per-frame control','CATS':'Experimental assignment','Vanderbilt':'Experimental assignment'}
body=[]
for c in roles:
 row=counts.loc[c];body.append([c,f'{int(row.get(0,0)):,}',f'{int(row.get(1,0)):,}',f'{int(row.sum()):,}',roles[c]])
table('camera_data','Prepared longitudinal trajectory data after the high-speed filter.','tab:data','lrrrl',['Source','HV windows','ACC windows','Total','Label basis'],body,'Counts refer to overlapping four-second windows. Training uses a four-second grid and source-specific exclusions. These existing trajectory records are separate from the planned roadside collection.')
body=[]
for m,name in [('cnn','CNN'),('transformer','Transformer'),('rf','Random forest'),('cnn_head','CNN, head-only pooling'),('cnn_joint','CNN, joint fine-tuning')]:
 if m=='cnn_head':body.append(r'\midrule')
 body.append([name]+[pm(r[m].loc[k,'accuracy_mean'],r[m].loc[k,'accuracy_std']) for k in [1,2,4,8]])
table('camera_matched','Identification accuracy (\\%) for single and repeated observations.','tab:matched','lrrrr',['Method','$K=1$','$K=2$','$K=4$','$K=8$'],body,'Values are means $\\pm$ sample SD over five seeds, using the same107groups and equal record-local-pair weights. The first three methods average segment probabilities; the last two pool CNN embeddings. Observed durations are4,8,16,and32s, respectively.'.replace('same107groups','same 107 groups').replace('are4,8,16,and32s','are 4, 8, 16, and 32 s'))
body=[]
for m,name,_,_ in models:
 row=r[m].loc[8];body.append([name,pct(row.accuracy_mean),pct(row.bag_accuracy_mean),pct(row.human_recall_mean),pct(row.acc_recall_mean),f'{row.bce_mean:.4f}'])
table('camera_diagnostics','Eight-segment classification performance and class recalls.','tab:diagnostics','lrrrrr',['Method','Pair accuracy','Group accuracy','HV recall','ACC recall','BCE'],body,'Accuracy and recall are percentages. Pair accuracy gives local pairs equal weight within each source; group accuracy gives groups equal weight. Recall and BCE use pair weighting. Sources are weighted equally in all metrics.')
body=[]
for regime,label in [('dense','Contiguous segments; 107 groups'),('gap4','4 s gap; 107 groups'),('gap12','12 s gap; 44 groups')]:
 body.append(r'\multicolumn{4}{l}{\textit{'+label+r'}} \\')
 for m,name,_,_ in models:
  z=q.query("role=='test' and regime==@regime and model==@m").set_index('k')
  body.append([name,pct(z.loc[1,'accuracy_mean']),pct(z.loc[8,'accuracy_mean']),f"{100*(z.loc[8,'accuracy_mean']-z.loc[1,'accuracy_mean']):+.2f}"])
table('camera_spacing','Identification sensitivity to temporal spacing.','tab:spacing','lrrr',['Method','$K=1$ (\\%)','$K=8$ (\\%)','Gain (pp)'],body,'Contiguous and4s-gap observations share eligible groups and anchors. The12s-gap cohort is smaller and different.'.replace('and4s-gap','and 4 s-gap').replace('The12s-gap','The 12 s-gap'))
# Three-module overview, with trajectory reconstruction occupying the central panel.
fig,ax=plt.subplots(figsize=(WIDTH,4.05));fig.subplots_adjust(left=.015,right=.985,top=.99,bottom=.02);ax.set_xlim(0,12);ax.set_ylim(0,7.3);ax.axis('off')
def box(x,y,w,h,txt='',fc='#F3F6F8',ec='#657888',size=8.7,ls='-',lw=.8):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.05,rounding_size=.07',facecolor=fc,edgecolor=ec,linewidth=lw,linestyle=ls))
 if txt:ax.text(x,y,txt,ha='center',va='center',fontsize=size,linespacing=1.45)
def arrow(a,b,style='-'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,lw=.9,color='#50606C',linestyle=style))
box(1.85,5.5,3.35,3.2,fc='#F1F5F9')
ax.text(1.85,6.75,'1. Data collection',ha='center',va='center',fontsize=10,fontweight='bold')
box(1.85,6.04,2.95,.6,'Roadside video')
box(1.85,5.20,2.95,.6,'Reference trajectories')
box(1.85,4.36,2.95,.6,'Driving-state labels')
box(8,5.5,7.3,3.2,fc='#EAF4F0',ec=PALETTE['green'],lw=1.3)
ax.text(8,6.75,'2. Trajectory reconstruction',ha='center',va='center',fontsize=10,fontweight='bold')
box(6.1,5.92,3.0,.75,'BEVHeight-based\n3D localization',fc='white',ec='#4F8475')
box(9.7,5.92,3.0,.75,'Road-coordinate\nregistration',fc='white',ec='#4F8475')
box(6.1,4.65,3.0,.75,'Temporal association\nand track continuity',fc='white',ec='#4F8475')
box(9.7,4.65,3.0,.75,'Trajectory refinement\nand motion estimation',fc='white',ec='#4F8475')
arrow((3.58,5.92),(4.52,5.92));arrow((7.65,5.92),(8.15,5.92));arrow((9.7,5.49),(9.7,5.09));arrow((6.1,5.49),(6.1,5.09));arrow((7.65,4.65),(8.15,4.65))
# Reference records support the trajectory module; this is a system schematic.
ax.text(3.98,5.45,'Aligned\nrecords',ha='center',va='center',fontsize=7.2)
arrow((8,3.85),(8,3.4))
box(8,3.0,7.3,.7,'Metric trajectories\nIdentity · time · position · velocity · acceleration',fc='#FFFFFF',ec=PALETTE['green'],size=8.2)
arrow((8,2.59),(8,2.15))
box(8,1.15,7.3,1.8,fc='#EEF3FA',ec=PALETTE['blue'],lw=1.1)
ax.text(8,1.77,'3. AV identification',ha='center',va='center',fontsize=10,fontweight='bold')
ax.text(8,1.15,'CNN  /  Transformer  /  Random forest',ha='center',va='center',fontsize=9)
ax.text(8,.62,'Single or repeated trajectory segments\nDriving-state classification',ha='center',va='center',fontsize=8.1)
box(1.85,1.55,3.35,2.15,'Other trajectory applications\n\nCar-following analysis\nTraffic-state estimation\nSafety indicators',fc='#FAFAFA',ec='#999999',size=8.4,ls='--')
arrow((4.3,3),(1.85,3),'--');arrow((1.85,3),(1.85,2.68),'--')
save(fig,'roadside_framework')
# Main accuracy plot at its actual 0.76*textwidth printed size.
fig,ax=plt.subplots(figsize=(.76*WIDTH,2.7));fig.subplots_adjust(left=.13,right=.97,bottom=.22,top=.81)
for m,name,c,mk in models:
 z=r[m].loc[[1,2,4,8]];ax.errorbar(np.arange(4),100*z.accuracy_mean.to_numpy(),yerr=100*z.accuracy_std.to_numpy(),color=c,marker=mk,capsize=2.5,label=name)
ax.set_xticks(range(4),['1','2','4','8']);ax.set_ylim(58,80);ax.set_yticks([60,65,70,75,80]);ax.set_ylabel('Accuracy (%)');ax.set_xlabel('Number of 4 s segments, K')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.54,.99));save(fig,'roadside_identification_accuracy')
# Source transfer and synthetic misassociation.
per=pd.read_csv(M/'per_source.csv');mis=pd.read_csv(M/'misassociation_summary.csv',skiprows=3,header=None,names=['model','rate','accuracy','sd','bce','bce_sd'])
fig,axs=plt.subplots(1,2,figsize=(WIDTH,2.85),gridspec_kw={'width_ratios':[1.25,1]});fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.75,wspace=.32)
a,b=axs
for j,(m,name,c,mk) in enumerate(models):
 for fold in range(3):
  z=per.query("model==@m and regime=='gap4' and fold==@fold").set_index('k');x=fold+(j-1)*.16;y0=100*z.loc[1,'accuracy_mean'];y1=100*z.loc[8,'accuracy_mean']
  a.plot([x,x],[y0,y1],color=c,lw=1.3);a.plot(x,y0,marker=mk,color=c,markerfacecolor='white',ms=4);a.plot(x,y1,marker=mk,color=c,ms=4,label=name if fold==0 else None)
 z=mis.query('model==@m');b.plot(100*z.rate.to_numpy(),100*z.accuracy.to_numpy(),color=c,marker=mk)
a.set_xticks(range(3),['Part 2','Part 3','Highway']);a.set_ylim(45,95);a.set_yticks([50,60,70,80,90]);a.set_ylabel('Accuracy (%)');a.set_xlabel('Held-out source');a.text(0,1.10,'(a) Source transfer',transform=a.transAxes,fontsize=9)
b.set_xlim(-2,52);b.set_ylim(60,80);b.set_xticks([0,10,25,50]);b.set_yticks([60,65,70,75,80]);b.set_xlabel('Replacement probability (%)');b.text(0,1.10,'(b) Association sensitivity',transform=b.transAxes,fontsize=9)
fig.legend(*a.get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.54,.99));save(fig,'roadside_identification_robustness')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[M/x for x in ['summary.csv','per_source.csv','misassociation_summary.csv','cohort.csv','observation_budget.csv','checkpoint_audit.json']]+[D/'counts.csv']
(PAPER/'display_manifest.json').write_text(json.dumps({'date':'2026-10-18','figure_full_width_inches':WIDTH,'figure_accuracy_width_inches':.76*WIDTH,'template':'uploaded cas-sc.cls, unchanged','inputs':{str(p.relative_to(EVIDENCE)):sha(p) for p in inputs},'outputs':{str(p.relative_to(PAPER)):sha(p) for p in outputs}},indent=2)+'\n')
print('Generated 3 vector figures and 4 numerical tables.')
