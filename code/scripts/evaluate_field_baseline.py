"""Export and evaluate the frozen transfer baseline without fitting field labels."""
import json,sys
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from build_field_core_dataset import ROOT,read,save,convert
PROJECT=ROOT/'classification/trajectory-avlike-transformer'
sys.path.insert(0,str(PROJECT/'inference_cnn'))
from networks import TrajectoryNet
from data_protocol import features,normalize

def metrics(frame,prob='prob_av'):
 if not len(frame):return dict(n=0,accuracy=None,balanced_accuracy=None,macro_f1=None,confusion_HV_AV=None)
 y=frame.label.to_numpy(int);pred=(frame[prob].to_numpy()>=.5).astype(int);both=len(set(y))==2
 return dict(n=len(y),class_counts=dict(Counter(map(str,y))),correct=int((y==pred).sum()),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,macro_f1=float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)) if both else None,confusion_HV_AV=confusion_matrix(y,pred,labels=[0,1]).tolist())

def nonoverlap(group):
 rows=[];end=-float('inf')
 for _,r in group.sort_values('start_s').iterrows():
  if r.start_s>=end+.1-1e-6:rows.append(r);end=r.end_s
 return pd.DataFrame(rows)

def main():
 torch.set_num_threads(4);cfg=read(ROOT/'configs/field_baseline.json');out=ROOT/'experiments'/cfg['experiment'];all_windows=[];reports=[]
 for clip in cfg['clips']:
  report,windows=convert(dict(clip=clip,tag=cfg['tag'],role='exploratory_test'),cfg,out);reports.append(report);all_windows+=windows
  print('EXPORT',clip,report.get('identified_target'),report.get('windows_by_duration'),flush=True)
 save(out/'dataset_summary.json',dict(selected_clips=cfg['clips'],processed_runs=reports,speed_floor_mps=None))
 members=[]
 for p in sorted((PROJECT/'artifacts').rglob('best_bce.pt')):
  relative=str(p.relative_to(PROJECT/'artifacts')).lower()
  if 'transformer' not in relative and 'cnn' not in relative:continue
  cp=torch.load(p,map_location='cpu',weights_only=True);c=cp['config']
  if c['architecture'] not in ['transformer','multiscale_cnn'] or c.get('study')=='multisegment':continue
  model=TrajectoryNet(cp['input_dim'],c['architecture'],c['width'],c['dropout'],c['layers']).eval();model.load_state_dict(cp['model_state_dict'],strict=True)
  members.append((p,model,c,read(p.parent/'feature_stats.json')))
 save(out/'model_manifest.json',[dict(path=str(p.relative_to(ROOT)),architecture=c['architecture'],config=c) for p,m,c,s in members]);results=[]
 for duration in cfg['classification']['window_durations_s']:
  selected=[(x,m) for x,m in all_windows if m['duration_s']==duration];n=int(round(duration*10));meta=pd.DataFrame([dict(m,window_index=i) for i,(x,m) in enumerate(selected)])
  raw=np.stack([x for x,m in selected]) if selected else np.empty((0,n,3),np.float32)
  np.savez_compressed(out/f'windows_{duration:g}s.npz',primitive_windows=raw,labels=np.array([m['label'] for _,m in selected],np.int64),dt=np.array(.1),split_group=np.array([m['clip'] for _,m in selected],dtype=str))
  meta.to_csv(out/f'windows_{duration:g}s.csv',index=False)
  if not len(raw):results.append(dict(duration_s=duration,windows=0,status='NO_WINDOWS'));continue
  assert np.isfinite(raw).all() and (raw[:,:,:2]>=0).all() and (raw[:,:,2]>0).all()
  for architecture in ['transformer','multiscale_cnn']:
   probabilities=[];clipping=[];member_metrics=[]
   for path,model,c,stats in members:
    if c['architecture']!=architecture:continue
    x=features(raw,c.get('features','legacy11'),c.get('smoothing',7));clipping.append(float(((x<np.array(stats['lower'])) | (x>np.array(stats['upper']))).mean()))
    with torch.no_grad():prob=torch.sigmoid(model(torch.from_numpy(normalize(x,stats)))).numpy()
    assert prob.shape==(len(raw),) and np.isfinite(prob).all();probabilities.append(prob)
    f=meta.copy();f['prob_av']=prob;first=f.sort_values('start_s').groupby('clip',sort=False).head(1);member_metrics.append(dict(model=str(path.relative_to(ROOT)),first_window_per_clip=metrics(first)))
   if not probabilities:continue
   ps=np.stack(probabilities);pred=meta.copy();pred['prob_av']=ps.mean(0);pred['model_sd']=ps.std(0);pred['predicted_label']=(pred.prob_av>=.5).astype(int);pred.to_csv(out/f'{architecture}_{duration:g}s_predictions.csv',index=False)
   # One prespecified first eligible window per clip is the headline single-segment unit.
   first=pred.sort_values('start_s').groupby('clip',sort=False).head(1);first.to_csv(out/f'{architecture}_{duration:g}s_first_per_clip.csv',index=False)
   fusion=[];matched=[]
   for clip,g in pred.groupby('clip',sort=False):
    disjoint=nonoverlap(g)
    # Same target/state/recording, two windows with no shared frames, fixed earliest-two rule.
    if len(disjoint)<2:continue
    pair=disjoint.head(2);fusion.append(dict(clip=clip,label=int(pair.iloc[0].label),prob_av=float(pair.prob_av.mean()),windows=2,observed_duration_s=2*duration,start_s=float(pair.start_s.min()),end_s=float(pair.end_s.max()),window_indices=pair.window_index.tolist()));matched.append(pair.iloc[0].to_dict())
   fused=pd.DataFrame(fusion);matched=pd.DataFrame(matched);save(out/f'{architecture}_{duration:g}s_fusion.json',fusion)
   row=dict(architecture=architecture,duration_s=duration,ensemble_members=len(probabilities),windows=len(raw),recordings=len(first),coverage=len(first)/len(cfg['clips']),single_first_window_per_clip=metrics(first),all_windows_descriptive_only=metrics(pred),fusion_earliest_two_nonoverlap=metrics(fused),single_on_fusion_cohort=metrics(matched),source_feature_clipping_fraction_range=[min(clipping),max(clipping)],source_model_first_window_accuracy_range=[min(r['first_window_per_clip']['accuracy'] for r in member_metrics),max(r['first_window_per_clip']['accuracy'] for r in member_metrics)],speed_range_mps=[float(raw[:,:,:2].min()),float(raw[:,:,:2].max())],gap_range_m=[float(raw[:,:,2].min()),float(raw[:,:,2].max())])
   results.append(row);save(out/f'{architecture}_{duration:g}s_member_metrics.json',member_metrics);print('RESULT',json.dumps(row),flush=True)
 save(out/'classification_summary.json',dict(mode='Frozen source models; no field training, tuning, threshold selection, or label-dependent exclusions',overall_field_accuracy=None,overall_field_accuracy_note='Not estimable from this one-class, one-recording usable subset; per-window correctness below is diagnostic only',threshold=.5,primary_duration_s=2,primary_unit='First eligible 2 s window of each recording; mean probability across all 15 restored members of the specified architecture',supplementary_durations_s=[1,3,4],selected_recordings=6,selected_class_counts={'AV':3,'HV':3},results=results,limits=['Small exploratory cohort, not publication-quality generalization estimate','Source windows used 4 s; short prefixes execute but have not been trained on these field durations','Missing leader windows are not passed to the current 11-feature model','All-window results are correlated, not independent observations','Two-window probability averaging is an initial fusion baseline, not retraining the historical multisegment architecture']))

if __name__=='__main__':main()
