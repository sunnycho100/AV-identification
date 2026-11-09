"""Post-hoc frozen-model/input crossover; no fitting or test-guided model selection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from todd_experiment_train import MaskedTransformer, tensors, predict
from variable_length_classifier import VariableTransformer, normalize, metrics

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'experiments/field_todd_complete_20261102'
NEW=ROOT/'experiments/field_todd_variable_20261103'
OUT=NEW/'diagnostics/model_input_crossover'


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    old=dict(np.load(OLD/'dataset.npz'));new=dict(np.load(NEW/'dataset.npz'))
    oldmeta=pd.read_csv(OLD/'windows.csv');newmeta=pd.read_csv(NEW/'windows.csv')
    clips=newmeta['clip'].tolist();labels=newmeta.label.to_numpy();old['clip_index']=np.array([clips.index(c) for c in oldmeta['clip']])
    new['clip_index']=np.arange(len(clips))
    # The original frozen model's missing/state projection is identically zero.
    # No inferred human relation category is needed for this frozen-model test.
    new['relation_state']=np.zeros((*new['valid_tokens'].shape,3),np.float32)
    cfg=json.loads((ROOT/'configs/field_todd_complete.json').read_text())
    path=ROOT/cfg['initial_checkpoint'];cp=torch.load(path,map_location='cpu',weights_only=True)
    stats=json.loads((path.parent/'feature_stats.json').read_text());st={k:stats[k] for k in ['mean','std']}
    source={k:np.asarray(v) for k,v in st.items()}
    original=MaskedTransformer(cp,st,source).eval()
    assert torch.count_nonzero(original.missing_proj.weight)==0
    probabilities={};details=[]
    for input_name,ds in [('old_windows',old),('new_windows',new)]:
        tx=tensors(ds,st,'masked');pr=predict(original,tx,np.arange(len(ds['features'])),ds['clip_index'])
        probabilities[('old_source',input_name)]=np.array([pr[i] for i in range(len(clips))])
        pp=[]
        for seed in [42,52,62]:
            ck=torch.load(NEW/f'public/seed{seed}.pt',map_location='cpu',weights_only=False)
            model=VariableTransformer().eval();model.load_state_dict(ck['model_state_dict'])
            x=torch.from_numpy(normalize(ds['features'],ds['feature_mask'],ck['normalization']))
            with torch.no_grad():p=torch.sigmoid(model(x,torch.from_numpy(ds['feature_mask']),torch.from_numpy(ds['valid_tokens']))).numpy()
            pp.append(np.array([p[ds['clip_index']==i].mean() for i in range(len(clips))]))
        probabilities[('new_source',input_name)]=np.stack(pp).mean(0)
    frozen=pd.read_csv(OLD/'predictions_by_video.csv');frozen=frozen[frozen.model=='source_transformer_masked'].set_index('clip').loc[clips]
    current=pd.read_csv(NEW/'field/video_predictions.csv').set_index('clip').loc[clips]
    assert np.allclose(probabilities[('old_source','old_windows')],frozen.prob_av,atol=1e-5)
    assert np.allclose(probabilities[('new_source','new_windows')],current.source_only_probability,atol=1e-5)
    summary=[]
    for (model,inputs),p in probabilities.items():
        summary.append(dict(model=model,inputs=inputs,**metrics(labels,p)))
        for clip,label,prob in zip(clips,labels,p):details.append(dict(model=model,inputs=inputs,clip=clip,label=int(label),probability=float(prob),correct=bool((prob>=.5)==label)))
    table=pd.DataFrame(summary);table.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'predictions.csv',index=False)
    adapted=pd.read_csv(OLD/'predictions_by_video.csv');adapted=adapted[adapted.model=='finetuned_transformer_masked'].set_index('clip').loc[clips]
    comparison=pd.DataFrame(dict(clip=clips,label=labels,old_adapted_probability=adapted.prob_av.to_numpy(),new_adapted_probability=current.probability.to_numpy(),
        old_correct=adapted.correct.to_numpy(),new_correct=current.correct.to_numpy()))
    comparison.to_csv(OUT/'old_new_adapted_comparison.csv',index=False)
    manifest={'status':'complete','classification_models_trained':0,'saved_predictions_reproduced':True,
        'source_checkpoint_config':cp['config'],'old_window_count':len(oldmeta),'new_window_count':len(newmeta),
        'old_window_durations_s':sorted(oldmeta.duration_s.round(1).unique().tolist()),
        'scope':'Post-hoc descriptive crossover of two already-trained SOURCE models and two frozen input sets. Old/new inputs differ in trajectories, selected intervals, leader availability and number/aggregation of windows; model factor includes architecture, source data and training history. Not a causal isolation of duration or field fine-tuning.',
        'old_source_missing_projection_zero':True,'source_checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'results':summary}
    (OUT/'report.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(table[['model','inputs','n','correct','accuracy']].to_string(index=False),flush=True)


if __name__=='__main__':main()
