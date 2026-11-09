"""Apply restored CNN/Transformer ensembles to eligible camera-derived windows."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent.parent
PROJECT=ROOT/'classification/trajectory-avlike-transformer'
sys.path.insert(0,str(PROJECT/'inference_cnn'))
from predict import predict,check_windows
from data_protocol import features,normalize
from networks import TrajectoryNet

@torch.no_grad()
def transformer(raw):
    _,x=check_windows(raw);probs=[]
    for path in sorted((PROJECT/'artifacts').rglob('best_bce.pt')):
        if 'transformer' not in str(path.relative_to(PROJECT/'artifacts')).lower():continue
        ckpt=torch.load(path,map_location='cpu',weights_only=True);cfg=ckpt['config']
        if cfg['architecture']!='transformer' or cfg.get('study')=='multisegment':continue
        model=TrajectoryNet(ckpt['input_dim'],cfg['architecture'],cfg['width'],cfg['dropout'],cfg['layers']).eval()
        model.load_state_dict(ckpt['model_state_dict'],strict=True)
        stats=json.loads((path.parent/'feature_stats.json').read_text())
        xx=torch.from_numpy(normalize(x,stats))
        probs.append(torch.cat([torch.sigmoid(model(b)) for b in xx.split(512)]).numpy())
    if len(probs)!=15:raise RuntimeError(f'Expected 15 Transformer members, got {len(probs)}')
    p=np.stack(probs);return pd.DataFrame(dict(acc_probability=p.mean(0),predicted_label=(p.mean(0)>=.5).astype(int),model_probability_sd=p.std(0),models=15))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='pilot_base');a=ap.parse_args()
    exp=ROOT/'experiments/field_pilot_20261101'/a.tag
    raw=np.load(exp/'windows_4s.npz',allow_pickle=False)['primitive_windows'];meta=pd.read_csv(exp/'windows.csv')
    torch.set_num_threads(4)
    if not len(raw):
        result=dict(status='NO_ELIGIBLE_FIELD_WINDOWS',rows=0,accuracy=None,classifier_finetuning='not run: no eligible labeled windows; speed gate and duration not relaxed')
    else:
        cnn,_=predict(raw,source_ids=meta.clip.astype(str).to_numpy());tf=transformer(raw)
        for name,pred in [('cnn',cnn),('transformer',tf)]:pd.concat([meta.reset_index(drop=True),pred],axis=1).to_csv(exp/f'{name}_predictions.csv',index=False)
        result=dict(status='PREDICTED',rows=len(raw),labeled_windows=int((meta.label>=0).sum()),accuracy=None,note='Probabilities on new video domain; correlated windows are not independent vehicles. Accuracy withheld pending a valid labeled evaluation cohort.')
    (exp/'classification_status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
