"""CPU model-loading check on synthetic inputs; no data, fitting or accuracy claim."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit
import torch

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'training/cnn'))
from networks import TrajectoryNet
from data_protocol import features,normalize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    raw=np.empty((8,40,3),np.float32)
    t=np.arange(40,dtype=np.float32)*.1
    for i in range(8):
        raw[i,:,0]=27+.02*i+.1*np.sin(t)
        raw[i,:,1]=28+.02*i+.1*np.sin(t)
        raw[i,:,2]=30+t
    x=features(raw,'legacy11',7)
    records=[]
    for p in sorted((root/'artifacts').rglob('best_bce.pt')):
        checkpoint=torch.load(p,map_location='cpu',weights_only=True)
        cfg=checkpoint['config']
        model=TrajectoryNet(11,cfg['architecture'],cfg['width'],cfg['dropout'],cfg['layers']).eval()
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        stats=json.loads((p.parent/'feature_stats.json').read_text())
        tx=torch.from_numpy(normalize(x,stats))
        with torch.no_grad():
            z=(model.classifier(model.encode(tx).mean(0,keepdim=True)).reshape(-1)
               if cfg.get('study')=='multisegment' else model(tx))
            prob=torch.sigmoid(z).numpy()
        assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
        records.append({'path':p.relative_to(root).as_posix(),'kind':cfg.get('mode',cfg['architecture']),'status':'PASS'})
    for p in sorted((root/'artifacts').rglob('model.joblib')):
        model=joblib.load(p)
        stats=json.loads((p.parent/'feature_stats.json').read_text())
        prob=model.predict_proba(normalize(x,stats).reshape(len(x),-1))
        assert prob.shape==(8,2) and np.isfinite(prob).all() and np.allclose(prob.sum(1),1)
        records.append({'path':p.relative_to(root).as_posix(),'kind':'rf','status':'PASS'})
    assert len(records)==75,f'Expected 75 principal models, found {len(records)}'
    result={'scope':'CPU restoration and finite predictions on synthetic inputs; no accuracy or training test',
            'models_checked':len(records),'versions':{k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','pandas','scikit-learn','joblib']},'records':records}
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS: {len(records)} models loaded with finite CPU outputs (synthetic inputs).')


if __name__=='__main__':main()
