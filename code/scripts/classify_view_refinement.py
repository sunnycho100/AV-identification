"""Frozen downstream diagnostic on automatic candidate windows from this pilot.

Geometry is unvalidated; output must not be quoted as field generalization accuracy.
"""
import numpy as np,pandas as pd,torch
from evaluate_field_baseline import ROOT,PROJECT,read,save,TrajectoryNet,features,normalize,metrics

def main():
    torch.set_num_threads(4);out=ROOT/'experiments/view_refinement_20261101';members=[]
    for row in read(ROOT/'experiments/field_baseline_20261101/model_manifest.json'):
        p=ROOT/row['path'];cp=torch.load(p,map_location='cpu',weights_only=True);c=cp['config']
        model=TrajectoryNet(cp['input_dim'],c['architecture'],c['width'],c['dropout'],c['layers']).eval();model.load_state_dict(cp['model_state_dict'],strict=True)
        members.append((row['path'],model,c,read(p.parent/'feature_stats.json')))
    results=[]
    for dur in [1,2,3,4]:
        raw=np.load(out/f'windows_{dur}s.npz')['primitive_windows'];meta=pd.DataFrame(read(out/f'windows_{dur}s_metadata.json'))
        if not len(raw):results.append(dict(duration_s=dur,status='NO_WINDOWS',windows=0));continue
        assert np.isfinite(raw).all() and (raw[:,:,2]>0).all()
        for arch in ['transformer','multiscale_cnn']:
            ps=[];paths=[];clipping=[]
            for path,model,c,stats in members:
                if c['architecture']!=arch:continue
                x=features(raw,c.get('features','legacy11'),c.get('smoothing',7));clipping.append(float(((x<np.array(stats['lower']))|(x>np.array(stats['upper']))).mean()))
                with torch.no_grad():p=torch.sigmoid(model(torch.from_numpy(normalize(x,stats)))).numpy()
                assert p.shape==(len(raw),) and np.isfinite(p).all();ps.append(p);paths.append(path)
            assert len(ps)==15
            pred=meta.copy();pred['prob_av']=np.mean(ps,axis=0);pred['model_sd']=np.std(ps,axis=0);pred['predicted_label']=(pred.prob_av>=.5).astype(int);pred.to_csv(out/f'{arch}_{dur}s_predictions.csv',index=False)
            first=pred.sort_values('start_s').groupby('clip',sort=False).head(1)
            results.append(dict(duration_s=dur,architecture=arch,windows=len(raw),members=15,first_per_recording=metrics(first),all_windows_correlated=metrics(pred),source_feature_clipping_range=[min(clipping),max(clipping)],first_predictions=first[['clip','label','start_s','end_s','prob_av','predicted_label']].to_dict('records')))
    report=dict(status='DIAGNOSTIC_ONLY_GEOMETRY_UNVALIDATED',training='None; all 30 source models frozen',overall_field_accuracy=None,primary_duration_s=2,threshold=.5,results=results,limitations=['Two previously inspected Verona recordings, not blind test','1 s and 2 s inputs differ from original 4 s source training','Automatic pair visibility and dimensions do not independently validate metric speeds/gaps','HV false track tail can affect earlier RTS velocities; see motion_check.json and review images','No missing-leader padding; no label-based window selection; overlapping windows are not independent'])
    save(out/'classification_summary.json',report);print(report)

if __name__=='__main__':main()
