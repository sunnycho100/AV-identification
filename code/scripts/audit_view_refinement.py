"""Verify artifacts of the completed local adaptation pilot."""
import ast,hashlib
import cv2,numpy as np,torch
from verify_field_branches import ROOT,TR,read,save

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);cfg=read(ROOT/'configs/view_refinement.json');base=read(ROOT/'configs/field_baseline.json');out=ROOT/'experiments'/cfg['experiment']
    original=torch.load(TR/cfg['base_checkpoint'],map_location='cpu')['state_dict'];original={k[6:] if k.startswith('model.') else k:v for k,v in original.items()}
    cp=out/'V_shared_class_ep5.ckpt';new=torch.load(cp,map_location='cpu')['state_dict'];assert set(new)==set(original)
    changed=[k for k in new if not torch.equal(new[k],original[k])];assert changed and all(k.startswith('head.shared_conv.') or k.startswith('head.task_heads.') and '.heatmap.' in k for k in changed)
    assert all(torch.isfinite(v).all() for v in new.values());assert changed==read(out/'training.json')['changed_keys']
    assert not set(sum(cfg['train'].values(),[]))&set(sum(cfg['development'].values(),[]))
    frames=0;following_frames=0
    def check_pair(c,oldtag,newtag,start,end):
        nonlocal frames
        a=TR/'outputs/object_detection/camera-data'/f'{c}_{oldtag}';b=TR/'outputs/object_detection/camera-data'/f'{c}_{newtag}'
        expected=set(range(start,end+1));assert {int(p.stem.split('_')[0]) for p in b.glob('*_pred.json')}==expected
        for k in ['K','lidar2cam','road_plane_z_in_output']:assert read(a/'calibration_used.json')[k]==read(b/'calibration_used.json')[k]
        ra=read(TR/'outputs/trajectories'/f'{c}_{oldtag}'/'run.json');rb=read(TR/'outputs/trajectories'/f'{c}_{newtag}'/'run.json')
        assert {k:v for k,v in ra['settings'].items() if k!='tag'}=={k:v for k,v in rb['settings'].items() if k!='tag'}
        return len(expected)
    for c in cfg['development']['V']:
        row=next(r for r in base['excerpts'] if r['clip']==c);frames+=check_pair(c,base['tag'],cfg['tag'],row['start'],row['end'])
        assert read(TR/'outputs/object_detection/camera-data'/f'{c}_{cfg["tag"]}'/'calibration_used.json')['checkpoint_sha256']==sha(cp)
        assert cv2.imread(str(out/f'{c}_before_after.jpg')).shape==(1320,1440,3)
    fc=read(ROOT/'configs/seminole_following.json')
    for row in fc['excerpts']:
        c=row['clip'];following_frames+=check_pair(c,fc['models']['original']['tag'],fc['models']['adapted']['tag'],row['start'],row['end'])
        for mc in fc['models'].values():
            dd=TR/'outputs/object_detection/camera-data'/f'{c}_{mc["tag"]}';assert len(list(dd.glob('*_pred.json')))==row['end']-row['start']+1
            assert read(dd/'calibration_used.json')['checkpoint_sha256']==sha(ROOT/mc['checkpoint'])
    arrays=[]
    for folder in [out,out/'seminole_following/original',out/'seminole_following/adapted']:
        for dur in [1,2,3,4]:
            with np.load(folder/f'windows_{dur}s.npz') as a:
                x=a['primitive_windows'];assert x.shape[1:]==(dur*10,3) and np.isfinite(x).all();assert len(x)==len(a['labels'])
                if len(x):assert (x[:,:,2]>0).all()
                arrays.append(dict(path=str((folder/f'windows_{dur}s.npz').relative_to(ROOT)),shape=list(x.shape)))
    for name in ['refine_field_views.py','run_seminole_following.py','diagnose_view_refinement.py','classify_view_refinement.py','prepare_view_refinement.py','visualize_field_baseline.py']:
        ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8'))
    report=dict(status='PASS',training_frames=len(read(out/'training_annotations.json')['annotations']),trainable_parameters=read(out/'training.json')['trainable_parameters'],checkpoint_sha256=sha(cp),changed_state_keys=len(changed),verona_inference_frames=frames,seminole_source_frames=following_frames,seminole_model_frame_evaluations=2*following_frames,arrays=arrays,checks=['Saved tensors finite; only shared detector convolution and class heatmaps changed','No recording overlap between adaptation training and development','Identical comparison calibration and tracker settings','All inference excerpts complete; checkpoint hashes match','Candidate arrays finite, correct shapes and positive gaps','Edited/new integration scripts parse'],scientific_limit='Engineering checks only; false positives, wrong tails, speed bias and unresolved leaders remain.')
    save(out/'validation.json',report);print(report)

if __name__=='__main__':main()
