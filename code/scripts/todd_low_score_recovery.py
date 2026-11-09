"""Re-run image-supported missing frames; preserve original predictions and weights."""
import sys
import argparse
import time
import numpy as np
import torch
import verify_field_branches
from scripts.finetune import train_finetune as T
from scripts.object_detection import run_bevheight_generic as G
from todd_three_second_tracks import ROOT,OUT,CLIPS,context,read,save
from track_review_seeds import iou,contact
from visualize_field_baseline import project


def main(feature=False):
    torch.set_num_threads(4)
    checkpoint='ft102feat_run5_feat_ep5.ckpt' if feature else 'ft102_run3_v2labels_ep5.ckpt'
    model=T.build_model();info=T.load_checkpoint(model,ROOT/'trajectory/checkpoints'/checkpoint)
    assert not info['missing'] and not info['unexpected'],info
    model.head.bbox_coder.score_threshold=.05
    model.head.test_cfg['score_threshold']=.05
    model.to(G.DEVICE).eval();exp=G.load_exp(G.DEFAULT_CONFIG)
    print('Recovery',G.DEVICE,'102.4m config; decoder/export threshold 0.05',flush=True)
    summaries=[];start=time.time()
    for clip in (['AV_T_WE_4','HV_T_WE_5'] if feature else CLIPS):
        r,folder,dd,cal,K,M,times=context(clip)
        doc=read(OUT/'candidates'/clip/'image_candidates.json');rows=doc['rows'];extra=0;tried=0
        dest=OUT/('feature_recovery' if feature else 'low_score_detections')/clip;dest.mkdir(parents=True,exist_ok=True)
        fm=None
        if feature:
            from scripts.object_detection import mask_input_frames as mi
            fm=mi.feature_mask(K,M[:3,:3],M[:3,3]-M[:3,:3]@np.array([0.,0.,1.73]))
            np.save(dest/'feature_mask.npy',fm)
        for row in rows:
            if feature:
                if not (88<=row['frame']<=191 if clip=='AV_T_WE_4' else 52<=row['frame']<=160):continue
                row['matched_detection']=None
            elif row.get('matched_detection'):continue
            path=dest/f'{row["frame"]:03d}_pred.json'
            if not path.exists():save(path,G.run_frame(model,folder/f'{row["frame"]:03d}.jpg',K,M,exp,score_thresh=.05,feat_mask=fm))
            ds=read(path);matches=[];tried+=1
            for j,d in enumerate(ds):
                if d['class_name'] not in ('car','truck','bus'):continue
                uv=project(d,K,M)
                if uv is None:continue
                bb=np.r_[uv.min(0),uv.max(0)];value=iou(row['bbox'],bb)
                if value>0:matches.append(dict(index=j,iou=value,bbox=bb.tolist(),detection=d))
            matches.sort(key=lambda v:v['iou'],reverse=True)
            if matches and matches[0]['iou']>=.35 and (len(matches)==1 or matches[0]['iou']-matches[1]['iou']>=.15):
                row['matched_detection']=matches[0];row['low_threshold_recovery']=True;extra+=1
            row['low_threshold_top_candidates']=matches[:2]
            if tried%50==0:print(clip,'tried',tried,'recovered',extra,'elapsed',round(time.time()-start),flush=True)
        doc['review_status']='pending_after_low_threshold';save(dest/'image_candidates.json',doc)
        contact(rows,folder,dest/'candidate_review.jpg',clip)
        s=dict(clip=clip,tried=tried,extra_matches=extra,total_matches=sum(bool(v.get('matched_detection')) for v in rows))
        summaries.append(s);save(OUT/('feature_recovery_summary.json' if feature else 'low_score_summary.json'),summaries);print(s,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('--feature',action='store_true');a=ap.parse_args();main(a.feature)
