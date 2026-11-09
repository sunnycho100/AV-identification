"""Image-supported Todd recovery; candidates require explicit visual adjudication."""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from build_field_core_dataset import ROOT,read,save,inside_image
from visualize_field_baseline import project
from track_review_seeds import adaptive_flow_direction,annotate,contact

OUT=ROOT/'experiments/field_todd_3s_20261102'
OLD=ROOT/'experiments/field_review_test_20261102'
CLIPS=read(ROOT/'configs/field_todd_focus.json')['primary_clips']
RECORDS={r['clip']:r for r in read(ROOT/'experiments/field_review_applied_20261102/decisions_source.json')['records']}


def context(clip):
    r=RECORDS[clip];tag=r['tag'];folder=ROOT/'trajectory/data/camera-data'/clip/'frames_all'
    dd=ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{tag}'
    cal=read(dd/'calibration_used.json');K,M=np.array(cal['K']),np.array(cal['lidar2cam'])
    times={s['frame']:s['t_s'] for s in read(folder.parent/'frame_times.json')['frames']}
    return r,folder,dd,cal,K,M,times


def candidates(only=None):
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True)
    summary=[]
    for clip in (only or CLIPS):
        r,folder,dd,cal,K,M,times=context(clip);dest=OUT/'candidates'/clip;dest.mkdir(parents=True,exist_ok=True)
        original=pd.read_csv(OLD/'clips'/clip/'target_trajectory.csv',dtype={'track_id':str})
        dims=read(OLD/'clips'/clip/'dimensions.json');seeds=[]
        for j in sorted(set([0,len(original)//2,len(original)-1])):
            s=original.iloc[j].to_dict();dim=dims[s['track_id']]
            uv=project(dict(s,l=dim['length_m'],w=dim['width_m'],h=dim['height_m']),K,M)
            if uv is None:continue
            bb=np.r_[uv.min(0),uv.max(0)];bb[:2]=np.maximum(bb[:2],1);bb[2:]=np.minimum(bb[2:],[1918,1078])
            if min(bb[2:]-bb[:2])<14:continue
            seeds.append(dict(frame=int(s['frame']),bbox=bb.tolist()))
        # Original user seed remains the strongest identity anchor when supplied.
        user=r.get('target',{})
        if 'bbox' in user and user.get('type')=='track':seeds.append(dict(frame=user['frame'],bbox=user['bbox']))
        seeds_path=OUT/'image_anchors.json'
        if seeds_path.exists() and clip in read(seeds_path):seeds=read(seeds_path)[clip]
        allrows={};stops=[]
        for seed in seeds:
            sf=seed['frame'];low=max(0,min(r['start_frame'],int(original.frame.min()))-100);high=min(max(times),max(r['end_frame'],int(original.frame.max()))+100)
            for f in range(sf,low,-1):
                if times[f]-times[f-1]>.2:low=f;break
            for f in range(sf,high):
                if times[f+1]-times[f]>.2:high=f;break
            back,bs=adaptive_flow_direction(folder,seed,low) if low<sf else ([],None)
            forward,fs=adaptive_flow_direction(folder,seed,high) if high>sf else ([],None)
            rows=annotate(sorted(back+[dict(seed,image_correlation=1.,source='known_identity_anchor')]+forward,key=lambda s:s['frame']),clip,r['tag'],OUT)
            for row in rows:
                row['anchor_frame']=sf
                allrows.setdefault(row['frame'],[]).append(row)
            stops.append(dict(seed=seed,backward=bs,forward=fs,range=[rows[0]['frame'],rows[-1]['frame']]))
        merged=[]
        for f,choices in sorted(allrows.items()):
            # Keep every anchor's evidence; pick closest-in-time supported match.
            good=[v for v in choices if v.get('matched_detection')]
            best=min(good or choices,key=lambda v:abs(f-v['anchor_frame']))
            best['anchor_votes']=[dict(anchor=v['anchor_frame'],bbox=v['bbox'],matched_index=v['matched_detection']['index'] if v.get('matched_detection') else None) for v in choices]
            merged.append(best)
        save(dest/'image_candidates.json',dict(clip=clip,stops=stops,rows=merged,review_status='pending'))
        if merged:contact(merged,folder,dest/'candidate_review.jpg',clip)
        origframes=set(original.frame)
        matched=[s for s in merged if s.get('matched_detection')]
        additions=[s['frame'] for s in matched if s['frame'] not in origframes]
        record=dict(clip=clip,original_points=len(original),image_frames=len(merged),matched=len(matched),
                    additional_detection_candidates=len(additions),addition_range=[min(additions),max(additions)] if additions else [],stops=stops)
        summary.append(record);print(clip,'image',len(merged),'matched',len(matched),'new',len(additions),flush=True)
    save(OUT/'candidate_summary.json',summary)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');ap=argparse.ArgumentParser();ap.add_argument('--clips',nargs='*');args=ap.parse_args();candidates(args.clips)
