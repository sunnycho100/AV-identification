"""Generate visual evidence for earlier anchors and entire proposed MOT IDs."""
import sys
import numpy as np
import cv2
from build_field_core_dataset import ROOT,read,save
from track_review_seeds import adaptive_flow_direction,annotate,contact
from visualize_field_baseline import project


def main():
    cv2.setNumThreads(2);cfg=read(ROOT/'configs/field_review_test.json');out=ROOT/'experiments'/cfg['experiment']
    records={r['clip']:r for r in read(ROOT/cfg['source_review']/'decisions_source.json')['records']}
    for choice in read(ROOT/'configs/field_seed_resolution.json')['records']:
        clip,role=choice['clip'],choice['role'];record=records[clip];folder=ROOT/'trajectory/data/camera-data'/clip/'frames_all'
        dest=out/'seeds'/clip/role
        if choice['method']=='existing_mot':
            td=ROOT/'trajectory/outputs/trajectories'/f'{clip}_{record["tag"]}'
            ss=read(td/'tracks.json')['tracks'][choice['track_id']]
            dim=read(ROOT/'experiments/field_leader_review_20261102/data'/f'{clip}_dimensions.json')[choice['track_id']]
            cal=read(ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{record["tag"]}'/'calibration_used.json')
            K,M=np.array(cal['K']),np.array(cal['lidar2cam']);rows=[]
            for s in ss:
                if not record['start_frame']<=s['frame']<=record['end_frame']:continue
                uv=project(dict(s,l=dim['length_m'],w=dim['width_m'],h=dim['height_m']),K,M)
                if uv is not None:rows.append(dict(frame=s['frame'],bbox=np.r_[uv.min(0),uv.max(0)].tolist(),association_label='MOT observed' if 'x_det' in s else 'MOT predicted'))
            contact(rows,folder,dest/'proposed_mot_review.jpg',clip+' ID'+choice['track_id'])
        elif choice.get('image_source')=='reanchor':
            seed=choice['anchor']; back,bs=adaptive_flow_direction(folder,seed,record['start_frame']);forward,fs=adaptive_flow_direction(folder,seed,record['end_frame'])
            rows=sorted(back+[dict(frame=seed['frame'],bbox=seed['bbox'],source='assistant_image_anchor',image_correlation=1.)]+forward,key=lambda v:v['frame'])
            rows=annotate(rows,clip,record['tag'],out)
            save(dest/'reanchor_track.json',dict(clip=clip,role=role,anchor=seed,backward_stop=bs,forward_stop=fs,rows=rows))
            contact(rows,folder,dest/'reanchor_review.jpg',clip)
            print(clip,'image_frames',len(rows),'detections',sum(bool(r['matched_detection']) for r in rows),bs,fs,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
