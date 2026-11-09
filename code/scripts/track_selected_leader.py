"""Image-seeded bidirectional optical flow; associate cached 3D detections.

This creates provisional 2D candidates, never labels them as 3D observations.
No calibration, model weights, original MOT, or classifier inputs are modified.
"""
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from build_field_core_dataset import ROOT, read, save
from visualize_field_baseline import project
from trial_manual_leader import iou

OUT = ROOT / 'experiments/field_leader_review_20261102'


def normalized_patch(gray, bbox):
    x0, y0, x1, y1 = np.round(bbox).astype(int)
    return cv2.resize(gray[y0:y1, x0:x1], (80, 48))


def track_direction(folder, seed_frame, seed_box, stop_frame):
    prev = cv2.imread(str(folder / f'{seed_frame:03d}.jpg'), 0)
    bbox = np.array(seed_box, float)
    reference = normalized_patch(prev, bbox)
    result = []
    direction = 1 if stop_frame > seed_frame else -1
    failure = None
    # Replenish image corners inside the selected car, with forward/backward
    # consistency and robust similarity fitting at each native frame.
    for f in range(seed_frame + direction, stop_frame + direction, direction):
        now = cv2.imread(str(folder / f'{f:03d}.jpg'), 0)
        mask = np.zeros_like(prev)
        x0, y0, x1, y1 = np.round(bbox).astype(int)
        inset = max(2, int(min(x1-x0, y1-y0) * .10))
        mask[y0+inset:y1-inset, x0+inset:x1-inset] = 255
        points = cv2.goodFeaturesToTrack(prev, 80, .015, 3, mask=mask)
        reason = None
        if points is None or len(points) < 6:
            reason = 'too_few_image_points'
        else:
            forward, sf, _ = cv2.calcOpticalFlowPyrLK(prev, now, points, None, winSize=(21,21), maxLevel=3)
            backward, sb, _ = cv2.calcOpticalFlowPyrLK(now, prev, forward, None, winSize=(21,21), maxLevel=3)
            good = (sf.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm(points-backward, axis=2).ravel() < 1.5)
            if good.sum() < 6:
                reason = 'forward_backward_inconsistency'
            else:
                affine, inliers = cv2.estimateAffinePartial2D(points[good], forward[good], method=cv2.RANSAC, ransacReprojThreshold=2)
                if affine is None or inliers.sum() < 6 or inliers.mean() < .55:
                    reason = 'inconsistent_image_motion'
                else:
                    scale = float(np.hypot(affine[0,0], affine[0,1]))
                    angle = float(np.arctan2(affine[1,0], affine[0,0]))
                    center = (bbox[:2] + bbox[2:]) / 2
                    nc = affine[:,:2] @ center + affine[:,2]
                    size = (bbox[2:] - bbox[:2]) * scale
                    new_box = np.r_[nc - size/2, nc + size/2]
                    if not .94 <= scale <= 1.06 or abs(angle) > .10 or np.linalg.norm(nc-center) > .40 * max(size):
                        reason = 'motion_gate'
                    elif (new_box[:2] < 0).any() or new_box[2] >= now.shape[1] or new_box[3] >= now.shape[0] or min(size) < 14:
                        reason = 'outside_or_too_small'
                    else:
                        correlation = float(cv2.matchTemplate(normalized_patch(now,new_box), reference, cv2.TM_CCOEFF_NORMED)[0,0])
                        if correlation < .30:
                            reason = 'appearance_changed'
                        else:
                            result.append(dict(frame=f, bbox=new_box.tolist(), image_correlation=correlation,
                                               flow_points=int(good.sum()), flow_inliers=int(inliers.sum()),
                                               source='image_optical_flow_candidate'))
                            bbox = new_box
        if reason:
            failure = dict(frame=f, reason=reason)
            break
        prev = now
    return result, failure


def main():
    cv2.setNumThreads(2)
    cfg = read(ROOT / 'configs/field_manual_leader_pilot.json')
    clip, tag = cfg['clip'], cfg['tag']
    folder = ROOT / 'trajectory/data/camera-data' / clip / 'frames_all'
    times = {r['frame']: r['t_s'] for r in read(folder.parent/'frame_times.json')['frames']}
    seed = cfg['leader_seed']
    # Continue across the existing leader MOT endpoint, up to the self track's
    # end. A failed image gate terminates that direction without extrapolation.
    backwards, bstop = track_direction(folder, seed['frame'], seed['bbox_xyxy_px'], 94)
    forwards, fstop = track_direction(folder, seed['frame'], seed['bbox_xyxy_px'], 209)
    rows = sorted(backwards + [dict(frame=seed['frame'], bbox=seed['bbox_xyxy_px'], image_correlation=1.,
                                    source='assistant_image_seed')] + forwards, key=lambda r:r['frame'])
    dd = ROOT / 'trajectory/outputs/object_detection/camera-data' / f'{clip}_{tag}'
    cal = read(dd/'calibration_used.json'); K, M = np.array(cal['K']), np.array(cal['lidar2cam'])
    td = ROOT / 'trajectory/outputs/trajectories' / f'{clip}_{tag}'
    tracks = read(td/'tracks.json')['tracks']
    old = {s['frame']:s for s in tracks['39']}
    summary_candidates = []
    for r in rows:
        f = r['frame']; r['t_s'] = times[f]
        candidates = []
        for index, det in enumerate(read(dd/f'{f:03d}_pred.json')):
            if det['class_name'] not in ('car','truck','bus'): continue
            uv = project(det, K, M)
            if uv is None: continue
            overlap = iou(r['bbox'], np.r_[uv.min(0),uv.max(0)])
            if overlap > 0: candidates.append(dict(index=index, iou=overlap, detection=det))
        candidates.sort(key=lambda x:x['iou'],reverse=True)
        accepted = bool(candidates and candidates[0]['iou'] >= .35 and
                        (len(candidates)==1 or candidates[0]['iou']-candidates[1]['iou'] >= .15))
        r['matched_detection'] = candidates[0] if accepted else None
        r['old_leader_detection_supported'] = f in old and 'x_det' in old[f]
        r['review_status'] = 'unverified_2d_candidate'
        if accepted and not r['old_leader_detection_supported']:
            summary_candidates.append(f)
    out = OUT/'targeted_tracking'/clip; out.mkdir(parents=True,exist_ok=True)
    save(out/'image_track.json',dict(clip=clip,tag=tag,seed=seed,rows=rows,
         status='provisional; image evidence only, not metric 3D trajectory',backward_stop=bstop,forward_stop=fstop))
    pd.DataFrame([dict(frame=r['frame'], t_s=r['t_s'], image_correlation=r['image_correlation'],
                       detection_matched=r['matched_detection'] is not None,
                       old_leader_detection_supported=r['old_leader_detection_supported'],
                       matched_class=r['matched_detection']['detection']['class_name'] if r['matched_detection'] else '') for r in rows]).to_csv(out/'association_audit.csv',index=False)
    chosen = sorted(set([rows[0]['frame'],seed['frame'],rows[-1]['frame']] +
                        [f for f in [150,160,170,180,190,200] if any(r['frame']==f for r in rows)]))
    tiles=[]
    for f in chosen:
        r=next(r for r in rows if r['frame']==f)
        im=cv2.imread(str(folder/f'{f:03d}.jpg'));x0,y0,x1,y1=np.round(r['bbox']).astype(int)
        cv2.rectangle(im,(x0,y0),(x1,y1),(255,230,0),3)
        cv2.putText(im,'2D candidate',(x0,max(y0-10,20)),0,.7,(255,230,0),2)
        tile=np.full((580,960,3),255,np.uint8);tile[40:]=cv2.resize(im,(960,540))
        cv2.putText(tile,f'{clip} f{f} | image score {r["image_correlation"]:.2f} | 3D match {r["matched_detection"] is not None}',(8,26),0,.57,(0,0,0),1);tiles.append(tile)
    if len(tiles)%2:tiles.append(np.full_like(tiles[0],255))
    cv2.imwrite(str(out/'tracking_review.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    report=dict(clip=clip,seed_frame=seed['frame'],image_candidate_frames=len(rows),start_frame=rows[0]['frame'],end_frame=rows[-1]['frame'],
                image_span_s=rows[-1]['t_s']-rows[0]['t_s'],matched_3d_detection_frames=sum(r['matched_detection'] is not None for r in rows),
                potential_recovered_detection_frames=summary_candidates,backward_stop=bstop,forward_stop=fstop,
                new_confirmed_3d_points=0,scope='Cached detector associations; optical flow does not invent 3D depth. Requires visual review before re-smoothing and downstream use.')
    save(out/'summary.json',report);print(json.dumps(report))


if __name__=='__main__':main()
