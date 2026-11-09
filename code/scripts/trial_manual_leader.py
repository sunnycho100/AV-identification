"""Map an image-selected leader to cached MOT, retaining common geometry.

Small provisional feasibility trial, not new training or a new visual tracker.
The annotation is separate from the automatic survey and its final metrics.
"""
import argparse
import hashlib
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from build_field_core_dataset import (
    ROOT, read, save, recover_dimensions, inside_image, interval_groups,
)
from visualize_field_baseline import project, draw


def iou(a, b):
    a, b = np.array(a), np.array(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/field_manual_leader_pilot.json')
    args = ap.parse_args()
    cfg = read(ROOT / args.config)
    clip, tag = cfg['clip'], cfg['tag']
    out = ROOT / 'experiments' / cfg['experiment']
    out.mkdir(parents=True, exist_ok=True)
    td = ROOT / 'trajectory/outputs/trajectories' / f'{clip}_{tag}'
    dd = ROOT / 'trajectory/outputs/object_detection/camera-data' / f'{clip}_{tag}_road'
    frames = ROOT / 'trajectory/data/camera-data' / clip / 'frames_all'
    inputs = [ROOT / args.config, td / 'tracks.json', td / 'instrumented.json',
              dd / 'calibration_used.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    identity = read(td / 'instrumented.json')
    assert identity['accepted'] and not identity.get('warnings')
    target = str(identity['vehicle_id'])
    tracks = read(td / 'tracks.json')['tracks']
    states = {k: {s['frame']: s for s in ss} for k, ss in tracks.items()}
    dims = recover_dimensions(tracks, dd)
    cal = read(dd / 'calibration_used.json')
    K, M = np.array(cal['K']), np.array(cal['lidar2cam'])
    seed = cfg['leader_seed']

    def box(k, s):
        d = dims[k]
        return dict(s, l=d['length_m'], w=d['width_m'], h=d['height_m'])

    candidates = []
    for k, ss in states.items():
        s = ss.get(seed['frame'])
        if k == target or k not in dims or s is None or 'x_det' not in s:
            continue
        uv = project(box(k, s), K, M)
        if uv is not None:
            candidates.append(dict(track_id=k, iou=iou(seed['bbox_xyxy_px'], np.r_[uv.min(0), uv.max(0)])))
    candidates.sort(key=lambda r: r['iou'], reverse=True)
    assert candidates and candidates[0]['iou'] >= cfg['minimum_seed_iou']
    assert len(candidates) == 1 or candidates[0]['iou'] - candidates[1]['iou'] >= cfg['minimum_iou_margin']
    leader = candidates[0]['track_id']
    first, last = cfg['interval_frames_inclusive']
    sample = cv2.imread(str(frames / f'{seed["frame"]:03d}.jpg'))
    height, width = sample.shape[:2]
    table = pd.read_csv(td / f'vehicle_{target}.csv', dtype={'lead_id': str}).set_index('frame')
    ltable = pd.read_csv(td / f'vehicle_{leader}.csv').set_index('frame')
    rows, audit, tiles = [], [], []
    for f in range(first, last + 1):
        fs, ls = states[target].get(f), states[leader].get(f)
        reasons = []
        for k, s, role in [(target, fs, 'target'), (leader, ls, 'leader')]:
            if s is None:
                reasons.append(role + '_state_missing')
            else:
                if 'x_det' not in s:
                    reasons.append(role + '_detection_gap')
                if not inside_image(s, dims[k], K, M, width, height):
                    reasons.append(role + '_outside_image')
        gap = None
        if fs is not None and ls is not None:
            assert abs(fs['t_s'] - ls['t_s']) < 1e-7
            direction = int(table.loc[f, 'direction'])
            if ls['vx'] * direction < 0:
                reasons.append('opposite_direction')
            gap = (ls['x'] - fs['x']) * direction - (dims[target]['length_m'] + dims[leader]['length_m']) / 2
            if not np.isfinite(gap) or gap <= 0:
                reasons.append('nonpositive_net_gap')
        audit.append(dict(frame=f, usable=not reasons, reason=';'.join(reasons)))
        if not reasons:
            rows.append(dict(frame=f, t_s=fs['t_s'], target_id=target, leader_id=leader,
                             target_x_m=fs['x'], target_y_m=fs['y'], leader_x_m=ls['x'], leader_y_m=ls['y'],
                             follower_speed_mps=abs(fs['vx']), leader_speed_mps=abs(ls['vx']),
                             net_gap_m=gap, direction=direction, token=1, is_imputed=False,
                             automatic_target_lane=int(table.loc[f, 'lane']),
                             automatic_leader_lane=int(ltable.loc[f, 'lane']),
                             relation_source='provisional_image_selected_leader', use_for_final_evaluation=False))
        if f in cfg['review_frames']:
            im = cv2.imread(str(frames / f'{f:03d}.jpg'))
            for k, s, color, name in [(target, fs, (0, 255, 255), 'SELF'),
                                      (leader, ls, (255, 230, 0), 'SELECTED LEADER')]:
                if s is not None:
                    draw(im, box(k, s), K, M, color, f'{name} {k}' + (' predicted' if 'x_det' not in s else ''))
            if f == seed['frame']:
                x0, y0, x1, y1 = seed['bbox_xyxy_px']
                cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 2)
            tile = np.full((580, 960, 3), 255, np.uint8)
            tile[40:] = cv2.resize(im, (960, 540))
            cv2.putText(tile, f'{clip} f{f} | provisional image selection', (8, 27), 0, .6, (0, 0, 0), 1)
            tiles.append(tile)
    assert rows
    pd.DataFrame(rows).to_csv(out / 'following_pairs.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'pair_audit.csv', index=False)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(out / 'selected_leader_review.jpg'), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)]))
    groups = interval_groups(rows, .2)
    intervals = [dict(start_frame=g[0]['frame'], end_frame=g[-1]['frame'],
                      start_s=g[0]['t_s'], end_s=g[-1]['t_s'], observations=len(g),
                      span_s=g[-1]['t_s'] - g[0]['t_s']) for g in groups]
    baseline = read(ROOT / 'experiments/field_full_survey_20261101/runs' / f'{clip}_adapted_S' / 'evaluation.json')
    report = dict(clip=clip, target_id=target, leader_id=leader, seed_candidates=candidates[:5],
                  review_status=cfg['review_status'], automatic_pair_observations_whole_clip=baseline['following_observations'],
                  selected_pair_observations=len(rows), intervals=intervals,
                  excluded_reasons=dict(Counter(r for a in audit for r in a['reason'].split(';') if r)),
                  median_lateral_difference_m=float(np.median([r['leader_y_m'] - r['target_y_m'] for r in rows])),
                  limits='Cached MOT identity association only. No targeted re-detection, track extension, interpolation, geometry correction, new classifier inference or human ground truth. Existing common coordinates, times and velocities retained. Lane override does not validate the relative geometry.',
                  use_for_final_evaluation=False)
    save(out / 'summary.json', report)
    save(out / 'annotation.json', cfg)
    for r in rows:
        for role, k in [('target', target), ('leader', leader)]:
            s = states[k][r['frame']]
            assert r[f'{role}_x_m'] == s['x'] and r[f'{role}_y_m'] == s['y']
            assert 'x_det' in s
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    save(out / 'validation.json', dict(status='PASS', original_geometry_and_times_preserved=True,
                                     only_detection_supported_observations=True, source_hashes=hashes))
    print(report)


if __name__ == '__main__':
    main()
