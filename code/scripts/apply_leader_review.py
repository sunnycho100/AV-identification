"""Apply version-checked user review without changing the automatic baseline.

Manual boxes are propagated as image candidates for diagnosis, not promoted to
metric 3D truth. Human identity/relation labels and metric availability remain
separate. Run with the existing .venv-bev environment.
"""
import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from build_field_core_dataset import ROOT, read, save, inside_image, interval_groups
from track_selected_leader import track_direction
from trial_manual_leader import iou


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clip_data(pack, clip):
    text = (pack / 'data' / f'{clip}.js').read_text(encoding='utf-8')
    return json.loads(text.removeprefix('window.registerClip(').removesuffix(';').removesuffix(')'))


def image_seed_candidates(record, role, pack, out):
    """Retain image evidence and possible MOT associations; never auto-label."""
    seed = record[role]
    folder = ROOT / 'trajectory/data/camera-data' / record['clip'] / 'frames_all'
    back, bstop = track_direction(folder, seed['frame'], seed['bbox'], record['start_frame'])
    forward, fstop = track_direction(folder, seed['frame'], seed['bbox'], record['end_frame'])
    rows = sorted(back + [dict(frame=seed['frame'], bbox=seed['bbox'],
                              source='user_manual_seed', image_correlation=1.)] + forward,
                  key=lambda r: r['frame'])
    data = {f['frame']: f for f in clip_data(pack, record['clip'])['frames']}
    votes = Counter()
    for row in rows:
        candidates = sorted([dict(track_id=b['id'], iou=iou(row['bbox'], b['bbox']))
                             for b in data[row['frame']]['boxes']
                             if b['kind'] == 'track' and b['observed']],
                            key=lambda b: b['iou'], reverse=True)
        row['t_s'] = data[row['frame']]['t_s']
        row['top_matches'] = candidates[:2]
        row['matched_track'] = None
        if candidates and candidates[0]['iou'] >= .35 and (
                len(candidates) == 1 or candidates[0]['iou'] - candidates[1]['iou'] >= .15):
            row['matched_track'] = candidates[0]['track_id']
            votes[row['matched_track']] += 1
    dest = out / 'image_candidates' / record['clip'] / role
    summary = dict(clip=record['clip'], role=role, seed=seed,
                   image_frames=len(rows), start_frame=rows[0]['frame'], end_frame=rows[-1]['frame'],
                   backward_stop=bstop, forward_stop=fstop, track_votes=dict(votes),
                   metric_status='pending_3d_association',
                   scope='User confirmed a pixel seed; propagated boxes and MOT matches are automatic candidates.')
    save(dest / 'image_track.json', dict(summary, rows=rows))
    chosen = sorted(set(np.linspace(0, len(rows)-1, 6).astype(int).tolist()))
    tiles = []
    for i in chosen:
        row = rows[i]
        im = cv2.imread(str(folder / f'{row["frame"]:03d}.jpg'))
        a, b, c, d = np.round(row['bbox']).astype(int)
        cv2.rectangle(im, (a,b), (c,d), (255,255,0), 3)
        tile = cv2.resize(im, (768,432))
        tile = cv2.copyMakeBorder(tile, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(tile, f'{record["clip"]} {role} f{row["frame"]} | image candidate',
                    (8,22), 0, .58, (0,0,0), 1)
        tiles.append(tile)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(dest / 'review.jpg'), np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/field_review_application.json')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cfg = read(ROOT / args.config)
    source = ROOT / cfg['decisions']
    document = read(source)
    source_hash = sha256(source)
    pack = ROOT / cfg['review_pack']
    sys.path.insert(0, str(ROOT / 'review_web'))
    import server
    server.ROOT, server.PACK = ROOT, pack
    assert server.validate_document(document) == len(document['records'])
    manifest = read(pack / 'manifest.json')
    cases = {c['clip']: c for c in manifest['cases']}
    records = document['records']
    assert len({r['clip'] for r in records}) == len(records), 'Overlapping or multiple intervals need explicit reconciliation'
    assert all(r['reviewer'] == 'user' and r['tag'] == cases[r['clip']]['tag'] for r in records)
    assert {r['clip'] for r in records} == {c['clip'] for c in cases.values() if c['core']}
    overrides = {(v['clip'], v['role']): v for v in cfg['visibility_overrides']}
    for v in overrides.values():
        assert next(r for r in records if r['clip'] == v['clip'])['note'] == v['note_exact']
    out = ROOT / 'experiments' / cfg['experiment']
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / 'decisions_source.json'
    if snapshot.exists():
        assert sha256(snapshot) == source_hash, 'New decision version requires a new experiment directory'
    else:
        shutil.copy2(source, snapshot)
    save(out / 'application_config.json', cfg)
    reports, provenance, pending, semantics = [], [], [], []
    target_cols = ['frame','t_s','tracker_t_s','x','y','z','yaw','vx','vy','vz','speed_mps','x_det','y_det',
                   'track_id','record_id','leader_state','identity_source','is_imputed']
    pair_cols = ['frame','t_s','target_id','leader_id','target_x_m','target_y_m','leader_x_m','leader_y_m',
                 'follower_speed_mps','leader_speed_mps','net_gap_m','direction','token','record_id',
                 'relation_source','is_imputed']
    for r in records:
        c = cases[r['clip']]
        td = ROOT / 'trajectory/outputs/trajectories' / f'{r["clip"]}_{r["tag"]}'
        dd = ROOT / 'trajectory/outputs/object_detection/camera-data' / f'{r["clip"]}_{r["tag"]}'
        video = (pack / c['video_path']).resolve()
        for path, expected in [(video, r['source_video_sha256']), (td / 'tracks.json', r['tracks_sha256'])]:
            actual = sha256(path)
            assert actual == expected, f'Changed source: {path}'
            provenance.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=actual))
        for path in [dd/'calibration_used.json', pack/'data'/f'{r["clip"]}_dimensions.json',
                     ROOT/'trajectory/data/camera-data'/r['clip']/'frame_times.json']:
            provenance.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256(path)))
        tracks = read(td / 'tracks.json')['tracks']
        states = {k: {s['frame']: s for s in ss} for k, ss in tracks.items()}
        dims = read(pack / 'data' / f'{r["clip"]}_dimensions.json')
        cal = read(dd / 'calibration_used.json')
        K, M = np.array(cal['K']), np.array(cal['lidar2cam'])
        times = {f['frame']: f['t_s'] for f in read(ROOT / 'trajectory/data/camera-data' / r['clip'] / 'frame_times.json')['frames']}
        ids = {role: (r[role]['id'] if r.get(role) and r[role]['type'] == 'track' else None)
               for role in ('target','leader')}
        for role in ('target', 'leader'):
            if r.get(role) and r[role]['type'] != 'track':
                pending.append(image_seed_candidates(r, role, pack, out))
        direction = {}
        if ids['target']:
            assert ids['target'] in tracks
            direction = {int(v.frame): int(v.direction) for v in
                         pd.read_csv(td / f'vehicle_{ids["target"]}.csv').itertuples()}
        own, lead, pairs, audit = [], [], [], []
        for f in range(r['start_frame'], r['end_frame'] + 1):
            relation = server.LEADER_STATES[r['decision']]
            if (r['clip'], 'leader') in overrides and f > overrides[r['clip'], 'leader']['last_visible_frame']:
                relation = 'not_visible'
            sample = dict(clip=r['clip'], record_id=r['record_id'], frame=f, t_s=times[f],
                          target_id=ids['target'], leader_id=ids['leader'] if relation == 'present' else None,
                          leader_state=relation, target_visible_by_review=not (
                              (r['clip'], 'target') in overrides and f > overrides[r['clip'], 'target']['last_visible_frame']))
            semantics.append(sample)
            reasons, selected = [], {}
            for role in ('target', 'leader'):
                k = ids[role]
                if role == 'leader' and relation != 'present':
                    reasons.append('leader_' + relation)
                    continue
                if (r['clip'], role) in overrides and f > overrides[r['clip'], role]['last_visible_frame']:
                    reasons.append(role + '_left_image_by_user_review')
                    continue
                s = states.get(k, {}).get(f)
                if k is None:
                    reasons.append(role + '_manual_seed_needs_3d')
                elif s is None:
                    reasons.append(role + '_state_missing')
                elif k not in dims:
                    reasons.append(role + '_dimensions_missing')
                elif 'x_det' not in s:
                    reasons.append(role + '_detection_gap')
                elif not inside_image(s, dims[k], K, M, 1920, 1080):
                    reasons.append(role + '_outside_image')
                elif not all(np.isfinite(s[x]) for x in ('x','y','z','vx','vy','yaw')):
                    reasons.append(role + '_nonfinite_state')
                else:
                    selected[role] = s
                    value = dict(s, t_s=times[f], tracker_t_s=s['t_s'], track_id=k,
                                 record_id=r['record_id'], leader_state=relation,
                                 identity_source='user_selected_track', is_imputed=False)
                    (own if role == 'target' else lead).append(value)
            if len(selected) == 2:
                fs, ls = selected['target'], selected['leader']
                sign = direction.get(f, 0)
                gap = (ls['x'] - fs['x']) * sign - (dims[ids['target']]['length_m'] + dims[ids['leader']]['length_m']) / 2
                if sign not in (-1,1):
                    reasons.append('unknown_longitudinal_direction')
                if ls['vx'] * sign < 0:
                    reasons.append('opposite_estimated_direction')
                if not np.isfinite(gap) or gap <= 0:
                    reasons.append('nonpositive_estimated_gap')
                if not reasons:
                    pairs.append(dict(frame=f,t_s=times[f],target_id=ids['target'],leader_id=ids['leader'],
                        target_x_m=fs['x'],target_y_m=fs['y'],leader_x_m=ls['x'],leader_y_m=ls['y'],
                        follower_speed_mps=abs(fs['vx']),leader_speed_mps=abs(ls['vx']),net_gap_m=float(gap),
                        direction=sign,token=f'{r["record_id"]}:{sign}',record_id=r['record_id'],
                        relation_source='user_confirmed_pair',is_imputed=False))
            audit.append(dict(sample, target_usable='target' in selected, leader_usable='leader' in selected,
                              pair_usable=not reasons, reason=';'.join(reasons)))
        dest = out / 'clips' / r['clip']
        dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(own).reindex(columns=target_cols).to_csv(dest / 'target_trajectory.csv', index=False)
        pd.DataFrame(lead).reindex(columns=target_cols).to_csv(dest / 'leader_trajectory.csv', index=False)
        pd.DataFrame(pairs, columns=pair_cols).to_csv(dest / 'following_pairs.csv', index=False)
        pd.DataFrame(audit).to_csv(dest / 'frame_audit.csv', index=False)
        groups = interval_groups(pairs, cfg['interval_gap_s'])
        intervals = [dict(start_frame=g[0]['frame'],end_frame=g[-1]['frame'],start_s=g[0]['t_s'],
                          end_s=g[-1]['t_s'],span_s=g[-1]['t_s']-g[0]['t_s'],observations=len(g)) for g in groups]
        save(dest / 'intervals.json', intervals)
        save(dest / 'review.json', dict(record=r, visibility_overrides=[v for v in overrides.values() if v['clip']==r['clip']]))
        report = dict(clip=r['clip'],site=c['site'],view=c['view'],decision=r['decision'],
                      start_frame=r['start_frame'],end_frame=r['end_frame'],target_id=ids['target'],leader_id=ids['leader'],
                      target_points=len(own),leader_points=len(lead),paired_points=len(pairs),
                      longest_pair_span_s=max([x['span_s'] for x in intervals], default=0),
                      pending_3d_roles=';'.join(role for role in ('target','leader') if r.get(role) and ids[role] is None),
                      reason_counts=dict(Counter(reason for a in audit for reason in a['reason'].split(';') if reason)))
        reports.append(report)
        print(r['clip'], r['decision'], 'self',len(own),'pair',len(pairs),'pending',report['pending_3d_roles'],flush=True)
    assert sha256(source) == source_hash
    for r in reports:
        if r['decision'] != 'confirm_pair':
            assert r['paired_points'] == 0
    assert not any(s['clip']=='HV_V_EW_3' and s['frame']>274 and s['leader_state']=='present' for s in semantics)
    save(out / 'reviewed_intervals.json', dict(schema='reviewed-field-intervals/v1', source_sha256=source_hash,
                                              records=records,visibility_overrides=cfg['visibility_overrides'],
                                              outside_reviewed_intervals='unreviewed'))
    save(out / 'input_provenance.json', dict(decisions_sha256=source_hash,files=provenance))
    save(out / 'pending_3d_seeds.json', pending)
    pd.DataFrame([{k:v for k,v in r.items() if k!='reason_counts'} for r in reports]).to_csv(out/'recording_results.csv', index=False)
    pd.DataFrame(semantics).to_csv(out / 'reviewed_frame_states.csv', index=False)
    summary = dict(records=len(records),core_clips=43,supplementary_clips_unreviewed=17,
                   decisions=dict(Counter(r['decision'] for r in records)),
                   by_view={site:dict(Counter(r['decision'] for r in reports if r['site']==site)) for site in sorted({r['site'] for r in reports})},
                   target_metric_clips=sum(r['target_points']>0 for r in reports),
                   paired_metric_clips=sum(r['paired_points']>0 for r in reports),
                   target_points=sum(r['target_points'] for r in reports), paired_points=sum(r['paired_points'] for r in reports),
                   manual_seed_roles=dict(Counter(r['role'] for r in pending)),reports=reports,
                   decisions_sha256=source_hash,source_hash_checks=len(provenance),
                   no_new_training=True, no_new_imputation=True,
                   superseded_leader_claims=[dict(clip='HV_S_EW_5',rejected_leader_id='39',source='user_no_leader')],
                   limits='Identity/relation review is not 3D ground truth. Manual image seeds remain pending metric association. Only explicit review intervals are applied; missing leader inputs are null, never zero-filled.')
    save(out / 'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
