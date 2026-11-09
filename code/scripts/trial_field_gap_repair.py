"""Bounded internal-gap interpolation on frozen survey outputs.

This is a trajectory-continuity diagnostic, not a detector/classifier retest.
No cross-ID joins, extrapolation, coordinate registration, or observed-point edits.
"""
from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd

from field_full_survey import ROOT, TR, OUT as SURVEY, CONFIG, read, save, sha

OUT = ROOT / 'experiments/field_gap_repair_20261101'
DURATIONS = [2, 4, 6, 8, 10]
SETTINGS = dict(max_gap_s=[0.5, 1.0], hz=10, stride_s=1,
                max_imputed_window_fraction=0.25,
                acceleration_gate_mps2=8, velocity_noise_mps=1,
                position_noise_m=1, lateral_noise_m=0.5,
                lateral_speed_gate_mps=1.5)
REFERENCES = [
    dict(title='ByteTrack: Multi-Object Tracking by Associating Every Detection Box',
         paper='https://arxiv.org/abs/2110.06864',
         code='https://github.com/FoundationVision/ByteTrack/blob/main/tools/interpolation.py',
         use='Reference for bounded linear interpolation of internal gaps in a track; not a reproduction of ByteTrack.'),
    dict(title='StrongSORT: Make DeepSORT Great Again',
         paper='https://arxiv.org/abs/2202.13514',
         code='https://github.com/dyhBUPT/StrongSORT/blob/master/GSI.py',
         use='GSI uses Gaussian process regression; AFLink addresses missing association separately. GSI not implemented in this trial.'),
    dict(title='Automatic vehicle trajectory data reconstruction at scale',
         paper='https://arxiv.org/abs/2212.07907',
         doi='https://doi.org/10.1016/j.trc.2024.104520',
         code='https://github.com/I24-MOTION/I24-postprocessing-lite',
         use='Traffic-specific fragment association and constrained rectification. Existing local stitcher is an adaptation, not the full published algorithm.')]


def same_pair(a, b):
    return all(a[k] == b[k] for k in ['target_id', 'leader_id', 'lane', 'direction'])


def linear_fill(a, b, frame_times, half_length, tracks=None):
    """Only fill internal timestamps in the two cars' unchanged shared coordinates."""
    dt = b['t_s'] - a['t_s']
    rows = []
    for f, t in frame_times:
        if not (a['frame'] < f < b['frame'] and a['t_s'] < t < b['t_s']):
            continue
        w = (t - a['t_s']) / dt
        r = dict(a, frame=int(f), t_s=float(t), is_imputed=True,
                 gap_left_frame=int(a['frame']), gap_right_frame=int(b['frame']))
        for key in ['target_x_m', 'target_y_m', 'leader_x_m', 'leader_y_m']:
            r[key] = float((1 - w) * a[key] + w * b[key])
        # Linear-position interpolation assumes constant speed within the hole.
        # Original measured/smoothed positions and velocities remain untouched.
        r['follower_speed_mps'] = abs((b['target_x_m'] - a['target_x_m']) / dt)
        r['leader_speed_mps'] = abs((b['leader_x_m'] - a['leader_x_m']) / dt)
        # A missing pair need not mean both individual cars were unobserved.
        # Preserve all detection-supported states inside the gap as well.
        for role, oid, speed in [('target', a['target_id'], 'follower_speed_mps'),
                                  ('leader', a['leader_id'], 'leader_speed_mps')]:
            state = (tracks or {}).get(oid, {}).get(f)
            measured = state is not None and 'x_det' in state
            r[role + '_position_imputed'] = not measured
            if measured:
                r[role + '_x_m'], r[role + '_y_m'] = state['x'], state['y']
                r[speed] = abs(state['vx'])
            else:
                anchors = [(a['t_s'], a[role + '_x_m'], a[role + '_y_m']),
                           (b['t_s'], b[role + '_x_m'], b[role + '_y_m'])]
                anchors += [(s['t_s'], s['x'], s['y'])
                            for s in (tracks or {}).get(oid, {}).values()
                            if 'x_det' in s and a['t_s'] < s['t_s'] < b['t_s']]
                left = max((q for q in anchors if q[0] < t), key=lambda q: q[0])
                right = min((q for q in anchors if q[0] > t), key=lambda q: q[0])
                local_dt = right[0] - left[0]; local_w = (t - left[0]) / local_dt
                r[role + '_x_m'] = (1 - local_w) * left[1] + local_w * right[1]
                r[role + '_y_m'] = (1 - local_w) * left[2] + local_w * right[2]
                r[speed] = abs((right[1] - left[1]) / local_dt)
                r[role + '_left_anchor_s'], r[role + '_right_anchor_s'] = left[0], right[0]
        r['net_gap_m'] = (r['leader_x_m'] - r['target_x_m']) * r['direction'] - half_length
        rows.append(r)
    return rows


def reject_bridge(a, b, table, audit, max_gap):
    dt = b['t_s'] - a['t_s']
    if dt <= 0 or dt > max_gap + 1e-7:
        return 'gap_exceeds_limit'
    if not same_pair(a, b):
        return 'endpoint_id_lane_or_direction_changed'
    middle = table[(table.frame > a['frame']) & (table.frame < b['frame'])]
    known = middle.lead_id.dropna().astype(str)
    known = known[~known.isin(['', '-1', '-1.0', 'None', 'nan'])]
    if any(k != str(a['leader_id']) for k in known):
        return 'different_leader_observed_inside_gap'
    if len(middle) and ((middle.lane != a['lane']).any() or (middle.direction != a['direction']).any()):
        return 'lane_or_direction_changed_inside_gap'
    checks = audit[(audit.frame > a['frame']) & (audit.frame < b['frame'])]
    reasons = ';'.join(checks.reason.fillna(''))
    if 'outside_image' in reasons:
        return 'known_outside_image'
    if 'nonpositive_net_gap' in reasons:
        return 'nonpositive_gap_inside_interval'
    for prefix, vname in [('target', 'follower_speed_mps'), ('leader', 'leader_speed_mps')]:
        v0, v1 = a[vname], b[vname]
        displacement = (b[prefix + '_x_m'] - a[prefix + '_x_m']) * a['direction']
        if displacement < -1e-6:
            return 'reversed_displacement'
        if abs(v1 - v0) > SETTINGS['acceleration_gate_mps2'] * dt + SETTINGS['velocity_noise_mps']:
            return 'endpoint_speed_inconsistent'
        if abs(displacement - .5 * (v0 + v1) * dt) > SETTINGS['position_noise_m'] + .5 * SETTINGS['acceleration_gate_mps2'] * dt ** 2:
            return 'position_velocity_inconsistent'
        if abs(b[prefix + '_y_m'] - a[prefix + '_y_m']) > SETTINGS['lateral_noise_m'] + SETTINGS['lateral_speed_gate_mps'] * dt:
            return 'lateral_jump'
    return None


def rebuild(records, table, audit, times, dims, max_gap, tracks=None):
    if not records:
        return [], [], []
    groups = [[dict(records[0], is_imputed=False)]]
    gap_audit, holes = [], []
    for a, b in zip(records, records[1:]):
        dt = b['t_s'] - a['t_s']
        original_connection = a['token'] == b['token'] and dt <= .2 + 1e-7
        if original_connection:
            # Record an existing small hole, even though the old exporter allowed it.
            if b['frame'] - a['frame'] > 1:
                holes.append((a['t_s'], b['t_s']))
            groups[-1].append(dict(b, is_imputed=False))
            continue
        why = reject_bridge(a, b, table, audit, max_gap) if max_gap else 'original_segment_boundary'
        decision = dict(left_frame=int(a['frame']), right_frame=int(b['frame']),
                        start_s=a['t_s'], end_s=b['t_s'], anchor_gap_s=dt,
                        target_id=a['target_id'], left_leader=a['leader_id'],
                        right_leader=b['leader_id'], accepted=why is None,
                        reason=why or 'bounded_internal_interpolation')
        if why is None:
            half_length = (dims[a['target_id']]['length_m'] + dims[a['leader_id']]['length_m']) / 2
            filled = linear_fill(a, b, times, half_length, tracks)
            if any(r['net_gap_m'] <= 0 for r in filled):
                why = 'reconstructed_nonpositive_gap'
                decision.update(accepted=False, reason=why)
            else:
                groups[-1].extend(filled)
                groups[-1].append(dict(b, is_imputed=False))
                holes.append((a['t_s'], b['t_s']))
                decision['filled_native_points'] = len(filled)
                decision['target_position_estimates'] = sum(r['target_position_imputed'] for r in filled)
                decision['leader_position_estimates'] = sum(r['leader_position_imputed'] for r in filled)
        if why is not None:
            groups.append([dict(b, is_imputed=False)])
        gap_audit.append(decision)
    return groups, gap_audit, holes


def window_metadata(groups, holes, rounded_start=False, original_starts=()):
    windows = []
    for gi, g in enumerate(groups):
        start, end = g[0]['t_s'], g[-1]['t_s']
        if rounded_start:
            start = np.ceil((start - 1e-8) * 10) / 10
        for duration in DURATIONS:
            # N samples at 10 Hz span (N-1)/10 seconds, matching existing models.
            span = (duration * 10 - 1) / 10
            starts = set(np.round(np.arange(start, end - span + 1e-7, 1), 7))
            # A merge must not shift the sampling grid and discard an existing
            # valid window. Retain the original segment grids inside each group.
            for anchor in original_starts:
                if start <= anchor <= end:
                    starts.update(np.round(np.arange(anchor, end - span + 1e-7, 1), 7))
            for t0 in sorted(starts):
                q = t0 + np.arange(duration * 10) / 10
                imputed = np.zeros(len(q), bool)
                for left, right in holes:
                    imputed |= (q > left + 1e-7) & (q < right - 1e-7)
                fraction = float(imputed.mean())
                windows.append(dict(group=gi, duration_s=duration, start_s=float(t0),
                                    end_s=float(q[-1]), imputed_fraction=fraction,
                                    accepted=fraction <= SETTINGS['max_imputed_window_fraction'],
                                    target_id=g[0]['target_id'], leader_id=g[0]['leader_id']))
    return windows


def mask_check(clip, observed):
    """Hide raw detections before interpolation. Compare with hidden raw points.

    This measures consistency with a noisy detector, not physical ground truth.
    No smoothed state/velocity is used to predict hidden observations.
    """
    parts = []
    for s in observed:
        if not parts or s['t_s'] - parts[-1][-1]['t_s'] > .1:
            parts.append([])
        parts[-1].append(s)
    if not parts:
        return []
    part = max(parts, key=lambda g: g[-1]['t_s'] - g[0]['t_s'])
    t = np.array([s['t_s'] for s in part])
    xy = np.array([[s['x_det'], s['y_det']] for s in part])
    results = []
    for duration in [.3, .5, 1.]:
        if t[-1] - t[0] < duration + 1:
            continue
        left = np.searchsorted(t, (t[0] + t[-1] - duration) / 2)
        right = np.searchsorted(t, t[left] + duration)
        if right >= len(t) or right <= left + 1:
            continue
        hidden = np.arange(left + 1, right)
        w = (t[hidden] - t[left]) / (t[right] - t[left])
        prediction = (1 - w[:, None]) * xy[left] + w[:, None] * xy[right]
        err = np.linalg.norm(prediction - xy[hidden], axis=1)
        results.append(dict(clip=clip, requested_anchor_gap_s=duration,
                            anchor_gap_s=float(t[right] - t[left]), hidden_points=len(hidden),
                            mean_error_to_hidden_detector_m=float(err.mean()),
                            median_error_to_hidden_detector_m=float(np.median(err)),
                            max_error_to_hidden_detector_m=float(err.max())))
    return results


def selfcheck():
    a = dict(frame=0, t_s=0., target_id='a', leader_id='b', lane=1, direction=1,
             target_x_m=0., target_y_m=0., leader_x_m=20., leader_y_m=0.,
             follower_speed_mps=10., leader_speed_mps=10., net_gap_m=16., token=1)
    b = dict(a, frame=30, t_s=1., target_x_m=10., leader_x_m=30.)
    fill = linear_fill(a, b, [(0, 0.), (15, .5), (30, 1.)], 4.)
    assert len(fill) == 1 and fill[0]['net_gap_m'] == 16 and fill[0]['target_x_m'] == 5
    known = {'a': {15: dict(x=6., y=.1, vx=9., x_det=6.2)}}
    preserved = linear_fill(a, b, [(15, .5)], 4., known)[0]
    assert preserved['target_x_m'] == 6. and preserved['follower_speed_mps'] == 9.
    assert not preserved['target_position_imputed'] and preserved['leader_position_imputed']
    table = pd.DataFrame([dict(frame=15, lead_id='b', lane=1, direction=1)])
    audit = pd.DataFrame([dict(frame=15, reason='leader_detection_gap')])
    assert reject_bridge(a, b, table, audit, 1.) is None
    assert reject_bridge(a, b, table, audit, .5) == 'gap_exceeds_limit'
    assert reject_bridge(a, dict(b, leader_id='c'), table, audit, 1.) is not None
    assert reject_bridge(a, b, table.assign(lead_id='c'), audit, 1.) is not None
    assert reject_bridge(a, b, table, audit.assign(reason='leader_outside_image'), 1.) is not None
    assert not linear_fill(a, b, [(-1, -.1), (31, 1.1)], 4.)
    short = [dict(a, t_s=19.519), dict(b, t_s=23.423)]
    merged = [dict(a, t_s=18.785), dict(b, t_s=23.423)]
    before = window_metadata([short], [])
    after = window_metadata([merged], [], original_starts=[19.519])
    assert any(w['duration_s'] == 4 for w in before)
    assert any(w['duration_s'] == 4 and w['start_s'] == 19.519 for w in after)


def source_review_sheets(cfg, gaps):
    import cv2
    from visualize_field_baseline import draw
    folder = OUT / 'source_reviews'
    folder.mkdir(exist_ok=True)
    for gap in gaps:
        if gap['variant'] != 'linear_1p0s' or not gap['accepted']:
            continue
        clip = gap['clip']; tag = cfg['models'][gap['model']]['tag']
        tracks = read(TR / 'outputs/trajectories' / f'{clip}_{tag}' / 'tracks.json')['tracks']
        dims = read(SURVEY / 'pairs' / f'{clip}_{tag}' / 'dimensions.json')
        cal = read(TR / 'outputs/object_detection/camera-data' / f'{clip}_{tag}' / 'calibration_used.json')
        K, M = np.array(cal['K']), np.array(cal['lidar2cam'])
        df = pd.read_csv(OUT / 'linear_1p0s' / clip / 'following_pairs.csv')
        frames = [gap['left_frame'], round((gap['left_frame'] + gap['right_frame']) / 2), gap['right_frame']]
        tiles = []
        for j, frame in enumerate(frames):
            r = df[df.frame == frame].iloc[0]
            im = cv2.imread(str(TR / 'data/camera-data' / clip / 'frames_all' / f'{frame:03d}.jpg'))
            for role, oid in [('target', gap['target_id']), ('leader', gap['left_leader'])]:
                st = min(tracks[oid], key=lambda s: abs(s['frame'] - frame)); dim = dims[oid]
                # Only x/y come from the trial. Orientation and height are existing
                # nearest track states for visualization, not new yaw estimates.
                box = dict(st, x=float(r[role + '_x_m']), y=float(r[role + '_y_m']),
                           l=dim['length_m'], w=dim['width_m'], h=dim['height_m'])
                draw(im, box, K, M, (0, 170, 255) if j == 1 else (0, 240, 240),
                     role + (' estimated' if j == 1 else ' endpoint'))
            tile = np.full((570, 960, 3), 255, np.uint8)
            tile[30:] = cv2.resize(im, (960, 540))
            cv2.putText(tile, f'{clip} f{frame}: ' + ('interpolated' if j == 1 else 'observed pair'),
                        (10, 23), 0, .6, (0, 0, 0), 1)
            tiles.append(tile)
        cv2.imwrite(str(folder / f'{clip}_{frames[0]}_{frames[-1]}.jpg'), np.hstack(tiles))


def main():
    selfcheck()
    cfg = read(CONFIG)
    OUT.mkdir(exist_ok=True)
    protocol = dict(settings=SETTINGS, durations_s=DURATIONS, references=REFERENCES,
                    selection='All 60 videos; fixed view-based model selection from full survey, no label-based tuning.',
                    constraints=['same endpoint target and leader IDs, lane and direction',
                                 'no known alternate leader, out-of-image observation, or negative gap inside',
                                 'two observed pair endpoints, bounded gap, endpoint motion checks',
                                 'linear x/y interpolation in unchanged common coordinates; no extrapolation',
                                 'observed points preserved; imputed positions separately flagged',
                                 'thresholds are local exploratory choices, not literature-prescribed values'],
                    classifier='Not run: assess continuity before treating reconstructed inputs as behavior evidence.')
    config_path = ROOT / 'configs/field_gap_repair.json'
    if config_path.exists():
        assert read(config_path) == protocol
    else:
        save(config_path, protocol)
    save(OUT / 'references.json', REFERENCES)
    manifest = {}
    results, all_gaps, masking = [], [], []
    variants = [('existing_grid', None, True), ('alignment_only', None, False),
                ('linear_0p5s', .5, False), ('linear_1p0s', 1., False)]
    figure_data = []
    for row in cfg['clips']:
        clip = row['clip']
        key = next((k for k in row['models'] if k != 'original'), 'original')
        tag = cfg['models'][key]['tag']
        run = read(SURVEY / 'runs' / f'{clip}_{key}' / 'evaluation.json')
        td = TR / 'outputs/trajectories' / f'{clip}_{tag}'
        src = SURVEY / 'pairs' / f'{clip}_{tag}'
        records, times, dims, states = [], [], {}, {}
        table = pd.DataFrame(); audit = pd.DataFrame()
        if run.get('target_observations', 0):
            target_path = src / 'target_trajectory.csv'
            target = pd.read_csv(target_path)
            masking.extend(mask_check(clip, target.sort_values('t_s').to_dict('records')))
        if run.get('following_observations', 0):
            inputs = [src / 'following_pairs.csv', src / 'pair_audit.csv', src / 'dimensions.json',
                      td / f'vehicle_{run["identified_target"]}.csv']
            for p in inputs:
                manifest[str(p.relative_to(ROOT))] = sha(p)
            records = pd.read_csv(inputs[0], dtype={'target_id': str, 'leader_id': str}).sort_values('t_s').to_dict('records')
            audit = pd.read_csv(inputs[1], dtype={'target_id': str, 'leader_id': str})
            dims = read(inputs[2])
            table = pd.read_csv(inputs[3], dtype={'lead_id': str})
            times = [(int(r.frame), float(r.t_s)) for r in table.itertuples()]
            track_path = td / 'tracks.json'
            manifest[str(track_path.relative_to(ROOT))] = sha(track_path)
            states = {oid: {s['frame']: s for s in st} for oid, st in read(track_path)['tracks'].items()}
        original_groups = rebuild(records, table, audit, times, dims, None)[0]
        original_starts = [g[0]['t_s'] for g in original_groups]
        for name, maximum, rounded in variants:
            groups, gaps, holes = rebuild(records, table, audit, times, dims, maximum, states)
            windows = window_metadata(groups, holes, rounded, original_starts if maximum else ())
            # Preserve original counts for the two controls; isolate fraction gating.
            if maximum is None:
                for w in windows:
                    w['accepted'] = True
            folder = OUT / name / clip
            folder.mkdir(parents=True, exist_ok=True)
            flattened = [dict(r, repaired_group=i) for i, g in enumerate(groups) for r in g]
            if flattened:
                df = pd.DataFrame(flattened)
                df.to_csv(folder / 'following_pairs.csv', index=False)
                # Input observations must survive byte-level numerical round trips.
                original = df.loc[~df.is_imputed]
                for col in ['t_s', 'target_x_m', 'target_y_m', 'leader_x_m', 'leader_y_m', 'net_gap_m']:
                    np.testing.assert_allclose(original[col], [r[col] for r in records], rtol=0, atol=0)
                for r in flattened:
                    expected = (r['leader_x_m'] - r['target_x_m']) * r['direction'] - (dims[r['target_id']]['length_m'] + dims[r['leader_id']]['length_m']) / 2
                    assert abs(expected - r['net_gap_m']) < 1e-6
                    if r['is_imputed']:
                        for role, oid in [('target', r['target_id']), ('leader', r['leader_id'])]:
                            st = states.get(oid, {}).get(r['frame'])
                            if st is not None and 'x_det' in st:
                                assert r[role + '_x_m'] == st['x'] and r[role + '_y_m'] == st['y']
            save(folder / 'windows.json', windows)
            accepted = [w for w in windows if w['accepted']]
            counts = {str(d): sum(w['duration_s'] == d for w in accepted) for d in DURATIONS}
            if name == 'existing_grid':
                for d in [2, 4]:
                    assert counts[str(d)] == run.get('windows_by_duration', {}).get(str(float(d)), 0)
            res = dict(clip=clip, model=key, variant=name, core=row['core'],
                       pair_points_before=len(records), imputed_points=sum(bool(r['is_imputed']) for r in flattened),
                       accepted_bridges=sum(g['accepted'] for g in gaps),
                       longest_pair_span_s=max((g[-1]['t_s'] - g[0]['t_s'] for g in groups), default=0),
                       windows=counts, windows_rejected_for_imputed_fraction=sum(not w['accepted'] for w in windows))
            results.append(res)
            all_gaps.extend(dict(g, clip=clip, model=key, variant=name) for g in gaps)
            if name == 'linear_1p0s' and res['accepted_bridges']:
                figure_data.append((clip, records, flattened, gaps))
    save(OUT / 'input_manifest.json', manifest)
    for path, digest in manifest.items():
        assert sha(ROOT / path) == digest
    summary = []
    for name, _, _ in variants:
        for scope in ['all', 'core']:
            rr = [r for r in results if r['variant'] == name and (scope == 'all' or r['core'])]
            summary.append(dict(variant=name, scope=scope, recordings=len(rr),
                                repaired_recordings=sum(r['accepted_bridges'] > 0 for r in rr),
                                accepted_bridges=sum(r['accepted_bridges'] for r in rr),
                                imputed_points=sum(r['imputed_points'] for r in rr),
                                **{f'window_{d}s_recordings': sum(r['windows'][str(d)] > 0 for r in rr) for d in DURATIONS}))
    save(OUT / 'summary.json', dict(groups=summary, recordings=results,
                                  note='Coverage only. No independent identity, metric, velocity or leader ground truth.'))
    pd.DataFrame(summary).to_csv(OUT / 'coverage.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_gaps).to_csv(OUT / 'gap_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(masking).to_csv(OUT / 'masked_gap_checks.csv', index=False, encoding='utf-8-sig')
    mask_summary = []
    for duration in [.3, .5, 1.]:
        group = [r for r in masking if r['requested_anchor_gap_s'] == duration]
        mask_summary.append(dict(anchor_gap_s=duration, recordings=len(group),
                                 median_recording_error_m=float(np.median([r['mean_error_to_hidden_detector_m'] for r in group])) if group else None))
    save(OUT / 'masked_gap_summary.json', dict(results=mask_summary,
          note='One internal block per eligible recording per duration, hidden before linear prediction. No use of smoothed states. Noisy raw detections are a consistency reference, not physical truth. Durations reuse recordings.'))
    save(OUT / 'validation.json', dict(status='PASS', original_2s_4s_counts_reproduced=True,
          source_files_unchanged=True, observed_points_unchanged=True,
          single_vehicle_observations_inside_pair_gaps_preserved=True,
          common_coordinate_net_gap_checked=True, bounded_internal_gap_selfchecks=True,
          note='Implementation invariants, not ground-truth tracking accuracy.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (OUT / 'figures').mkdir(exist_ok=True)
    # Save every repaired recording, not a best-looking selection.
    for clip, records, repaired, gaps in figure_data:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        obs = pd.DataFrame(records); dense = pd.DataFrame(repaired)
        for ax, col in zip(axes, ['target_x_m', 'leader_x_m', 'net_gap_m']):
            ax.scatter(obs.t_s, obs[col], s=7, label='Existing observed pair points', color='#276aa4')
            imp = dense[dense.is_imputed]
            ax.scatter(imp.t_s, imp[col], s=12, label='Bounded linear fill', color='#d56724')
            for g in gaps:
                if g['accepted']: ax.axvspan(g['start_s'], g['end_s'], color='#d56724', alpha=.1)
            ax.set_ylabel(col); ax.grid(alpha=.2)
        axes[0].legend(fontsize=8); axes[-1].set_xlabel('Source video time (s)')
        fig.suptitle(clip + ': internal-gap trial, not independently validated trajectories')
        fig.savefig(OUT / 'figures' / f'{clip}.png', dpi=140); plt.close(fig)
    source_review_sheets(cfg, all_gaps)
    lines = ['短缺口插值对照 | 2026-11-01', '',
             '沿用全量测试固定视角组合，覆盖全部 60 段；不重跑检测、不训练、不按分类正确率选择参数。',
             '现有 stitch_tracks.py 已做碎片关联，postprocess_tracks.clean 已做双向 RTS 平滑和缺帧状态估计。原分类导出仍要求 x_det，且前车关联变化会切段，补全状态未直接用于分类。',
             '本次新增独立线性插值对照，限定同一自车/前车、同车道和方向、缺口两端均有有效观测；不跨已知其他前车、出画、负间距或异常运动，不外推。保持原观测点和共同坐标不变。',
             '配对缺口内若其中一辆车仍有检测支持，保留该车已有的位置和速度；只对缺失车辆在相邻可用观测之间插值。target_position_imputed / leader_position_imputed 分别标记两车位置是否估计，净间距统一从两车位置重算。',
             '缺口上限指两端观测的时间差。0.5 / 1 秒及 25% 补点比例是本地探索参数，不是文献推荐的通用阈值。原有 <=0.2 秒缺口也计入补点比例。',
             'existing_grid 复现旧切窗；alignment_only 仅取消起点向上取整；linear 两组再加入短缺口连接和每窗最多 25% 插值比例。连接后同时保留原片段的切窗起点，避免网格平移丢掉旧窗口。固定采样 10 Hz，N 点覆盖 (N-1)/10 秒。', '',
             '每组视频覆盖：2 / 4 / 6 / 8 / 10 秒窗口，分母为该组全部视频。']
    for r in summary:
        counts = ' / '.join(str(r[f'window_{d}s_recordings']) for d in DURATIONS)
        lines.append(f"{r['variant']} {r['scope']} n={r['recordings']}: {counts}；连接 {r['accepted_bridges']} 处，涉及 {r['repaired_recordings']} 段。")
    lines += ['', '人工遮挡检验：仅遮掉原检测观测中的内部点，用两端原检测位置插值，再与被遮掉的检测位置比较；不使用预先平滑后的轨迹预测。参考本身有噪声，不是实际定位精度。']
    for r in mask_summary:
        lines.append(f"{r['anchor_gap_s']}s: {r['recordings']} 段，逐段平均位置差的中位数 {r['median_recording_error_m']} m。")
    lines += ['', '这一步只检查连续性与插值可行性，没有重新报告 AV/HV 精度。线性补点会抹去缺口内真实加速度变化；无法纠正检测到的错误身份、尺度误差和错误前车。RTS/GSI 后续比较需同样保留缺失标记与遮挡检验。',
              '此前 HV_V_EW_6 错误尾段仍需单独修正；短缺口插值不会证明整条轨迹可靠。',
              '文件：coverage.csv、gap_audit.csv、summary.json、masked_gap_checks.csv、各组/视频/following_pairs.csv 和 windows.json、figures/。source_reviews/ 提供各新增连接的前、中、后三帧叠加；中点框的位置来自插值，朝向/高度沿用邻近跟踪状态，仅用于复核位置和身份，不是新的姿态估计。',
              '文献：']
    lines += [r['title'] + '\n' + r['paper'] + '\n' + r['code'] for r in REFERENCES]
    lines += ['', '复现：.venv-bev/Scripts/python.exe scripts/trial_field_gap_repair.py']
    (OUT / 'RESULTS.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(coverage=summary, masked_checks=mask_summary), indent=2))


if __name__ == '__main__':
    main()
