"""All-video bounded gap repair, target continuity and frozen classification.

Read-only reuse of the completed 60-video detector/MOT survey. New outputs are
isolated from the 20261101 experiments. Observations and shared geometry stay fixed.
"""
import argparse
from collections import Counter
import numpy as np
import pandas as pd

import trial_field_gap_repair as G
from field_full_survey import ROOT, TR, OUT as SURVEY, CONFIG, read, save, sha
from build_field_core_dataset import inside_image

OUT = ROOT / 'experiments/field_gap_full_20261102'
VARIANTS = [('existing_grid', None, True), ('alignment_only', None, False),
            ('linear_0p5s', .5, False), ('linear_1p0s', 1., False)]
DURATIONS = G.DURATIONS


def repair_target(observed, full, audit, dims, K, M, shape, maximum):
    """Repair only bounded holes within the already-associated target ID."""
    if not observed:
        return [], [], []
    groups = [[dict(observed[0], is_imputed=False)]]; decisions = []; holes = []
    for a, b in zip(observed, observed[1:]):
        dt = b['t_s'] - a['t_s']
        assert dt > 0
        missing = [s for s in full if a['frame'] < s['frame'] < b['frame']
                   and a['t_s'] < s['t_s'] < b['t_s']]
        filled = []; reason = None
        if missing and maximum is not None:
            if dt > maximum + 1e-7:
                reason = 'gap_exceeds_limit'
            mid_audit = audit[(audit.frame > a['frame']) & (audit.frame < b['frame'])]
            if 'target_outside_image' in ';'.join(mid_audit.reason.fillna('')):
                reason = reason or 'known_target_outside_image'
            velocity0 = np.array([a['vx'], a['vy']]); velocity1 = np.array([b['vx'], b['vy']])
            displacement = np.array([b['x'] - a['x'], b['y'] - a['y']])
            if np.linalg.norm(velocity1 - velocity0) > G.SETTINGS['acceleration_gate_mps2'] * dt + G.SETTINGS['velocity_noise_mps']:
                reason = reason or 'endpoint_velocity_inconsistent'
            if np.linalg.norm(displacement - .5 * (velocity0 + velocity1) * dt) > G.SETTINGS['position_noise_m'] + .5 * G.SETTINGS['acceleration_gate_mps2'] * dt ** 2:
                reason = reason or 'position_velocity_inconsistent'
            if abs(displacement[1]) > G.SETTINGS['lateral_noise_m'] + G.SETTINGS['lateral_speed_gate_mps'] * dt:
                reason = reason or 'lateral_jump'
            if a['vx'] * b['vx'] < 0 and min(abs(a['vx']), abs(b['vx'])) > 1:
                reason = reason or 'direction_reversal'
            if reason is None:
                for s in missing:
                    w = (s['t_s'] - a['t_s']) / dt
                    r = {k: v for k, v in a.items() if k not in ['x_det', 'y_det']}
                    r.update(frame=s['frame'], t_s=s['t_s'], is_imputed=True, score=None,
                             left_anchor_frame=a['frame'], right_anchor_frame=b['frame'])
                    for col in ['x', 'y', 'z']:
                        r[col] = (1 - w) * a[col] + w * b[col]
                    r['yaw'] = a['yaw'] + w * np.arctan2(np.sin(b['yaw'] - a['yaw']), np.cos(b['yaw'] - a['yaw']))
                    r['vx'], r['vy'] = displacement / dt
                    r['speed_mps'] = float(np.linalg.norm(displacement) / dt)
                    if not inside_image(r, dims, K, M, shape[1], shape[0]):
                        reason = 'interpolated_box_outside_image'; break
                    filled.append(r)
            accepted = reason is None and bool(filled)
            decisions.append(dict(left_frame=a['frame'], right_frame=b['frame'],
                                  start_s=a['t_s'], end_s=b['t_s'], anchor_gap_s=dt,
                                  accepted=accepted, reason=reason or 'bounded_linear_target',
                                  filled_points=len(filled) if accepted else 0))
        else:
            accepted = False
        if accepted:
            groups[-1].extend(filled); holes.append((a['t_s'], b['t_s']))
        elif dt > .2 + 1e-7:
            groups.append([])
        elif missing:
            holes.append((a['t_s'], b['t_s']))
        groups[-1].append(dict(b, is_imputed=False))
    return groups, decisions, holes


def span(groups):
    return max((g[-1]['t_s'] - g[0]['t_s'] for g in groups), default=0.)


def build():
    import cv2
    G.selfcheck(); cfg = read(CONFIG); OUT.mkdir(exist_ok=True)
    protocol = dict(source_survey=str(SURVEY.relative_to(ROOT)), recordings=60,
                    models='Fixed Verona adapted_V / Seminole adapted_S / original elsewhere',
                    variants=VARIANTS, durations_s=DURATIONS, settings=G.SETTINGS,
                    target='Same existing ID, bounded internal holes, endpoint motion and image visibility gates; no leader required',
                    pairing='Same frozen pair-repair algorithm as trial_field_gap_repair; no reassignment or cross-ID joins',
                    classification='30 frozen source members. Report all durations and all nonoverlapping windows per recording; longest available duration also reported.',
                    limits='Candidate geometry and identities unvalidated; no new training or camera changes')
    cp = ROOT / 'configs/field_gap_full.json'
    if cp.exists(): assert read(cp) == read_json_compatible(protocol)
    else: save(cp, protocol)
    manifest = {}
    def used(p):
        manifest[str(p.relative_to(ROOT))] = sha(p); return p
    used(CONFIG); used(ROOT / 'scripts/trial_field_gap_repair.py')
    for m in cfg['models'].values():
        assert sha(ROOT / m['checkpoint']) == m['sha256']
    rows = []; target_audit = []; pair_audit = []
    arrays = {(v, d): [] for v, _, _ in VARIANTS for d in DURATIONS}
    metadata = {(v, d): [] for v, _, _ in VARIANTS for d in DURATIONS}
    for row in cfg['clips']:
        clip = row['clip']; model = next((k for k in row['models'] if k != 'original'), 'original')
        tag = cfg['models'][model]['tag']; src = SURVEY / 'pairs' / f'{clip}_{tag}'
        td = TR / 'outputs/trajectories' / f'{clip}_{tag}'
        result = read(used(SURVEY / 'runs' / f'{clip}_{model}' / 'evaluation.json'))
        assert result['status'] not in ['EVALUATION_FAILED', 'SOURCE_IDENTITY_UNRESOLVED']
        records = []; observed = []; full = []; tracks = {}; dims = {}; times = []
        table = pd.DataFrame(); audit = pd.DataFrame(); K = M = None; shape = None
        if result.get('target_observations', 0):
            tid = str(result['identified_target'])
            observed = pd.read_csv(used(src / 'target_trajectory.csv')).sort_values('t_s').to_dict('records')
            tracks = {k: {s['frame']: s for s in st} for k, st in read(used(td / 'tracks.json'))['tracks'].items()}
            full = sorted(tracks[tid].values(), key=lambda s: s['t_s'])
            dims = read(used(src / 'dimensions.json'))
            table = pd.read_csv(used(td / f'vehicle_{tid}.csv'), dtype={'lead_id': str})
            audit = pd.read_csv(used(src / 'pair_audit.csv'), dtype={'target_id': str, 'leader_id': str})
            records = pd.read_csv(used(src / 'following_pairs.csv'), dtype={'target_id': str, 'leader_id': str}).sort_values('t_s').to_dict('records')
            times = [(int(r.frame), float(r.t_s)) for r in table.itertuples()]
            cal = read(used(TR / 'outputs/object_detection/camera-data' / f'{clip}_{tag}' / 'calibration_used.json'))
            K, M = np.array(cal['K']), np.array(cal['lidar2cam'])
            shape = cv2.imread(str(TR / 'data/camera-data' / clip / 'frames_all/000.jpg')).shape
        original_groups = G.rebuild(records, table, audit, times, dims, None)[0]
        original_starts = [g[0]['t_s'] for g in original_groups]
        base = dict(clip=clip, model=model, site=row['site'], view=row['view'], core=row['core'],
                    label=row['label'], identity_accepted=bool(result.get('identity_accepted')),
                    source_status=result['status'], training_exposed=clip in cfg['models'][model]['training_clips'],
                    previously_inspected=row['previously_inspected'])
        for variant, maximum, rounded in VARIANTS:
            folder = OUT / 'runs' / variant / clip; folder.mkdir(parents=True, exist_ok=True)
            own_groups, own_gaps, _ = repair_target(observed, full, audit, dims.get(str(result.get('identified_target'))), K, M, shape, maximum)
            own_flat = [dict(s, repaired_group=i) for i, g in enumerate(own_groups) for s in g]
            if own_flat:
                own_df = pd.DataFrame(own_flat); own_df.to_csv(folder / 'target_trajectory.csv', index=False)
                original = own_df.loc[~own_df.is_imputed]
                for col in ['frame', 't_s', 'x', 'y', 'vx', 'vy']:
                    np.testing.assert_allclose(original[col], [r[col] for r in observed], rtol=0, atol=0)
                assert own_df.frame.is_unique and (np.diff(own_df.t_s) > 0).all()
            groups, gaps, holes = G.rebuild(records, table, audit, times, dims, maximum, tracks)
            windows = G.window_metadata(groups, holes, rounded, original_starts if maximum else ())
            if maximum is None:
                for w in windows: w['accepted'] = True
            pair_flat = [dict(r, repaired_group=i) for i, g in enumerate(groups) for r in g]
            if pair_flat:
                pf = pd.DataFrame(pair_flat); pf.to_csv(folder / 'following_pairs.csv', index=False)
                for col in ['t_s', 'target_x_m', 'target_y_m', 'leader_x_m', 'leader_y_m', 'net_gap_m']:
                    np.testing.assert_allclose(pf.loc[~pf.is_imputed, col], [r[col] for r in records], rtol=0, atol=0)
                for r in pair_flat:
                    expected = (r['leader_x_m'] - r['target_x_m']) * r['direction'] - (dims[r['target_id']]['length_m'] + dims[r['leader_id']]['length_m']) / 2
                    assert expected > 0 and abs(expected - r['net_gap_m']) < 1e-6
                    for role, oid in [('target', r['target_id']), ('leader', r['leader_id'])]:
                        state = tracks[oid].get(r['frame'])
                        if r['is_imputed'] and state is not None and 'x_det' in state:
                            assert r[role + '_x_m'] == state['x'] and r[role + '_y_m'] == state['y']
            save(folder / 'windows.json', windows)
            window_counts = {}
            for duration in DURATIONS:
                raw = []; meta = []
                for w in windows:
                    if not w['accepted'] or w['duration_s'] != duration: continue
                    group = groups[w['group']]; ts = np.array([r['t_s'] for r in group])
                    query = w['start_s'] + np.arange(duration * 10) / 10
                    assert query[0] >= ts[0] - 1e-7 and query[-1] <= ts[-1] + 1e-7
                    values = np.array([[r['follower_speed_mps'], r['leader_speed_mps'], r['net_gap_m']] for r in group])
                    x = np.stack([np.interp(query, ts, values[:,i]) for i in range(3)], 1).astype(np.float32)
                    assert np.isfinite(x).all() and (x[:, :2] >= 0).all() and (x[:, 2] > 0).all()
                    raw.append(x); meta.append(dict(w, **base, variant=variant))
                raw = np.stack(raw) if raw else np.empty((0, duration * 10, 3), np.float32)
                window_counts[str(duration)] = len(raw)
                if variant == 'existing_grid' and duration in [2, 4]:
                    old = SURVEY / 'runs' / f'{clip}_{model}' / f'windows_{duration}s.npz'
                    previous = np.load(used(old))['primitive_windows'] if old.exists() else np.empty_like(raw)
                    np.testing.assert_allclose(raw, previous, rtol=0, atol=1e-5)
                arrays[variant, duration].append(raw); metadata[variant, duration].extend(meta)
            old_ref = read(ROOT / 'experiments/field_gap_repair_20261101/summary.json')['recordings']
            reference = next(r for r in old_ref if r['clip'] == clip and r['variant'] == variant)
            assert reference['windows'] == window_counts
            pair_audit.extend(dict(g, **base, variant=variant) for g in gaps)
            target_audit.extend(dict(g, **base, variant=variant) for g in own_gaps)
            r = dict(base, variant=variant, status='COMPLETE', target_observed_points=len(observed),
                     target_imputed_points=sum(s['is_imputed'] for s in own_flat),
                     target_repaired_gaps=sum(g['accepted'] for g in own_gaps),
                     longest_target_span_s=float(span(own_groups)),
                     target_total_point_fraction_imputed=sum(s['is_imputed'] for s in own_flat) / len(own_flat) if own_flat else None,
                     pair_observed_points=len(records), pair_imputed_points=sum(s['is_imputed'] for s in pair_flat),
                     pair_repaired_gaps=sum(g['accepted'] for g in gaps), longest_pair_span_s=float(span(groups)),
                     windows=window_counts)
            save(folder / 'result.json', r); rows.append(r)
        save(OUT / 'progress.json', dict(completed_recordings=len(rows)//4, total_recordings=60, clip=clip))
    assert len(rows) == 240
    for variant, _, _ in VARIANTS:
        folder = OUT / 'classification' / variant; folder.mkdir(parents=True, exist_ok=True)
        for d in DURATIONS:
            raw = np.concatenate(arrays[variant, d]); meta = metadata[variant,d]
            np.savez_compressed(folder / f'windows_{d}s.npz', primitive_windows=raw,
                                labels=np.array([r['label'] for r in meta], np.int64), dt=np.array(.1))
            save(folder / f'windows_{d}s_metadata.json', meta)
    save(OUT / 'build_summary.json', dict(complete=True, recordings=60, runs=rows))
    pd.DataFrame([dict({k:v for k,v in r.items() if k != 'windows'}, **{f'windows_{d}s':r['windows'][str(d)] for d in DURATIONS}) for r in rows]).to_csv(OUT / 'recording_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(target_audit).to_csv(OUT / 'target_gap_audit.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pair_audit).to_csv(OUT / 'pair_gap_audit.csv', index=False, encoding='utf-8-sig')
    for path, digest in manifest.items(): assert sha(ROOT / path) == digest
    save(OUT / 'input_manifest.json', manifest)
    save(OUT / 'build_validation.json', dict(status='PASS', model_recording_variants=240,
         source_inputs_unchanged=True, original_classifier_windows_reproduced=True,
         previous_pair_repair_counts_reproduced=True, observed_positions_velocities_unchanged=True,
         no_extrapolation=True, positive_net_gap_common_coordinates=True))
    print('BUILD_COMPLETE', len(rows))


def read_json_compatible(value):
    import json
    return json.loads(json.dumps(value))


def classify():
    import torch
    from evaluate_field_baseline import TrajectoryNet, features, normalize, metrics, nonoverlap
    assert read(OUT / 'build_summary.json')['complete']; torch.set_num_threads(4)
    manifest = read(ROOT / 'experiments/field_baseline_20261101/model_manifest.json'); members = []
    for r in manifest:
        p = ROOT / r['path']; cp = torch.load(p, map_location='cpu', weights_only=True); c = cp['config']
        m = TrajectoryNet(cp['input_dim'], c['architecture'], c['width'], c['dropout'], c['layers']).eval()
        m.load_state_dict(cp['model_state_dict'], strict=True)
        members.append((m,c,read(p.parent / 'feature_stats.json')))
        r['sha256'] = sha(p); r['feature_stats_sha256'] = sha(p.parent / 'feature_stats.json')
    save(OUT / 'classifier_manifest.json', manifest); results = []; all_recordings = []
    for variant, _, _ in VARIANTS:
        folder = OUT / 'classification' / variant
        for duration in DURATIONS:
            raw = np.load(folder / f'windows_{duration}s.npz')['primitive_windows']
            meta = read(folder / f'windows_{duration}s_metadata.json')
            if not len(raw):
                results.append(dict(variant=variant, duration_s=duration, status='NO_WINDOWS')); continue
            for arch in ['transformer','multiscale_cnn']:
                probs = []; clipping = []
                for m,c,stats in members:
                    if c['architecture'] != arch: continue
                    x = features(raw, c.get('features','legacy11'), c.get('smoothing',7))
                    clipping.append(float(((x < np.array(stats['lower'])) | (x > np.array(stats['upper']))).mean()))
                    with torch.no_grad(): prob = torch.sigmoid(m(torch.from_numpy(normalize(x,stats)))).numpy()
                    assert np.isfinite(prob).all(); probs.append(prob)
                assert len(probs) == 15
                pred = pd.DataFrame(meta); pred['prob_av'] = np.mean(probs, 0); pred['model_sd'] = np.std(probs,0)
                pred['predicted_label'] = (pred.prob_av >= .5).astype(int)
                pred.to_csv(folder / f'{arch}_{duration}s_predictions.csv', index=False)
                first = pred.sort_values('start_s').groupby('clip', sort=False).head(1)
                aggregated = []
                for clip,g in pred.groupby('clip', sort=False):
                    selected = nonoverlap(g)
                    a = selected.iloc[0].to_dict(); a.update(prob_av=float(selected.prob_av.mean()),
                        contributing_windows=len(selected), input_time_span_s=float((selected.end_s-selected.start_s+.1).sum()),
                        window_starts=selected.start_s.tolist(), variant=variant, architecture=arch)
                    a['predicted_label'] = int(a['prob_av'] >= .5)
                    a['first_window_model_sd'] = a.pop('model_sd')
                    a['start_s'], a['end_s'] = float(selected.start_s.min()), float(selected.end_s.max())
                    a['imputed_fraction'] = float(selected.imputed_fraction.mean())
                    a['maximum_window_imputed_fraction'] = float(selected.imputed_fraction.max())
                    a['leader_ids'] = selected.leader_id.unique().tolist()
                    aggregated.append(a)
                rec = pd.DataFrame(aggregated); all_recordings.extend(aggregated)
                rec.to_csv(folder / f'{arch}_{duration}s_recordings.csv', index=False)
                subsets = {n:metrics(rec.loc[mask]) for n,mask in [('core',rec.core),('supplementary',~rec.core),
                    ('not_detector_training',~rec.training_exposed),('not_previously_in_pipeline',~rec.previously_inspected)]}
                results.append(dict(variant=variant, architecture=arch, duration_s=duration, windows=len(raw),
                     nonoverlap_recording_metrics=metrics(rec), first_window_metrics=metrics(first),
                     all_windows_correlated_metrics=metrics(pred), subsets=subsets,
                     feature_clipping_range=[min(clipping),max(clipping)]))
                if variant == 'existing_grid' and duration == 2:
                    old = pd.read_csv(SURVEY / 'classification/view_adapted_mix' / f'{arch}_2s_predictions.csv')
                    joined = pred.merge(old, on=['clip','start_s','end_s'], suffixes=('_new','_old'))
                    assert len(joined) == len(old) == len(pred)
                    np.testing.assert_allclose(joined.prob_av_new, joined.prob_av_old, rtol=0, atol=1e-5)
    allr = pd.DataFrame(all_recordings); longest = []
    for (variant, arch),g in allr.groupby(['variant','architecture']):
        selected = g.sort_values('duration_s', ascending=False).groupby('clip',sort=False).head(1)
        selected.to_csv(OUT / 'classification' / variant / f'{arch}_longest_available_recordings.csv', index=False)
        longest.append(dict(variant=variant, architecture=arch, metrics=metrics(selected),
             duration_counts={str(int(k)):int(v) for k,v in selected.duration_s.value_counts().items()},
             not_detector_training=metrics(selected.loc[~selected.training_exposed])))
    save(OUT / 'classification_summary.json', dict(status='CANDIDATE_TRAJECTORY_DIAGNOSTIC', results=results,
          longest_available=longest, existing_grid_probabilities_reproduced=True,
          notes=['All members frozen; no new training or label-dependent selection',
                 'Every eligible window predicted; video metric averages all chronological nonoverlapping windows at each duration',
                 'Longest available selects duration by coverage only, not correctness; mixed durations are diagnostic',
                 'Original source training used 4 s; 2 s is supplementary',
                 'Missing-leader target tracks retained, not padded into 11-feature classifiers',
                 'No independent metric/identity truth, no claim of field generalization accuracy']))
    print('CLASSIFICATION_COMPLETE')


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    build = read(OUT / 'build_summary.json'); rows = build['runs']; classification = read(OUT / 'classification_summary.json')
    assert build['complete'] and len(rows) == 240
    for path,digest in read(OUT / 'input_manifest.json').items(): assert sha(ROOT/path) == digest
    summary = []
    original_by = {r['clip']:r for r in rows if r['variant']=='existing_grid'}
    for variant, _, _ in VARIANTS:
        for scope in ['all','core','supplementary']:
            rr = [r for r in rows if r['variant']==variant and (scope=='all' or r['core']==(scope=='core'))]
            valid = [r for r in rr if r['target_observed_points']]
            summary.append(dict(variant=variant, scope=scope, recordings=len(rr), matched_targets=len(valid),
                  target_repaired_recordings=sum(r['target_repaired_gaps']>0 for r in rr),
                  target_gaps_filled=sum(r['target_repaired_gaps'] for r in rr),
                  target_imputed_points=sum(r['target_imputed_points'] for r in rr),
                  target_span_extended_recordings=sum(r['longest_target_span_s'] > original_by[r['clip']]['longest_target_span_s'] + 1e-6 for r in rr),
                  median_matched_target_longest_span_s=float(np.median([r['longest_target_span_s'] for r in valid])) if valid else None,
                  pair_repaired_recordings=sum(r['pair_repaired_gaps']>0 for r in rr),
                  pair_gaps_filled=sum(r['pair_repaired_gaps'] for r in rr),
                  **{f'target_span_at_least_{d}s':sum(r['longest_target_span_s']>=d for r in rr) for d in [4,6,8,10]},
                  **{f'pair_window_{d}s_recordings':sum(r['windows'][str(d)]>0 for r in rr) for d in DURATIONS}))
    save(OUT/'summary.json', dict(coverage=summary, complete=True)); pd.DataFrame(summary).to_csv(OUT/'coverage.csv',index=False,encoding='utf-8-sig')
    by={(r['clip'],r['variant']):r for r in rows}; comparisons=[]
    for row in read(CONFIG)['clips']:
        a=by[row['clip'],'existing_grid'];b=by[row['clip'],'linear_1p0s']
        comparisons.append(dict(clip=row['clip'], core=row['core'],site=row['site'],view=row['view'],
             source_status=b['source_status'], target_before_s=a['longest_target_span_s'],target_after_s=b['longest_target_span_s'],
             target_added_points=b['target_imputed_points'], target_filled_gaps=b['target_repaired_gaps'],
             pair_before_s=a['longest_pair_span_s'],pair_after_s=b['longest_pair_span_s'],
             pair_added_points=b['pair_imputed_points'], pair_filled_gaps=b['pair_repaired_gaps']))
    pd.DataFrame(comparisons).to_csv(OUT/'before_after.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    x=np.array([r['target_before_s'] for r in comparisons]);y=np.array([r['target_after_s'] for r in comparisons])
    axes[0].scatter(x,y,c=['#267a9a' if r['core'] else '#ba7040' for r in comparisons],alpha=.7)
    lim=max(x.max(),y.max())+1; axes[0].plot([0,lim],[0,lim],'k--',lw=1)
    axes[0].set(xlabel='Before: longest target span (s)',ylabel='After: longest target span (s)',title='All 60 recordings; zeros retained')
    index=np.arange(len(DURATIONS));width=.25
    for offset,name,label in [(-width,'existing_grid','Previous'),(0,'alignment_only','Window alignment'),(width,'linear_1p0s','+ internal gaps <=1 s')]:
        r=next(r for r in summary if r['variant']==name and r['scope']=='all')
        axes[1].bar(index+offset,[r[f'pair_window_{d}s_recordings'] for d in DURATIONS],width,label=label)
    axes[1].set_xticks(index,[str(d) for d in DURATIONS]);axes[1].set(xlabel='Following-window duration (s)',ylabel='Recordings with a candidate window',title='Coverage, not ground-truth accuracy');axes[1].legend(fontsize=8)
    fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    lines=['全视频短缺口补全与分类测试 | 2026-11-02','','60 段全部纳入（核心 43 / 补充 17），复用已完成的原帧检测和 MOT；Verona/Seminole 用现有适配权重，其他视角用原权重。没有新增训练。',
      '分别保存自车和前后车配对：自车补全不要求前车存在；分类仍要求有效前车，未给旧网络填零。',
      '原观测位置、速度和共同坐标不变；仅同一轨迹内部、有两端观测且通过运动/可见性检查的缺口插值。0.5 / 1 秒是本地试验阈值，补点标记保留。',
      '自车连续区间长度是候选连续性指标，不等于独立核验的准确轨迹。配对窗口仍限制估计比例 <=25%。', '', '全量覆盖：']
    for r in summary:
        if r['scope']!='all':continue
        lines.append(f"{r['variant']}: 自车匹配 {r['matched_targets']}/60；自车补全 {r['target_repaired_recordings']} 段/{r['target_gaps_filled']} 处/{r['target_imputed_points']} 点，其中 {r['target_span_extended_recordings']} 段最长区间延长；自车最长连续区间中位数（匹配者）{r['median_matched_target_longest_span_s']:.3f}s；配对补全 {r['pair_repaired_recordings']} 段/{r['pair_gaps_filled']} 处。")
        lines.append('  自车连续跨度 >=4/6/8/10s：'+'/'.join(str(r[f'target_span_at_least_{d}s']) for d in [4,6,8,10])+'；配对 2/4/6/8/10s 窗口视频数：'+'/'.join(str(r[f'pair_window_{d}s_recordings']) for d in DURATIONS))
    lines+=['','冻结分类器：所有合格窗口均推理，每架构 15 个模型平均概率。按视频、按时长选取按时间排列的全部不重叠窗口，平均概率后判别；另存首窗口结果供旧统计口径对照。下面正确数不是代表性路测准确率。']
    for r in classification['results']:
        if 'architecture' not in r:continue
        m=r['nonoverlap_recording_metrics'];nt=r['subsets']['not_detector_training']
        lines.append(f"{r['variant']} {r['architecture']} {r['duration_s']}s: {m['correct']}/{m['n']} ({100*m['accuracy']:.1f}%)；排除对应检测器训练记录 {nt.get('correct',0)}/{nt['n']}。")
    lines+=['','每视频使用最长可用时长（仅按覆盖选择），再平均该时长所有不重叠窗口：']
    for r in classification['longest_available']:
        m=r['metrics'];lines.append(f"{r['variant']} {r['architecture']}: {m['correct']}/{m['n']}；各时长视频数 {r['duration_counts']}。")
        if r['variant']=='linear_1p0s':
            cm=m['confusion_HV_AV'];lines.append(f"  HV 判对 {cm[0][0]}/{sum(cm[0])}；AV 判对 {cm[1][1]}/{sum(cm[1])}。当前全部判为 HV，整体正确数不能说明已具备 AV/HV 区分能力。" if cm[0][1]+cm[1][1]==0 else f"  混淆矩阵（HV/AV）{cm}。")
    lines+=['','解释：起点取整修正与插值贡献分开统计；更长轨迹或更多补点不代表更高物理精度。线性补点会削弱缺口内真实加速度/jerk，不宜当作独立实测行为。',
      '线性位置插值的缺口内速度为常数，接缝处可能产生速度跳变和虚假加速度/jerk。输出保留缺失标记，但旧分类器没有使用这些标记，当前推理仅作诊断。',
      '短时身份匹配不保证整条轨迹无换车/错误尾段；此前 Verona 已发现出画后的错误跟踪，仍需可靠区间审核。',
      '没有同车两端观测的出画/长缺失、不同 ID、真实前车切换不连接。当前 car-only 跟踪的重型前车缺口仍存在。',
      '未参与检测器训练不等于车辆级盲测；源分类器以 4 秒训练，2 秒结果为补充诊断。当前 4 秒样本数不足以评估 AV/HV 泛化。',
      '文件：before_after.csv（每视频前后时长）、recording_results.csv（四种设置）、target_gap_audit.csv、pair_gap_audit.csv、runs/（补点标记轨迹与窗口）、classification/（NPZ/逐窗口/逐视频结果）、overview.png。source_reviews/ 为每个补全视频最长缺口的前/中/后源图，figures/ 为对应轨迹和速度曲线；图像生成不等于全部人工核验。',
      '复现：','.venv-bev/Scripts/python.exe scripts/run_field_gap_full.py build','.venv-classifier/Scripts/python.exe scripts/run_field_gap_full.py classify','.venv-bev/Scripts/python.exe scripts/run_field_gap_full.py report','.venv-bev/Scripts/python.exe scripts/review_field_gap_full.py']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    save(OUT/'validation.json',dict(status='PASS',recordings=60,variant_recording_runs=240,
         build=read(OUT/'build_validation.json'),baseline_predictions_reproduced=classification['existing_grid_probabilities_reproduced'],source_hashes_unchanged=True,
         note='Software/data invariants, not ground-truth accuracy.'))
    print('REPORT_COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','classify','report']);a=p.parse_args()
    {'build':build,'classify':classify,'report':report}[a.stage]()
