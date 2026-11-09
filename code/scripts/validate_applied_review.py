"""Check applied labels against original observations and human exit notes."""
import hashlib
import json
import sys
import pandas as pd
from build_field_core_dataset import ROOT, read, save


def main():
    cfg = read(ROOT / 'configs/field_review_application.json')
    out = ROOT / 'experiments' / cfg['experiment']
    summary = read(out / 'summary.json')
    doc = read(out / 'decisions_source.json')
    checks = []
    def check(name, value):
        assert value, name
        checks.append(name)
    check('source_annotation_preserved', hashlib.sha256((ROOT/cfg['decisions']).read_bytes()).hexdigest()==summary['decisions_sha256'])
    check('all_43_core_clips_applied', len(summary['reports'])==43)
    target_count = pair_count = 0
    for r in doc['records']:
        folder = out/'clips'/r['clip']
        own = pd.read_csv(folder/'target_trajectory.csv')
        leader = pd.read_csv(folder/'leader_trajectory.csv')
        pairs = pd.read_csv(folder/'following_pairs.csv')
        audit = pd.read_csv(folder/'frame_audit.csv')
        tracks = read(ROOT/'trajectory/outputs/trajectories'/f'{r["clip"]}_{r["tag"]}'/'tracks.json')['tracks']
        times = {v['frame']:v['t_s'] for v in read(ROOT/'trajectory/data/camera-data'/r['clip']/'frame_times.json')['frames']}
        for data in [own,leader,pairs]:
            check(r['clip']+'_within_review_interval_'+str(len(checks)), all(r['start_frame']<=f<=r['end_frame'] for f in data.frame))
            check(r['clip']+'_native_timestamps_'+str(len(checks)), all(abs(t-times[int(f)])<1e-9 for f,t in zip(data.frame,data.t_s)))
        for role,data in [('target',own),('leader',leader)]:
            selection = r.get(role)
            if selection and selection['type']=='track':
                states = {s['frame']:s for s in tracks[selection['id']]}
                check(r['clip']+'_'+role+'_original_metric_observations', all(
                    'x_det' in states[row.frame] and all(abs(getattr(row,key)-states[row.frame][key])<1e-8 for key in ['x','y','vx','vy'])
                    for row in data.itertuples()))
            else:
                check(r['clip']+'_'+role+'_no_invented_metric_points', data.empty)
        if r['decision']!='confirm_pair':
            check(r['clip']+'_absent_or_invisible_has_no_pairs', pairs.empty and leader.empty)
        check(r['clip']+'_complete_frame_audit',len(audit)==r['end_frame']-r['start_frame']+1)
        check(r['clip']+'_positive_gap',bool((pairs.net_gap_m>0).all()))
        if r['clip']=='HV_V_EW_3':
            check('leader_exit_274_no_stale_tail',all(leader.frame<=274) and all(pairs.frame<=274))
            check('leader_exit_changes_to_not_visible',all(audit[audit.frame>274].leader_state=='not_visible'))
        if r['clip']=='HV_V_EW_6':
            check('target_exit_308_no_stale_tail',all(own.frame<=308))
        target_count+=len(own);pair_count+=len(pairs)
    check('summary_matches_exported_points',target_count==summary['target_points'] and pair_count==summary['paired_points'])
    check('eight_manual_seeds_retained',len(read(out/'pending_3d_seeds.json'))==8)
    result=dict(status='PASS',checks=len(checks),target_points=target_count,paired_points=pair_count,
                check_names=checks,scope='Data integrity, review semantics, and unchanged metric observations; not independent trajectory accuracy.')
    save(out/'validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='check_names'}))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
