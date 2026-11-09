"""Expose applied user decisions as read-only provenance in the review pack."""
import hashlib
import json
from build_field_core_dataset import ROOT, read, save


def annotate_manifest(manifest):
    config = ROOT / 'configs/field_review_application.json'
    if not config.exists():
        return manifest
    cfg = read(config)
    applied = ROOT / 'experiments' / cfg['experiment']
    if not (applied / 'summary.json').exists():
        return manifest
    summary = read(applied / 'summary.json')
    source = ROOT / cfg['decisions']
    if hashlib.sha256(source.read_bytes()).hexdigest() != summary['decisions_sha256']:
        return manifest  # A newer user export must be applied explicitly first.
    document = read(applied / 'reviewed_intervals.json')
    records = {r['clip']: r for r in document['records']}
    reports = {r['clip']: r for r in summary['reports']}
    # Derived associations supplement, but never replace, the user's selections.
    recovered, seed_roles, recovery_dataset = {}, {}, None
    active_path = ROOT / 'configs/field_active_dataset.json'
    if active_path.exists():
        active = read(active_path)
        derived = ROOT / active['dataset']
        needed = ['build_summary.json', 'seed_recovery_summary.json', 'input_manifest.json', 'validation.json']
        if all((derived / name).exists() for name in needed):
            inputs = read(derived / 'input_manifest.json')
            if (active.get('annotations_sha256') == summary['decisions_sha256']
                    and inputs.get(cfg['decisions']) == summary['decisions_sha256']
                    and read(derived / 'validation.json').get('status') == 'PASS'):
                recovered = {r['clip']: r for r in read(derived / 'build_summary.json')['rows']
                             if r['variant'] == 'recovered'}
                for item in read(derived / 'seed_recovery_summary.json'):
                    seed_roles.setdefault(item['clip'], []).append(item)
                recovery_dataset = active['dataset']
    labels = {'confirm_pair':'有前车', 'no_leader':'无前车', 'leader_not_visible':'前车暂时不可见'}
    count = 0
    for c in manifest['cases']:
        r = records.get(c['clip'])
        if not r:
            continue
        assert r['source_video_sha256'] == c['source_video_sha256'] and r['tracks_sha256'] == c['tracks_sha256']
        c.setdefault('original_proposal', {k:c.get(k) for k in
                     ['target_id','leader_id','range','frame','question','initial_assessment']})
        c['applied_review'] = r
        c['target_id'] = r['target'].get('id') if r.get('target') else None
        c['leader_id'] = r['leader'].get('id') if r.get('leader') else None
        c['range'] = [r['start_frame'],r['end_frame']]
        selection = r.get('leader') or r.get('target') or {}
        c['frame'] = selection.get('frame', r['start_frame'])
        c['question'] = '已应用你的审核。可查看原始帧，或为其他时段补充记录；原自动初选不再作为结论。'
        report = reports[r['clip']]
        c['initial_assessment'] = (f'已应用：{labels[r["decision"]]}，第 {r["start_frame"]}–{r["end_frame"]} 帧；'
                                   f'自车 {report["target_points"]} 个检测支持点，配对 {report["paired_points"]} 点。'
                                   + ('人工框仍待三维关联。' if report['pending_3d_roles'] else '')
                                   + ('备注：'+r['note'] if r['note'] else ''))
        c.pop('derived_resolution', None)
        if c['clip'] in recovered:
            latest = recovered[c['clip']]
            roles = seed_roles.get(c['clip'], [])
            c['derived_resolution'] = dict(dataset=recovery_dataset, roles=roles,
                                           target_points=latest['target_points'],
                                           paired_points=latest['paired_points'],
                                           status='candidate_metric_association_not_ground_truth')
            descriptions = []
            for item in roles:
                role = '自车' if item['role'] == 'target' else '前车'
                if item['method'] == 'pending':
                    descriptions.append(f'{role}人工框仍待可靠三维关联')
                elif item['method'] == 'existing_mot':
                    descriptions.append(f'{role}人工框已由助手关联到 MOT {item["track_id"]}')
                else:
                    descriptions.append(f'{role}人工框已由助手关联到原始三维检测片段')
            c['initial_assessment'] = (f'已应用：{labels[r["decision"]]}，第 {r["start_frame"]}–{r["end_frame"]} 帧；'
                                       f'当前自车 {latest["target_points"]} 个检测支持点，配对 {latest["paired_points"]} 点。'
                                       + ('；'.join(descriptions)+'。' if descriptions else '')
                                       + ('关联结果为三维候选，用户原始选择保留。' if descriptions else '')
                                       + ('备注：'+r['note'] if r['note'] else ''))
        c['review_visibility_overrides'] = [v for v in document['visibility_overrides'] if v['clip']==c['clip']]
        count += 1
    manifest['applied_review_count'] = count
    manifest['applied_decisions_sha256'] = summary['decisions_sha256']
    manifest.pop('derived_recovery_dataset', None)
    if recovery_dataset:
        manifest['derived_recovery_dataset'] = recovery_dataset
    return manifest


def main():
    pack = ROOT / read(ROOT/'configs/field_review_application.json')['review_pack']
    manifest = annotate_manifest(read(pack/'manifest.json'))
    save(pack/'manifest.json', manifest)
    (pack/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';',encoding='utf-8')
    print('Applied review displayed for',manifest.get('applied_review_count',0),'clips')


if __name__ == '__main__':
    main()
