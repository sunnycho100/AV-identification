"""Build review inputs only; never creates or modifies human decisions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'experiments/field_leader_review_20261102'
DEST = ROOT / 'data/review_projects/todd_se_extension'
NOTES = {
    'AV_T_WE_4': (2.636, [155, 165, 168, 169, 175, 190], '之前在 168–169 帧发生三维位置跳变。请从整段视频寻找连续可见区间，并在跳变前后补充自车框。'),
    'AV_T_WE_5': (2.803, [29, 60, 100, 113, 146], '之前远处的三维检测提前终止。可选择更早或更晚仍能辨认自车的画面，不受旧轨迹边界限制。'),
    'HV_T_EW_2': (.567, [0, 38, 80, 130, 162, 179, 180], '你已确认前面足够三秒，后几秒画面静止。请优先选择前段连续可见区间，标记冻结尾段起点；保留范围用于后续裁剪，原视频保持完整。'),
    'HV_T_WE_5': (2.603, [52, 100, 130, 134, 142, 148, 152, 157], '之前 131–148 帧缺少检测。请确认缺口前后是否为同一辆自车，并在缺口中补充自车框。'),
    'HV_T_WE_6': (2.135, [30, 47, 65, 85, 115, 149], '之前算法只信任较晚的一段。请重点查看较早的画面，自行确认真实自车，避免将画面标识或其他车辆接入。'),
}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    old = json.loads((PACK / 'manifest.json').read_text(encoding='utf-8'))
    cases = []
    for clip, (duration, checkpoints, note) in NOTES.items():
        source = next(c for c in old['cases'] if c['clip'] == clip)
        raw = (PACK / 'data' / f'{clip}.js').read_text(encoding='utf-8').strip()
        payload = json.loads(raw[len('window.registerClip('):].rstrip(';').rstrip(')'))
        frames = payload['frames']
        times = json.loads((ROOT / 'trajectory/data/camera-data' / clip / 'frame_times.json').read_text(encoding='utf-8'))['frames']
        boxes = {f['frame']: f.get('boxes', []) for f in frames}
        full = [dict(frame=f['frame'], t_s=f['t_s'], boxes=boxes.get(f['frame'], [])) for f in times]
        data = dict(clip=clip, width=1920, height=1080, frames=full,
                    timestamp_gaps=[dict(before=a['frame'], after=b['frame'], gap_s=b['t_s']-a['t_s'])
                                    for a, b in zip(full, full[1:]) if b['t_s']-a['t_s'] > .2])
        dest = DEST / 'clips' / f'{clip}.json'
        write(dest, data)
        video = (PACK / source['video_path']).resolve()
        assert video.is_relative_to(ROOT.resolve()) and video.is_file()
        assert all((ROOT / f'trajectory/data/camera-data/{clip}/frames_all/{f["frame"]:03d}.jpg').is_file() for f in full)
        cases.append(dict(clip=clip, title=clip, note=note, current_extracted_duration_s=duration,
            frames=len(full), duration_s=full[-1]['t_s']-full[0]['t_s'],
            checkpoints=[f for f in checkpoints if any(r['frame']==f for r in full)],
            target_hint=source.get('target_id'), video=video.relative_to(ROOT.resolve()).as_posix(),
            frames_template=f'trajectory/data/camera-data/{clip}/frames_all/{{frame:03d}}.jpg',
            frame_data=f'clips/{clip}.json', source_video_sha256=source['source_video_sha256'],
            frame_data_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    write(DEST / 'project.json', dict(schema='trajectory-review-project/v1', id='todd_se_extension',
        title='Todd SE · 五段轨迹补充审核', required_duration_s=3, media_root='../../..', demo=False,
        description='五段视频均由采集者确认可见轨迹超过三秒。本轮标注可见区间、连续自车身份与冻结段，供后续重新跟踪；审核完成不等于三维轨迹重建成功。', cases=cases))
    print(json.dumps(dict(project=str(DEST), clips=[dict(clip=c['clip'], frames=c['frames'], duration_s=c['duration_s']) for c in cases]), ensure_ascii=False))


if __name__ == '__main__':
    main()
