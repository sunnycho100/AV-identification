"""Generate explicitly synthetic, redistributable example frames. No field data."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'examples/demo'
root.mkdir(parents=True, exist_ok=True)
cases = []
for name, frozen in [('DEMO_CONTINUOUS', False), ('DEMO_FROZEN_TAIL', True)]:
    folder = root / name
    folder.mkdir(exist_ok=True)
    frames = []
    for i in range(51):
        x = 80 + min(i, 40 if frozen else 50) * 12
        box = [x, 220, x+85, 270]
        (folder / f'{i:03d}.svg').write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
<rect width="960" height="540" fill="#e5ebe7"/><rect y="140" width="960" height="250" fill="#596566"/>
<path d="M0 320H960" stroke="#f1e9cb" stroke-width="4" stroke-dasharray="35 25"/>
<rect x="{x}" y="220" width="85" height="50" rx="8" fill="#efb856"/><rect x="{x+20}" y="227" width="30" height="36" rx="3" fill="#42626b"/>
<text x="30" y="65" font-family="sans-serif" font-size="26" fill="#253e3c">SYNTHETIC DEMO / NOT RESEARCH DATA</text>
<text x="30" y="465" font-family="sans-serif" font-size="22" fill="#253e3c">{'Frozen after frame 40' if frozen else 'Continuous target motion'} · frame {min(i,40) if frozen else i}</text></svg>''', encoding='utf-8')
        frames.append(dict(frame=i, t_s=i/10, boxes=[dict(id='demo-target', bbox=box, kind='track', observed=True)]))
    data = root / f'{name}.json'
    data.write_text(json.dumps(dict(clip=name, width=960, height=540, frames=frames, timestamp_gaps=[]), indent=2), encoding='utf-8')
    cases.append(dict(clip=name, title=name, note='合成演示，不属于研究样本。可练习选三秒区间、画框和冻结尾段标记。',
        current_extracted_duration_s=None, frames=51, duration_s=5, checkpoints=[0,15,30,40,50],
        video=None, frames_template=name+'/{frame:03d}.svg', frame_data=data.name,
        source_video_sha256=hashlib.sha256(('synthetic:'+name).encode()).hexdigest(),
        frame_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest()))
(root / 'project.json').write_text(json.dumps(dict(schema='trajectory-review-project/v1', id='synthetic_demo',
    title='合成演示 · 区间与自车审核', description='随源码发布的合成数据。可独立运行，无需模型、视频或 GPU。',
    required_duration_s=3, media_root='.', demo=True, cases=cases), ensure_ascii=False, indent=2), encoding='utf-8')
print('Synthetic demo generated.')
