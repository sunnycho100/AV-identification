"""Decode source clips for inspection; preserve native-frame timestamps and indices."""
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'.runtime/modern-deps'))
import cv2,numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
parser=argparse.ArgumentParser();parser.add_argument('--clips',nargs='*');args=parser.parse_args()
records=list(csv.DictReader((root/'catalogs/field_clips.csv').open(encoding='utf-8-sig')))
out=root/'experiments/field_pilot_20261101';(out/'inspection').mkdir(parents=True,exist_ok=True)
selected=[r for r in records if r['site']=='T_Todd_Dr' and r['view']=='View_SE']
thumbs=[];probes=[]
for row in selected:
    cap=cv2.VideoCapture(str(root/row['video_path']))
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frame=max(0,count//2)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release()
    if not ok:raise RuntimeError(row['clip_id'])
    image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));image.thumbnail((480,270))
    tile=Image.new('RGB',(480,300),'white');tile.paste(image,(0,30));ImageDraw.Draw(tile).text((8,7),f"{row['clip_id']} | frame {frame} | {fps:.3f} fps",fill='black')
    thumbs.append(tile);image.save(out/'inspection'/f"{row['clip_id']}_mid.jpg")
    probes.append(dict(clip=row['clip_id'],fps=fps,container_frame_count=count,mid_frame=frame))
canvas=Image.new('RGB',(480*3,300*((len(thumbs)+2)//3)),(230,230,230))
for i,im in enumerate(thumbs):canvas.paste(im,((i%3)*480,(i//3)*300))
canvas.save(out/'inspection/todd_views.jpg')
(out/'inspection/video_probe.json').write_text(json.dumps(probes,indent=2))
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
for clip in args.clips or []:
    row=next(r for r in records if r['clip_id']==clip)
    data=root/'trajectory/data/camera-data'/clip;frames=data/'frames_all';frames.mkdir(parents=True,exist_ok=True)
    if any(frames.glob('*.jpg')):raise RuntimeError(f'Use existing frames or a new directory: {frames}')
    command=[ffmpeg,'-hide_banner','-i',str(root/row['video_path']),'-vf','showinfo','-vsync','0','-start_number','0','-q:v','2',str(frames/'%03d.jpg')]
    p=subprocess.run(command,text=True,capture_output=True)
    (out/f'{clip}_decode.log').write_text(p.stderr)
    if p.returncode:raise RuntimeError(p.stderr[-2000:])
    import re
    times=[]
    for line in p.stderr.splitlines():
        m=re.search(r'\bn:\s*(\d+)\s+pts:.*?pts_time:([^\s]+)',line)
        if m:times.append((int(m.group(1)),float(m.group(2))))
    if len(times)!=len(list(frames.glob('*.jpg'))):raise RuntimeError('Frame timestamp mismatch')
    t0=times[0][1]
    table={'clip':clip,'source':row['video_path'],'fps_nominal':next(p['fps'] for p in probes if p['clip']==clip),
           'method':'ffmpeg native decoded frame indices and presentation timestamps (-vsync 0)',
           'frames':[dict(frame=i,src=i,t_s=round(t-t0,6),copy=False) for i,t in times]}
    (data/'frame_times.json').write_text(json.dumps(table,indent=2))
    print(f'{clip}: {len(times)} source frames, span {times[-1][1]-t0:.3f}s',flush=True)
print('Inspection montage ready',flush=True)