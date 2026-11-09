"""Build an offline, all-recording review workspace; no source annotation edits."""
import hashlib
import json
import shutil

import cv2
import numpy as np

from build_field_core_dataset import ROOT, read, save, recover_dimensions
from visualize_field_baseline import project
from sync_applied_review import annotate_manifest


def main():
    cv2.setNumThreads(2)
    cfg=read(ROOT/'configs/field_leader_review.json')
    out=ROOT/'experiments'/cfg['experiment']; (out/'data').mkdir(parents=True,exist_ok=True)
    survey=read(ROOT/'configs/field_full_survey.json')
    priority={r['clip']:r for r in cfg['priority_cases']}
    rows=[]; missing_images=[]; total_frames=0; total_boxes=0
    for rec in survey['clips']:
        c=rec['clip']
        model='adapted_V' if 'adapted_V' in rec['models'] else 'adapted_S' if 'adapted_S' in rec['models'] else 'original'
        tag=survey['models'][model]['tag']
        td=ROOT/'trajectory/outputs/trajectories'/f'{c}_{tag}'
        dd=ROOT/'trajectory/outputs/object_detection/camera-data'/f'{c}_{tag}'
        ev=read(ROOT/'experiments/field_full_survey_20261101/runs'/f'{c}_{model}'/'evaluation.json')
        ts=read(ROOT/'trajectory/data/camera-data'/c/'frame_times.json')['frames']
        tracks=read(td/'tracks.json')['tracks'] if (td/'tracks.json').exists() else {}
        dimension_cache=out/'data'/f'{c}_dimensions.json'
        if dimension_cache.exists():dims=read(dimension_cache)
        else:
            dims=recover_dimensions(tracks,dd.with_name(dd.name+'_road'));save(dimension_cache,dims)
        cal=read(dd/'calibration_used.json');K,M=np.array(cal['K']),np.array(cal['lidar2cam'])
        frames=[dict(frame=t['frame'],t_s=t['t_s'],boxes=[]) for t in ts]
        byframe={f['frame']:f for f in frames}
        for k,ss in tracks.items():
            if k not in dims:continue
            di=dims[k]
            for s in ss:
                if s['frame'] not in byframe:continue
                uv=project(dict(s,l=di['length_m'],w=di['width_m'],h=di['height_m']),K,M)
                if uv is None:continue
                bbox=np.r_[uv.min(0),uv.max(0)]
                if bbox[2]<0 or bbox[3]<0 or bbox[0]>1920 or bbox[1]>1080:continue
                byframe[s['frame']]['boxes'].append(dict(id=k,bbox=bbox.round(1).tolist(),kind='track',observed='x_det' in s))
        # Heavy detections have no car-MOT IDs. Expose them for human selection,
        # with per-frame detection IDs, never suggesting temporal identity.
        for f in frames:
            fp=ROOT/'trajectory/data/camera-data'/c/'frames_all'/f'{f["frame"]:03d}.jpg'
            if not fp.exists():missing_images.append(str(fp.relative_to(ROOT)))
            detfile=dd/f'{f["frame"]:03d}_pred.json'
            if detfile.exists():
                for i,d in enumerate(read(detfile)):
                    if d['class_name'] not in ('truck','bus') or d['score']<.2:continue
                    uv=project(d,K,M)
                    if uv is None:continue
                    bbox=np.r_[uv.min(0),uv.max(0)]
                    if bbox[2]<0 or bbox[3]<0 or bbox[0]>1920 or bbox[1]>1080:continue
                    f['boxes'].append(dict(id=f'det:{f["frame"]}:{i}',bbox=bbox.round(1).tolist(),kind=d['class_name'],observed=True))
        proposal=priority.get(c,{})
        target=ev.get('identified_target')
        if target is None and isinstance(ev.get('identity'),dict):target=ev['identity'].get('vehicle_id')
        target=str(target) if target is not None else None
        st=tracks.get(target,[])
        default_range=[st[0]['frame'],st[-1]['frame']] if st else [0,frames[-1]['frame']]
        case_range=proposal.get('range',default_range)
        frame=proposal.get('frame',int(sum(case_range)/2))
        chosen_frames=proposal.get('checkpoints',np.linspace(case_range[0],case_range[1],6).astype(int).tolist())
        flowpath=out/'targeted_tracking'/c/'image_track.json'
        if flowpath.exists():
            for r in read(flowpath)['rows']:
                byframe[r['frame']]['boxes'].append(dict(id='image-candidate',bbox=np.round(r['bbox'],1).tolist(),kind='image',observed=False))
        source_hash=hashlib.sha256((td/'tracks.json').read_bytes()).hexdigest() if tracks else None
        payload=dict(clip=c,tag=tag,frames=frames,tracks_sha256=source_hash,
                     width=1920,height=1080,source_video_sha256=rec['video_sha256'])
        (out/'data'/f'{c}.js').write_text('window.registerClip('+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+');',encoding='utf-8')
        item=dict(clip=c,core=rec['core'],site=rec['site'],view=rec['view'],tag=tag,
                  id=proposal.get('id',''),priority=bool(proposal),target_id=target,
                  target_identity_accepted=bool(ev.get('identity_accepted',False)),
                  leader_id=proposal.get('leader_id'),range=case_range,frame=frame,
                  checkpoints=[f for f in chosen_frames if f in byframe],
                  frames=len(frames),video_path='../../'+rec['video_path'],
                  question=proposal.get('question','请确认自车身份与直接前车；只有在同车道、没有其他车辆插入的区间内保存配对。'),
                  initial_assessment=proposal.get('initial_assessment','本轮尚未逐段人工复核；候选框来自既有检测和跟踪。'),
                  automatic_pair_points=ev.get('following_observations',0),status=ev.get('status'),
                  source_video_sha256=rec['video_sha256'],tracks_sha256=source_hash)
        rows.append(item);total_frames+=len(frames);total_boxes+=sum(len(f['boxes']) for f in frames)
        print('REVIEW_DATA',c,len(frames),flush=True)
    rows.sort(key=lambda r:(not r['priority'],r['id'],not r['core'],r['clip']))
    manifest=dict(schema='field-leader-review/v1',pack_id=cfg['experiment'],cases=rows,
                  priority_count=len(priority),core_count=sum(r['core'] for r in rows),all_count=len(rows),
                  reviewed_source_sheets=cfg['reviewed_source_sheets'])
    manifest=annotate_manifest(manifest)
    save(out/'manifest.json',manifest)
    (out/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';',encoding='utf-8')
    template=ROOT/'review_web/public'
    for name in ['index.html','app.js','style.css']:shutil.copyfile(template/name,out/name)
    assert not missing_images, missing_images[:10]
    save(out/'build_validation.json',dict(status='PASS',clips=len(rows),native_frames=total_frames,
         displayed_candidate_boxes=total_boxes,missing_source_images=missing_images,
         original_clips_and_tracks_not_modified=True,review_decisions_initialized=False))
    print('REVIEW_READY',len(rows),total_frames,total_boxes)


if __name__=='__main__':main()
