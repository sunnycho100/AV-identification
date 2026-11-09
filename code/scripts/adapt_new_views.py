"""Conservative, recording-disjoint car-presence adaptation for Verona/Seminole.

Image-reviewed hood marks locate the instrumented car; propagated body boxes
are weak 2D labels, NOT metric 3D ground truth. Reuses the original BEVHeight
loader/calibration and the principle of partial supervision from gps_loss,
without assigning a GPS label or a guessed regression target. Keeps the full
original checkpoint intact and saves a separate checkpoint for each view.
"""
import argparse, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from verify_field_branches import ROOT, TR, read, save
from scripts.object_detection import run_bevheight_generic as G
from scripts.finetune import train_finetune as T
from identify_baseline_targets import markers, identify, marker_centres
from build_field_core_dataset import corners8, convert

CFG=read(ROOT/'configs/view_adaptation.json')
OUT=ROOT/'experiments'/CFG['experiment']
os.environ.setdefault('TORCH_HOME',str(ROOT/'.runtime/torch-cache'))

def calibration(clip):
    cal=TR/'outputs/calibration/camera-data'/clip
    return (T.load_K_from_anycalib(cal/'150_anycalib_pinhole_pinhole.json'),
            G.to_dair_ground(T.load_extrinsic_json(cal/'metric_extrinsic_baseline_h15.json')))

def framepath(c,f): return TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'

def labels():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]
    for site,clips in CFG['train'].items():
        for c in clips:
            seed=CFG['marker_seeds'][c]
            n=len(list(framepath(c,0).parent.glob('*.jpg')))
            centers=markers(c,seed,{'excerpts':[dict(clip=c,start=0,end=n-1)]},OUT)
            pf=read(OUT/f'{c}_markers.json')['per_frame']; ff=sorted(centers)
            selected=[ff[i] for i in np.unique(np.linspace(0,len(ff)-1,min(len(ff),CFG['training']['frames_per_clip'])).astype(int))]
            body=np.array(seed['body_box'],float); hood=np.array(seed['box'],float)
            tiles=[]
            for f in selected:
                m=pf[str(f)]; scale=np.array([m['w']/(hood[2]-hood[0]),m['h']/(hood[3]-hood[1])]*2)
                box=(body-np.tile(hood[:2],2))*scale+np.array([m['x'],m['y']]*2)
                row=dict(clip=c,site=site,frame=f,body_box=box.tolist(),marker_xy=list(map(float,centers[f])),marker_score=m['score'],label_type='image-reviewed seed with local template propagation; weak 2D car-presence supervision')
                rows.append(row)
                im=cv2.imread(str(framepath(c,f))); a,b,d,e=box.astype(int);cv2.rectangle(im,(a,b),(d,e),(0,255,0),2)
                u,v=map(int,centers[f]);cv2.circle(im,(u,v),4,(0,255,255),-1)
                left=max(0,min(1920-360,u-180));top=max(0,min(1080-260,v-160))
                tile=im[top:top+260,left:left+360].copy();cv2.putText(tile,f'{c} f{f} ({m["score"]:.2f})',(5,20),0,.5,(0,255,255),1);tiles.append(tile)
            while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(OUT/f'{c}_training_labels.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
            print('LABELS',c,len(selected),'frames',selected,flush=True)
    save(OUT/'training_annotations.json',dict(provenance='Reviewed source images; no GPS or detector class predictions used to assign car presence',annotations=rows))

def positive_mask(model,K,M,box,shape):
    cfg=model.head.train_cfg; cell=cfg['voxel_size'][0]*cfg['out_size_factor'];H,W=shape
    y,x=np.mgrid[:H,:W];xyz=np.stack([cfg['point_cloud_range'][0]+(x+.5)*cell,cfg['point_cloud_range'][1]+(y+.5)*cell,np.full_like(x,CFG['training']['projection_z_m'],dtype=float)],-1)
    cam=xyz@M[:3,:3].T+M[:3,3];q=cam@K.T;uv=q[...,:2]/np.maximum(q[...,2:],1e-8)
    x0,y0,x1,y1=box;dx=(x1-x0)*CFG['training']['body_inset_fraction'];dy=(y1-y0)*CFG['training']['body_inset_fraction']
    mask=(cam[...,2]>0)&(uv[...,0]>=x0+dx)&(uv[...,0]<=x1-dx)&(uv[...,1]>=y0+dy)&(uv[...,1]<=y1-dy)
    if not mask.any():raise ValueError('No BEV cells project into target box')
    return torch.from_numpy(mask)[None,None]

def train():
    tc=CFG['training'];random.seed(tc['seed']);np.random.seed(tc['seed']);torch.manual_seed(tc['seed']);torch.cuda.manual_seed_all(tc['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    train_clips=sum(CFG['train'].values(),[]);dev_clips=sum(CFG['development'].values(),[])
    assert not set(train_clips)&set(dev_clips)
    annotations=read(OUT/'training_annotations.json')['annotations'];summaries=[]
    for site,clips in CFG['train'].items():
        model=T.build_model(); info=T.load_checkpoint(model,TR/CFG['base_checkpoint']); assert not info['missing'] and not info['unexpected']
        model.cuda().eval()
        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(tc['trainable_prefix']))
        branch=model.head.task_heads[0].heatmap;cache=[];captured={}
        hook=branch.register_forward_pre_hook(lambda module,args:captured.update(feature=args[0].detach().cpu().clone()))
        for row in [r for r in annotations if r['clip'] in clips]:
            K,M=calibration(row['clip']);img,mats,_=T.build_mats_dict(str(framepath(row['clip'],row['frame'])),K,M,T.final_dim,T.img_conf)
            mats={k:v.cuda() if torch.is_tensor(v) else v for k,v in mats.items()}
            with torch.no_grad(): logits=model(img.cuda(),mats)[0][0]['heatmap'].detach().cpu().clone()
            mask=positive_mask(model,K,M,row['body_box'],logits.shape[-2:]);assert mask.sum()<mask.numel()//10
            cache.append(dict(row=row,feature=captured['feature'],teacher=logits,mask=mask))
            print('CACHE',site,row['clip'],row['frame'],'cells',int(mask.sum()),'p',float(logits.sigmoid()[mask].max()),flush=True)
        hook.remove()
        opt=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
        history=[];order=list(range(len(cache)))
        for epoch in range(1,tc['epochs']+1):
            random.shuffle(order); losses=[]
            for idx in order:
                sample=cache[idx];x=sample['feature'].cuda();target=sample['teacher'].cuda().sigmoid();mask=sample['mask'].cuda()
                pred=branch(x);positive=F.softplus(-pred[mask].max())
                # Teacher KL up to a constant, balanced between existing confident
                # car cells and the remaining cells. Unknown objects are not hard negatives.
                per=F.binary_cross_entropy_with_logits(pred,target,reduction='none')
                ent=F.binary_cross_entropy(target.clamp(1e-6,1-1e-6),target,reduction='none')
                kl=(per-ent).clamp_min(0);groups=[(~mask)&(target>=.1),(~mask)&(target<.1)]
                preserve=sum(kl[g].mean() for g in groups if g.any())
                loss=positive+tc['teacher_weight']*preserve
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(branch.parameters(),5);opt.step()
                losses.append([float(loss.detach()),float(positive.detach()),float(preserve.detach())])
            means=np.mean(losses,0).tolist();history.append(dict(epoch=epoch,loss=means[0],positive_loss=means[1],teacher_kl=means[2]));print('TRAIN',site,history[-1],flush=True)
        diagnostic=[]
        with torch.no_grad():
            for s in cache:
                p=branch(s['feature'].cuda()).sigmoid().cpu();m=s['mask'];diagnostic.append(dict(clip=s['row']['clip'],frame=s['row']['frame'],positive_cells=int(m.sum()),baseline_peak=float(s['teacher'].sigmoid()[m].max()),adapted_peak=float(p[m].max())))
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),original[k])]
        assert changed and all(k.startswith(tc['trainable_prefix']) for k in changed),changed
        meta=dict(site=site,train_clips=clips,development_clips=CFG['development'][site],training_frames=len(cache),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),changed_keys=changed,geometry_and_other_classes_unchanged=True,history=history,train_presence=diagnostic,config=CFG)
        path=OUT/f'{site}_car_presence_ep{tc["epochs"]}.ckpt'
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},view_adaptation=meta),path)
        save(OUT/f'{site}_training.json',meta);summaries.append(dict(site=site,checkpoint=str(path.relative_to(ROOT)),**{k:meta[k] for k in ['training_frames','trainable_parameters','changed_keys','history']}))
        print('SAVED',path,flush=True);del model,cache,original;torch.cuda.empty_cache()
    save(OUT/'training_summary.json',dict(recording_disjoint=True,sites=summaries))

def infer():
    baseline=read(ROOT/CFG['evaluation']['baseline_config']);env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_DISABLE_JIT='1');runs=[]
    for site,clips in CFG['development'].items():
        model=T.build_model();path=OUT/f'{site}_car_presence_ep{CFG["training"]["epochs"]}.ckpt'
        info=T.load_checkpoint(model,path);assert not info['missing'] and not info['unexpected'];model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG)
        for clip in clips:
            row=next(r for r in baseline['excerpts'] if r['clip']==clip);base=TR/'outputs/object_detection/camera-data'/f'{clip}_{baseline["tag"]}'
            cal=read(base/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
            dst=TR/'outputs/object_detection/camera-data'/f'{clip}_{CFG["tag"]}';dst.mkdir(parents=True,exist_ok=True);start=time.time()
            for f in range(row['start'],row['end']+1):
                p=dst/f'{f:03d}_pred.json'
                if not p.exists():save(p,G.run_frame(model,framepath(clip,f),K,M,exp,score_thresh=CFG['evaluation']['score_thresh']))
                if (f-row['start'])%40==0:print('INFER',clip,f,flush=True)
            cal.update(checkpoint=str(path.relative_to(ROOT)),adaptation_config='configs/view_adaptation.json');save(dst/'calibration_used.json',cal)
            cmd=[sys.executable,'scripts/pipeline/extract_trajectories.py','--clip',clip,'--tag',CFG['tag'],'--extrinsic','metric_extrinsic_baseline_h15.json','--score-thresh','0.2','--track-low-thresh','0.2']
            with (OUT/f'{clip}_tracking.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=TR,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            centers=marker_centres(ROOT/CFG['evaluation']['baseline_directory']/f'{clip}_markers.json')[0]
            identity=identify(clip,centers,dict(tag=CFG['tag'],marker_seeds=baseline['marker_seeds']),OUT)
            runs.append(dict(row,tag=CFG['tag'],runtime_s=time.time()-start,identity=identity));save(OUT/'runs.json',dict(runs=runs));print('TRACKED',clip,identity,flush=True)
        del model;torch.cuda.empty_cache()

def contains(det,point,K,M):
    xyz=corners8([det['l'],det['w'],det['h']],det['yaw'],[det['x'],det['y'],det['z']]);cam=xyz@M[:3,:3].T+M[:3,3]
    if (cam[:,2]<=0).any():return False
    q=cam@K.T;uv=q[:,:2]/q[:,2:]
    return cv2.pointPolygonTest(cv2.convexHull(uv.astype(np.float32)),tuple(map(float,point)),True)>=-CFG['evaluation']['marker_margin_px']

def review_comparison(c,centers,baseline):
    """Fixed first/middle/last marker frames, so illustrations are not cherry-picked."""
    ff=sorted(centers);tiles=[]
    for f in [ff[0],ff[len(ff)//2],ff[-1]]:
        point=centers[f];u,v=map(int,point);left=max(0,min(1920-720,u-360));top=max(0,min(1080-440,v-220));pair=[]
        for kind,tag in [('Baseline',baseline['tag']),('Adapted',CFG['tag'])]:
            dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}';cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);im=cv2.imread(str(framepath(c,f)))
            for d in read(dd/f'{f:03d}_pred.json'):
                if d['class_id']!=0 and not contains(d,point,K,M):continue
                xyz=corners8([d['l'],d['w'],d['h']],d['yaw'],[d['x'],d['y'],d['z']]);cam=xyz@M[:3,:3].T+M[:3,3]
                if (cam[:,2]<=0).any():continue
                q=cam@K.T;uv=(q[:,:2]/q[:,2:]).astype(np.int32);color=(0,230,0) if d['class_id']==0 else (0,140,255)
                for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:cv2.line(im,tuple(uv[a]),tuple(uv[b]),color,2)
                a,b=uv.min(0);cv2.putText(im,f'{d["class_name"]} {d["score"]:.2f}',(max(0,a),max(15,b-4)),0,.55,color,2)
            cv2.drawMarker(im,(u,v),(0,255,255),cv2.MARKER_CROSS,14,2)
            tile=im[top:top+440,left:left+720].copy();tile[:35]=0;cv2.putText(tile,f'{kind}  {c}  frame {f}',(10,24),0,.6,(255,255,255),1);pair.append(tile)
        tiles.append(np.hstack(pair))
    cv2.imwrite(str(OUT/f'{c}_before_after.jpg'),np.vstack(tiles))

def continuity(states):
    observed=sorted(s['frame'] for s in states if s.get('x_det') is not None)
    runs=[]
    for f in observed:
        if not runs or f!=runs[-1][-1]+1:runs.append([])
        runs[-1].append(f)
    return dict(observed_runs=len(runs),longest_observed_run_frames=max(map(len,runs),default=0),internal_missing_frames=(observed[-1]-observed[0]+1-len(observed)) if observed else 0)

def evaluate():
    baseline=read(ROOT/CFG['evaluation']['baseline_config']);rows=[]
    for clips in CFG['development'].values():
        for c in clips:
            centers=marker_centres(ROOT/CFG['evaluation']['baseline_directory']/f'{c}_markers.json')[0];result=dict(clip=c,marker_frames=len(centers))
            for kind,tag in [('baseline',baseline['tag']),('adapted',CFG['tag'])]:
                dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}';cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);frame_rows=[]
                for f,point in sorted(centers.items()):
                    dets=read(dd/f'{f:03d}_pred.json');cars=[d for d in dets if d['class_id']==0];matched=[d for d in cars if contains(d,point,K,M)];scores=[d['score'] for d in matched]
                    frame_rows.append(dict(frame=f,car_present=bool(scores),car_birth=bool(scores and max(scores)>=.45),best_score=max(scores) if scores else None,total_cars=len(cars),off_target_cars=len(cars)-len(matched)))
                td=TR/'outputs/trajectories'/f'{c}_{tag}';ident=read(td/'instrumented.json');tracks=read(td/'tracks.json')['tracks'];states=tracks.get(str(ident['vehicle_id']),[]) if ident['accepted'] else []
                result[kind]=dict(marker_car_frames=sum(x['car_present'] for x in frame_rows),marker_birth_frames=sum(x['car_birth'] for x in frame_rows),mean_off_target_cars=float(np.mean([x['off_target_cars'] for x in frame_rows])),identity=ident,target_states=len(states),target_observed_states=sum(s.get('x_det') is not None for s in states),target_first=states[0]['frame'] if states else None,target_last=states[-1]['frame'] if states else None,continuity=continuity(states),frames=frame_rows)
            review_comparison(c,centers,baseline)
            rows.append(result);print('COMPARE',c,{k:{a:b for a,b in result[k].items() if a not in ['identity','frames']} for k in ['baseline','adapted']},flush=True)
    save(OUT/'comparison.json',dict(note=CFG['evaluation']['note'],rows=rows))
    reports=[];windows=[]
    for clips in CFG['development'].values():
        for c in clips:
            report,ww=convert(dict(clip=c,tag=CFG['tag'],role='adaptation_development'),baseline,OUT);reports.append(report);windows+=ww
    save(OUT/'dataset_summary.json',dict(processed_runs=reports,note='Same target/leader validity and window rules as the frozen baseline; no classifier training or accuracy claim.'))
    for dur in baseline['classification']['window_durations_s']:
        selected=[(x,m) for x,m in windows if m['duration_s']==dur]
        raw=np.stack([x for x,_ in selected]) if selected else np.empty((0,int(round(dur*10)),3),np.float32)
        np.savez_compressed(OUT/f'windows_{dur:g}s.npz',primitive_windows=raw,labels=np.array([m['label'] for _,m in selected],np.int64),dt=np.array(.1),split_group=np.array([m['clip'] for _,m in selected],dtype=str))
        save(OUT/f'windows_{dur:g}s_metadata.json',[dict(m,window_index=i) for i,(_,m) in enumerate(selected)])

def audit():
    """Check actual saved weights, frame coverage, and unchanged evaluation inputs."""
    summary=[];base=read(ROOT/CFG['evaluation']['baseline_config']);nframes=0
    cp=torch.load(TR/CFG['base_checkpoint'],map_location='cpu')['state_dict'];original={k[6:] if k.startswith('model.') else k:v for k,v in cp.items()}
    assert not set(sum(CFG['train'].values(),[]))&set(sum(CFG['development'].values(),[]))
    for site,clips in CFG['development'].items():
        path=OUT/f'{site}_car_presence_ep{CFG["training"]["epochs"]}.ckpt';saved=torch.load(path,map_location='cpu');state=saved['state_dict']
        assert set(state)==set(original)
        changed=[k for k,v in state.items() if not torch.equal(v,original[k])]
        assert changed and all(k.startswith(CFG['training']['trainable_prefix']) for k in changed)
        assert all(torch.isfinite(v).all() for v in state.values())
        summary.append(dict(site=site,checkpoint=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changed_keys=changed))
        for c in clips:
            old=TR/'outputs/object_detection/camera-data'/f'{c}_{base["tag"]}';new=TR/'outputs/object_detection/camera-data'/f'{c}_{CFG["tag"]}'
            row=next(r for r in base['excerpts'] if r['clip']==c);expected=set(range(row['start'],row['end']+1));actual={int(p.stem.split('_')[0]) for p in new.glob('*_pred.json')};assert expected==actual
            nframes+=len(actual)
            for k in ['K','lidar2cam','road_plane_z_in_output']:assert read(old/'calibration_used.json')[k]==read(new/'calibration_used.json')[k]
            old_run=read(TR/'outputs/trajectories'/f'{c}_{base["tag"]}'/'run.json');new_run=read(TR/'outputs/trajectories'/f'{c}_{CFG["tag"]}'/'run.json')
            assert {k:v for k,v in old_run['settings'].items() if k!='tag'}=={k:v for k,v in new_run['settings'].items() if k!='tag'}
            image=cv2.imread(str(OUT/f'{c}_before_after.jpg'));assert image is not None and image.shape==(1320,1440,3)
    for dur in base['classification']['window_durations_s']:
        with np.load(OUT/f'windows_{dur:g}s.npz') as a:
            raw=a['primitive_windows'];assert raw.shape[1:]==(int(round(dur*10)),3) and np.isfinite(raw).all();assert len(raw)==len(a['labels'])
    result=dict(status='PASS',development_source_frames=nframes,training_frames=len(read(OUT/'training_annotations.json')['annotations']),weights=summary,checks=['recording-disjoint adaptation/development','saved checkpoints contain finite tensors and exactly the original keys','only car heatmap branch tensors changed','original calibration and tracker settings identical','all four contiguous inference excerpts complete','fixed-frame comparison images decode','exported windows finite with expected shapes'],scientific_limit='Engineering validation, not proof of trajectory accuracy or generalization.')
    save(OUT/'validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['labels','train','infer','evaluate','audit']);args=ap.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1);globals()[args.stage]()
