"""Second adaptation pilot: shared detector features, class-aware weak labels.

Preserves the first pilot, and never invents 3D supervision from a 2D box.
"""
import argparse,hashlib,json,os,random,sys
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
import adapt_new_views as A
from verify_field_branches import ROOT,TR,read,save
from scripts.finetune import train_finetune as T
from scripts.object_detection import run_bevheight_generic as G

CFG=read(ROOT/'configs/view_refinement.json');OUT=ROOT/'experiments'/CFG['experiment'];TC=CFG['training']
A.CFG=CFG;A.OUT=OUT

def labels():
    A.labels()
    for c,boxes in TC['negative_boxes_px'].items():
        seed=CFG['marker_seeds'][c]['frame'];ims=[]
        for f in [seed-25,seed,seed+15]:
            im=cv2.imread(str(A.framepath(c,f)))
            for x0,y0,x1,y1 in boxes:cv2.rectangle(im,(x0,y0),(x1,y1),(0,0,255),3)
            cv2.putText(im,f'{c} f{f} empty region',(30,50),0,1,(0,0,255),2);ims.append(cv2.resize(im,(960,540)))
        cv2.imwrite(str(OUT/f'{c}_negative_review.jpg'),np.vstack(ims))

def trainable(name):return name.startswith('head.shared_conv.') or (name.startswith('head.task_heads.') and '.heatmap.' in name)

def train():
    random.seed(TC['seed']);np.random.seed(TC['seed']);torch.manual_seed(TC['seed']);torch.cuda.manual_seed_all(TC['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    assert not set(sum(CFG['train'].values(),[]))&set(sum(CFG['development'].values(),[]))
    rows=read(OUT/'training_annotations.json')['annotations'];cache_dir=OUT/'feature_cache';cache_dir.mkdir(exist_ok=True)
    model=T.build_model();info=T.load_checkpoint(model,TR/CFG['base_checkpoint']);assert not info['missing'] and not info['unexpected'];model.cuda().eval()
    original={k:v.cpu().clone() for k,v in model.state_dict().items()}
    for name,p in model.named_parameters():p.requires_grad_(trainable(name))
    captured={};hook=model.head.shared_conv.register_forward_pre_hook(lambda m,args:captured.update(feature=args[0].detach().cpu().half()))
    paths=[]
    for row in rows:
        c,f=row['clip'],row['frame'];K,M=A.calibration(c);img,mats,_=T.build_mats_dict(str(A.framepath(c,f)),K,M,T.final_dim,T.img_conf)
        with torch.no_grad():preds=model(img.cuda(),{k:v.cuda() if torch.is_tensor(v) else v for k,v in mats.items()})
        teacher=[{k:v.detach().cpu() for k,v in p[0].items()} for p in preds]
        shape=teacher[0]['heatmap'].shape[-2:];pos=A.positive_mask(model,K,M,row['body_box'],shape);neg=torch.zeros_like(pos)
        for box in TC['negative_boxes_px'][c]:neg|=A.positive_mask(model,K,M,box,shape)
        assert pos.any() and neg.any() and not (pos&neg).any()
        p=cache_dir/f'{c}_{f:03d}.pt';torch.save(dict(feature=captured['feature'],teacher=teacher,pos=pos,neg=neg,row=row),p);paths.append(p)
        print('CACHE',c,f,'positive',int(pos.sum()),'negative',int(neg.sum()),flush=True)
    hook.remove()
    groups=[dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('head.shared_conv.')],lr=TC['shared_learning_rate']),dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('head.shared_conv.')],lr=TC['learning_rate'])]
    opt=torch.optim.AdamW(groups,weight_decay=TC['weight_decay']);history=[]
    for epoch in range(1,TC['epochs']+1):
        random.shuffle(paths);stats=[]
        for path in paths:
            s=torch.load(path,map_location='cpu');x=s['feature'].cuda().float();pos=s['pos'].cuda();neg=s['neg'].cuda();x=model.head.shared_conv(x)
            preds=[h(x) for h in model.head.task_heads];teachers=[{k:v.cuda() for k,v in t.items()} for t in s['teacher']]
            positive=F.softplus(-preds[0]['heatmap'][pos].max());wrong=[];negative=[];preservation=[];geometry=[]
            for i,(p,t) in enumerate(zip(preds,teachers)):
                logits=p['heatmap'];mpos=pos.expand_as(logits);mneg=neg.expand_as(logits);prob=t['heatmap'].sigmoid()
                if i:wrong.append(F.softplus(logits[mpos]).mean())
                negative.append(F.softplus(logits[mneg]).mean())
                kl=F.binary_cross_entropy_with_logits(logits,prob,reduction='none')-F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),prob,reduction='none')
                for mask in [(~mpos)&(~mneg)&(prob>=.1),(~mpos)&(~mneg)&(prob<.1)]:
                    if mask.any():preservation.append(kl[mask].clamp_min(0).mean())
                gm=(prob.max(1,keepdim=True).values>=.1)|pos
                for k in ['reg','height','dim','rot','vel']:
                    mm=gm.expand_as(p[k]);geometry.append(F.smooth_l1_loss(p[k][mm],t[k][mm]))
            wrong_loss=torch.stack(wrong).mean();neg_loss=torch.stack(negative).mean();teacher_loss=torch.stack(preservation).mean();reg_loss=torch.stack(geometry).mean()
            loss=positive+wrong_loss+TC['negative_weight']*neg_loss+TC['teacher_weight']*teacher_loss+TC['regression_preservation_weight']*reg_loss
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5);opt.step()
            stats.append([float(v.detach()) for v in [loss,positive,wrong_loss,neg_loss,teacher_loss,reg_loss]])
        hist=dict(zip(['total','car_presence','noncar_in_car','empty_region','teacher_kl','regression_preservation'],np.mean(stats,0).tolist()));hist['epoch']=epoch;history.append(hist);print('TRAIN',hist,flush=True)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),original[k])];assert changed and all(trainable(k) for k in changed)
    meta=dict(config=CFG,train_frames=len(rows),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),changed_keys=changed,history=history,geometry_parameters_frozen=True,geometry_output_invariance_not_guaranteed=True)
    dst=OUT/'V_shared_class_ep5.ckpt';torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},view_refinement=meta),dst);save(OUT/'training.json',meta);print('SAVED',dst,flush=True)

def infer():
    import subprocess,time
    base=read(ROOT/CFG['evaluation']['baseline_config']);model=T.build_model();ckpt=OUT/'V_shared_class_ep5.ckpt';info=T.load_checkpoint(model,ckpt);assert not info['missing'] and not info['unexpected'];model.cuda().eval();exp=G.load_exp(G.DEFAULT_CONFIG)
    sha=hashlib.sha256(ckpt.read_bytes()).hexdigest();runs=[]
    for c in CFG['development']['V']:
        row=next(r for r in base['excerpts'] if r['clip']==c);old=TR/'outputs/object_detection/camera-data'/f'{c}_{base["tag"]}';cal=read(old/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
        dd=TR/'outputs/object_detection/camera-data'/f'{c}_{CFG["tag"]}';dd.mkdir(exist_ok=True)
        if any(dd.glob('*_pred.json')):assert read(dd/'calibration_used.json')['checkpoint_sha256']==sha,'Use a new tag for changed weights'
        cal.update(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha,adaptation_config='configs/view_refinement.json');save(dd/'calibration_used.json',cal)
        for f in range(row['start'],row['end']+1):
            p=dd/f'{f:03d}_pred.json'
            if not p.exists():save(p,G.run_frame(model,A.framepath(c,f),K,M,exp,score_thresh=.2))
            if (f-row['start'])%40==0:print('INFER',c,f,flush=True)
        cmd=[sys.executable,'scripts/pipeline/extract_trajectories.py','--clip',c,'--tag',CFG['tag'],'--extrinsic','metric_extrinsic_baseline_h15.json','--score-thresh','0.2','--track-low-thresh','0.2']
        with (OUT/f'{c}_tracking.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=TR,env=dict(os.environ,NUMBA_DISABLE_JIT='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
        centers=A.marker_centres(ROOT/CFG['evaluation']['baseline_directory']/f'{c}_markers.json')[0];ident=A.identify(c,centers,dict(tag=CFG['tag'],marker_seeds=base['marker_seeds']),OUT);runs.append(dict(row,tag=CFG['tag'],identity=ident));print('IDENTITY',ident,flush=True)
    save(OUT/'runs.json',dict(runs=runs))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['labels','train','infer','evaluate']);a=ap.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    A.evaluate() if a.stage=='evaluate' else globals()[a.stage]()
