"""Bounded GPU integration checks for existing BEVHeight research variants.

All adaptation uses the frozen pilot's two training recordings. Evaluation
losses are pseudo-label consistency only. No test recording trains a branch.
"""
import argparse, gc, json, random, traceback
import cv2
import numpy as np
import torch
from verify_field_branches import ROOT, TR, OUT, CFG, read, save, calibration, frames, sample
from scripts.finetune import train_finetune as T
from scripts.object_detection import run_bevheight_generic as G
from scripts.orientation import virtual_camera as VC, bn_recalib as BN

torch.set_num_threads(4)
BASE=TR/CFG['baseline_checkpoint']

def model_for(checkpoint=BASE):
    model=T.build_model();info=T.load_checkpoint(model,checkpoint)
    assert not info['missing'] and not info['unexpected'],info
    return model.cuda().eval()

def load(c,f,mode='pseudo',feature=False,aug=False,v2=False):
    K,M=calibration(c)
    img,mats,_=T.build_mats_dict(str(TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'),K,M,T.final_dim,T.img_conf)
    if feature:mats['feat_mask']=torch.from_numpy(np.load(OUT/'geometry'/c/'feature_mask.npy'))[None,None]
    if mode=='pseudo':
        path=(OUT/'labels_v2' if v2 else TR/'outputs/finetune/pilot_labels')/c/f'{f:03d}_label.json'
        objs=read(path);boxes=T.to_boxes(objs)
        if aug:img,mats,boxes=T.augment(img,mats,boxes)
        return (*T.to_device(img,mats,'cuda'),boxes.cuda(),torch.zeros(len(boxes),dtype=torch.long,device='cuda'))
    objs=read(OUT/'gps_labels_enu'/c/f'{f:03d}_label.json')
    boxes={r:T.to_boxes([o for o in objs if o.get('role')==r]).cuda() for r in ['gps','other','ignore']}
    return (*T.to_device(img,mats,'cuda'),boxes)

def inference():
    rows=[];exp=G.load_exp(G.DEFAULT_CONFIG)
    chosen=[(c,int(p.stem)) for c in CFG['train']+CFG['validation'] for p in sample(frames(c),3)]
    model=model_for()
    for variant in ['baseline','input_mask','feature_mask','legacy_feature_checkpoint','virtual_camera']:
        checkpoint=TR/'checkpoints/ft102feat_run5_feat_ep5.ckpt' if variant=='legacy_feature_checkpoint' else BASE
        T.load_checkpoint(model,checkpoint);model.eval()
        out=OUT/'inference'/variant;out.mkdir(parents=True,exist_ok=True);counts=[]
        for c,f in chosen:
            K,M=calibration(c);p=TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'; fm=None
            if variant=='input_mask':
                mask=cv2.imread(str(OUT/'geometry'/c/'input_mask.png'),0)>0
                dst=out/f'{c}_{f:03d}.jpg';cv2.imwrite(str(dst),np.where(mask[...,None],cv2.imread(str(p)),np.array([104,116,124],np.uint8)));p=dst
            if variant in ['feature_mask','legacy_feature_checkpoint']:fm=np.load(OUT/'geometry'/c/'feature_mask.npy')
            if variant=='virtual_camera':
                box=VC.crop_box(1920,1080,K[0,0],2183.);K=VC.new_K(K,box,1920,1080)
                dst=out/f'{c}_{f:03d}.jpg';cv2.imwrite(str(dst),VC.warp(cv2.imread(str(p)),box,1920,1080));p=dst
            pred=G.run_frame(model,p,K,M,exp,score_thresh=.2,feat_mask=fm)
            assert all(np.isfinite([b[k] for k in ['x','y','z','l','w','h','yaw','score']]).all() for b in pred)
            save(out/f'{c}_{f:03d}_pred.json',pred)
            counts.append(dict(clip=c,frame=f,objects=len(pred),cars=sum(b['class_id']==0 for b in pred)))
        rows.append(dict(variant=variant,checkpoint=str(checkpoint.relative_to(TR)),frames=len(counts),car_detections=sum(r['cars'] for r in counts),counts=counts)); print(variant,rows[-1]['car_detections'],flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',rows=rows,note='12 identical sampled train/validation frames per variant, threshold 0.20; counts do not measure recall or precision. Virtual crop has a narrower field of view.')

def adaptation():
    results=[]
    for variant,mode in [('feature_aug_v2','pseudo'),('gps_only','gps_only'),('hybrid','hybrid')]:
        src=OUT/('labels_v2' if mode=='pseudo' else 'gps_labels_enu')
        train=[(c,int(p.stem.split('_')[0])) for c in CFG['train'] for p in sample(sorted((src/c).glob('*_label.json')),8)]
        if len(train)!=16:
            results.append(dict(variant=variant,status='NOT_TRAINED',reason='Insufficient eligible training labels after shared GPS registration',eligible_sampled_frames=len(train)))
            print('NOT_TRAINED',variant,results[-1]['reason'],flush=True)
            continue
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        model=model_for();params=T.freeze_image_branch(model)
        frozen={k:v.detach().cpu().clone() for k,v in model.backbone.state_dict().items()}
        optim=torch.optim.Adam(params,lr=2e-6)
        val=[(c,int(p.stem.split('_')[0])) for c in CFG['validation'] for p in sample(sorted((TR/'outputs/finetune/pilot_labels'/c).glob('*_label.json')),4)]
        def evaluate():
            losses=[]
            with torch.no_grad():
                for c,f in val:
                    img,mats,boxes,labels=load(c,f,feature=(mode=='pseudo'))
                    loss=T.car_only_loss(model,model.get_targets([boxes],[labels]),model(img,mats));losses.append(float(loss))
            return float(np.mean(losses))
        before=evaluate();steps=[]
        for c,f in train:
            optim.zero_grad()
            data=load(c,f,mode=mode,feature=(mode=='pseudo'),aug=(mode=='pseudo'),v2=(mode=='pseudo'))
            img,mats=data[:2];pred=model(img,mats)
            loss=T.car_only_loss(model,model.get_targets([data[2]],[data[3]]),pred) if mode=='pseudo' else T.gps_loss(model,pred,data[2],mode)
            assert torch.isfinite(loss),loss
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,5.)
            assert torch.isfinite(norm) and float(norm)>0,norm
            optim.step();steps.append(dict(clip=c,frame=f,loss=float(loss),gradient_norm=float(norm)))
        after=evaluate()
        assert all(torch.equal(v,model.backbone.state_dict()[k].cpu()) for k,v in frozen.items())
        checkpoint=OUT/'checkpoints'/f'{variant}_16steps.ckpt';checkpoint.parent.mkdir(exist_ok=True)
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},variant=variant,train_clips=CFG['train'],steps=steps,seed=0,lr=2e-6,validation_loss=after),checkpoint)
        results.append(dict(variant=variant,steps=steps,validation_frames=len(val),validation_before=before,validation_after=after,backbone_bit_identical=True,checkpoint=str(checkpoint.relative_to(ROOT))))
        save(OUT/f'adaptation_{variant}.json',results[-1]);print('adaptation',variant,before,after,flush=True)
        del model,optim,params,frozen;gc.collect();torch.cuda.empty_cache()
    return dict(status='PARTIAL' if any(r.get('status')=='NOT_TRAINED' for r in results) else 'PASS',rows=results,note='16 gradient updates for branches with eligible labels, 8 fixed validation frames. Different training targets make training losses incomparable. All validation values use original pseudo-labels; checkpoints are integration artifacts, not selected deployment weights.')

def batch_norm():
    model=model_for();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    bns=BN.prepare_bn(model);cal={c:calibration(c) for c in CFG['train']}
    # Every batch includes both clips so camera-vector BN has nonzero variation.
    picked={c:sample(frames(c),4) for c in CFG['train']}
    ff=[(picked[c][i],c) for i in range(4) for c in CFG['train']]
    with torch.no_grad():
        for img,mats in BN.batch_inputs(ff,cal):model(*T.to_device(img,mats,'cuda'))
    model.eval();after={k:v.cpu() for k,v in model.state_dict().items()};changed=BN.diff_keys(before,after)
    assert changed and not (changed-BN.bn_keys(model))
    out=OUT/'checkpoints/bn_8frames.ckpt';out.parent.mkdir(exist_ok=True);torch.save(dict(state_dict=after),out)
    exp=G.load_exp(G.DEFAULT_CONFIG);counts=[]
    for c in CFG['validation']:
        K,M=calibration(c)
        for p in sample(frames(c),2):
            pred=G.run_frame(model,p,K,M,exp,score_thresh=.2)
            save(OUT/'inference/bn'/f'{c}_{p.stem}_pred.json',pred);counts.append(len(pred))
    return dict(status='PASS',frames=8,batches=4,bn_layers=len(bns),buffers_changed=len(changed),all_learned_parameters_bit_identical=True,validation_detection_counts=counts,note='BN-only integration trial, not a validated replacement model.')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stages',nargs='+',default=['inference','adaptation','batch_norm']);a=ap.parse_args()
    for name in a.stages:
        print('START',name,flush=True)
        try:result=globals()[name]()
        except Exception as e:traceback.print_exc();result=dict(status='FAILED',error=str(e),traceback=traceback.format_exc())
        save(OUT/f'gpu_{name}.json',result);print('DONE',name,result['status'],flush=True)
