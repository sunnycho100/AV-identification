"""Read a named source frame for human annotation, independent of model output."""
import argparse
import cv2,numpy as np
from field_full_survey import ROOT,TR,OUT,CONFIG,read

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('requests',nargs='+',help='clip:frame or clip:frame:x0:y0:x1:y1');args=ap.parse_args();cv2.setNumThreads(1)
    cat={r['clip']:r for r in read(CONFIG)['clips']};out=OUT/'source_frames';out.mkdir(exist_ok=True)
    for req in args.requests:
        parts=req.split(':');c=parts[0];f=int(parts[1]);fp=TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'
        if fp.exists():im=cv2.imread(str(fp))
        else:
            cap=cv2.VideoCapture(str(ROOT/cat[c]['video_path']));cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();cap.release();assert ok
        if len(parts)>2:
            x0,y0,x1,y1=map(int,parts[2:]);im=im[y0:y1,x0:x1].copy();cv2.putText(im,f'{c} f{f} origin ({x0},{y0})',(6,24),0,.6,(0,255,255),1)
        cv2.imwrite(str(out/f'{c}_{f}.jpg'),im)
