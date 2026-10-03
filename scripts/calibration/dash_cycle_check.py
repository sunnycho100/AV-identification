"""Camera scale check from lane-line dashes: no GPS, no timing.

WisDOT lane lines are 12.5 ft of white with a 37.5 ft gap, a 15.24 m cycle
(TEOPS 3-2). Dashes are found in the clip's median background as LSD segments
that point at the lane vanishing point and are brighter than both sides (the
contrast tape's black tail is dropped), grouped into lane lines by lateral
position, and back-projected through each calibration. The along-road spacing
of neighbouring dashes, divided by 15.24 m, is the calibration's scale error;
how it changes with range shows a pitch error.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/calibration/dash_cycle_check.py AV_T_EW_3 out.jpg
Writes out.jpg (detected dashes in red, lateral position in yellow) and out.json.
"""
import json, glob, sys, numpy as np, cv2
CLIP, OUT = sys.argv[1], sys.argv[2]
cal=f"outputs/calibration/camera-data/{CLIP}/"
k=json.load(open(glob.glob(cal+"*_anycalib_pinhole_pinhole.json")[0]))["prediction"]["intrinsics"][:4]
K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]])
CAL={"h15.1_dp-0.31":("metric_extrinsic_h151_dpm031.json",15.1),"h15.6_dp0":("metric_extrinsic_site_h156.json",15.6)}
Rs={n:np.array(json.load(open(cal+f))["rotation"]) for n,(f,h) in CAL.items()}
def ground(R,h,uv):
    d=R.T@(np.linalg.inv(K)@np.c_[uv,np.ones(len(uv))].T); s=-h/d[2]; return np.c_[s*d[0],s*d[1]]
fr=sorted(glob.glob(f"data/camera-data/{CLIP}/frames_all/*.jpg"))[::10]
bgc=np.median(np.stack([cv2.imread(f) for f in fr]),0).astype(np.uint8)
g=cv2.cvtColor(bgc,cv2.COLOR_BGR2GRAY)
vp=K@Rs["h15.6_dp0"]@np.array([1.,0,0]); vp=vp[:2]/vp[2]
road=cv2.imread(cal+"road_mask.png",0)>0
segs=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0][:,0]
R0=Rs["h15.1_dp-0.31"]
keep=[]
for x1,y1,x2,y2 in segs:
    a,b=np.array([x1,y1]),np.array([x2,y2]); m=(a+b)/2
    if not road[int(m[1]),int(m[0])] or (m[0]<560 and m[1]>980) or m[1]<vp[1]+30: continue
    d=(b-a)/np.linalg.norm(b-a); n=np.array([-d[1],d[0]])
    if abs(n@(vp-a))>8+0.01*np.linalg.norm(m-vp): continue            # must point at the VP
    G=ground(R0,15.1,np.array([a,b]))
    if np.linalg.norm(G[1]-G[0])<1.5: continue
    keep.append((a,b,G))
# keep white paint only (contrast tape has a black tail): brighter than both sides
from scipy import ndimage
gf=g.astype(np.float32); W=[]
for a,b,G in keep:
    d=(b-a)/np.linalg.norm(b-a); n=np.array([-d[1],d[0]]); t=np.linspace(0.1,0.9,15); P=a+t[:,None]*(b-a)
    c=max(ndimage.map_coordinates(gf,[P[:,1]+o*n[1],P[:,0]+o*n[0]],order=1).mean() for o in (-3,-2,-1,0,1,2,3))
    side=max(ndimage.map_coordinates(gf,[P[:,1]+o*n[1],P[:,0]+o*n[0]],order=1).mean() for o in (-10,10))
    if c>side+8: W.append((a,b,G))
# one point per white piece (midpoint), cluster laterally, merge pieces of one dash
pts=np.array([[G[:,0].mean(),G[:,1].mean()] for _,_,G in W]); mids=[(a+b)/2 for a,b,_ in W]
o=np.argsort(pts[:,1]); groups=[]
for i in o:
    if groups and pts[i,1]-pts[groups[-1][-1],1]<0.45: groups[-1].append(i)
    else: groups.append([i])
vis=bgc.copy(); res={n:[] for n in CAL}
for grp in groups:
    if len(grp)<5: continue
    ylat=float(np.median(pts[grp,1]))
    ii=sorted(grp,key=lambda i:pts[i,0]); dash=[[ii[0]]]
    for i in ii[1:]:
        if pts[i,0]-pts[dash[-1][-1],0]<2.0: dash[-1].append(i)
        else: dash.append([i])
    if len(dash)<4: continue
    uvs=[np.mean([mids[i] for i in d],0) for d in dash]       # image point of each dash
    for d in dash:
        for i in d: cv2.line(vis,tuple(map(int,W[i][0])),tuple(map(int,W[i][1])),(0,0,255),3)
    cv2.putText(vis,f"{ylat:.1f}",tuple(map(int,uvs[0]+[8,0])),0,0.8,(0,255,255),2)
    for name,(f,h) in CAL.items():
        x=np.sort(ground(Rs[name],h,np.array(uvs))[:,0])
        P0=15.0; kk=np.round((x-x[0])/P0)                       # dash index (missing dashes allowed)
        for _ in range(3):
            A=np.c_[np.ones_like(kk),kk]; c=np.linalg.lstsq(A,x,rcond=None)[0]; kk=np.round((x-c[0])/c[1])
        for _ in range(3):                                      # drop dashes off the lattice (solid lines, wrong line)
            resid=x-(c[0]+c[1]*kk); good=np.abs(resid)<1.2
            if good.sum()<3: break
            x,kk=x[good],kk[good]; A=np.c_[np.ones_like(kk),kk]; c=np.linalg.lstsq(A,x,rcond=None)[0]
        resid=x-(c[0]+c[1]*kk)
        if len(x)<4 or not (13<c[1]<17.5): continue
        gaps=np.diff(x)/np.maximum(np.diff(kk),1); ok=np.diff(kk)>0
        res[name].append(dict(lateral=round(ylat,1),n=len(x),x=[round(float(v),1) for v in x],period=round(float(c[1]),3),
                              resid_rms=round(float(np.sqrt(np.mean(resid**2))),2),
                              cycles=[(round(float(a),1),round(float(b),2)) for a,b in zip(x[:-1][ok],gaps[ok])]))
cv2.imwrite(OUT,vis)
summ={}
for name,rows in res.items():
    rows=[r for r in rows if r["n"]>=4]
    A=np.array([c for r in rows for c in r["cycles"]]).reshape(-1,2)
    band=lambda lo,hi:(round(float(np.median(A[(A[:,0]>=lo)&(A[:,0]<hi),1])),2) if ((A[:,0]>=lo)&(A[:,0]<hi)).any() else None,int(((A[:,0]>=lo)&(A[:,0]<hi)).sum()))
    wmean=lambda R_:round(float(np.average([r["period"] for r in R_],weights=[r["n"]-1 for r in R_])),3) if R_ else None
    summ[name]=dict(lines=len(rows),period_all=wmean(rows),near_carriageway=wmean([r for r in rows if r["lateral"]>-20]),
                    far_carriageway=wmean([r for r in rows if r["lateral"]<=-20]),by_range=({f"{lo}-{hi}":band(lo,hi) for lo,hi in ((0,40),(40,60),(60,80),(80,200))} if len(A) else None))
    summ[name]["h_implied"]=round(CAL[name][1]*15.24/summ[name]["period_all"],2) if summ[name]["period_all"] else None
    print(name,json.dumps(summ[name]))
    for r in rows: print("   ",{k:r[k] for k in ("lateral","n","period","resid_rms","x")})
json.dump({"clip":CLIP,"summary":summ,"lines":res},open(OUT.rsplit(".",1)[0]+".json","w"),indent=1)
