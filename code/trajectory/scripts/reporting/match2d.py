"""One-to-one matching of every label in a frame to 2D detections, by image overlap.

Each 2D box goes to the label that fits it best. A label whose best 2D box went to a
better-fitting label is a second box on the same car ("dup"). A label with no 2D box
at IoU 0.3 or more is "nosupport". No distance threshold anywhere.
"""
import json, glob, sys, collections
import numpy as np
sys.path.insert(0, "scripts/reporting")
import render_untracked_conflicts as R

MIN_IOU, MIN_SCORE, SAME_2D = 0.3, 0.5, 0.7

def rect(b, K, M):
    uv = R.project(b, K, M)
    return None if uv is None else (*uv.min(0), *uv.max(0))

def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return w * h / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - w * h + 1e-9)

def match(labels, boxes2d, K, M):
    """status per label: 'ok', 'dup' or 'nosupport', plus the matched 2D index."""
    rs = [rect(b, K, M) for b in labels]
    pairs = sorted(((iou(r, d["box2d"]), i, j) for i, r in enumerate(rs) if r for j, d in enumerate(boxes2d)), reverse=True)
    lab_of, box_of = {}, {}
    for v, i, j in pairs:
        if v < MIN_IOU: break
        if i not in lab_of and j not in box_of:
            lab_of[i], box_of[j] = j, i
    out = []
    for i in range(len(labels)):
        if i in lab_of: out.append(("ok", lab_of[i]))
        elif any(v >= MIN_IOU and a == i for v, a, _ in pairs): out.append(("dup", None))
        else: out.append(("nosupport", None))
    return out

def load(c):
    cal = json.load(open(f"outputs/object_detection/camera-data/{c}_r2lab/calibration_used.json"))
    return np.array(cal["K"]), np.array(cal["lidar2cam"])

def boxes2d(c, f):
    """2D detections, merged across classes: Faster R-CNN suppresses overlaps only within a
    class, so one van can carry both a 'car' and a 'truck' box."""
    bs = sorted((b for b in json.load(open(f"outputs/object_detection/camera-data/{c}_det2d/{f:03d}_pred.json")) if b["score"] >= MIN_SCORE),
                key=lambda b: -b["score"])
    kept = []
    for b in bs:
        if all(iou(b["box2d"], k["box2d"]) < SAME_2D for k in kept):
            kept.append(b)
    return kept

if __name__ == "__main__":
    for c in R.CLIPS:
        K, M = load(c); C = collections.Counter()
        for p in sorted(glob.glob(f"outputs/finetune/pseudo_labels_r2hb_v2/{c}/*_label.json")):
            f = int(p.split("/")[-1][:3]); labs = json.load(open(p))
            for b, (s, _) in zip(labs, match(labs, boxes2d(c, f), K, M)):
                C[(s, "tracked" if R.tracked(b) else "single")] += 1
        print(c, dict(sorted(C.items())))
