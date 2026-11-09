"""The 21 conflict scenes again, now decided by 2D matching. Yellow: 2D detections.
White: label kept (owns a 2D box). Red: second box on a car another label fits better.
Orange: no 2D box behind it. Each tile's caption gives the two boxes' verdicts."""
import json, glob, math, sys
import cv2, numpy as np
sys.path.insert(0, "scripts/reporting")
import render_untracked_conflicts as R
import match2d as X
COL = {"ok": (255, 255, 255), "dup": (0, 0, 255), "nosupport": (0, 140, 255)}
tiles, res = [], []
for c in R.CLIPS:
    K, M = X.load(c)
    labels = [(int(p.split("/")[-1][:3]), json.load(open(p))) for p in sorted(glob.glob(f"outputs/finetune/pseudo_labels_r2hb_v2/{c}/*_label.json"))]
    by = dict(labels); events = []
    for h in R.conflicts(labels):
        for ev in events:
            f0, d0, _ = ev[-1]
            if h[0] - f0 <= 10 and math.hypot(h[1]["x"] - d0["x"], h[1]["y"] - d0["y"]) < 15:
                ev.append(h); break
        else:
            events.append([h])
    for ev in events:
        f, a, b = ev[len(ev) // 2]
        labs, d2 = by[f], X.boxes2d(c, f)
        st = X.match(labs, d2, K, M)
        img = cv2.imread(f"data/camera-data/{c}/frames_all/{f:03d}.jpg")
        for d in d2:
            x0, y0, x1, y1 = map(int, d["box2d"]); cv2.rectangle(img, (x0, y0), (x1, y1), (0, 230, 255), 1)
        pts = []
        for lab, (s, _) in zip(labs, st):
            uv = R.project(lab, K, M)
            if uv is None: continue
            thick = 3 if (lab is a or lab is b) else 1
            for i, j in R.E: cv2.line(img, tuple(uv[i]), tuple(uv[j]), COL[s], thick)
            if thick == 3: pts.append(uv.mean(0))
        sa, sb = st[labs.index(a)][0], st[labs.index(b)][0]
        cx, cy = np.mean(pts, 0).astype(int)
        x0, y0 = int(np.clip(cx - 320, 0, 1280)), int(np.clip(cy - 180, 0, 720))
        t = img[y0:y0 + 360, x0:x0 + 640].copy()
        cv2.rectangle(t, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(t, f"{len(tiles) + 1}. {c} f{f}: single-frame {sa}, other {sb}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(t); res.append((len(tiles), c, f, sa, sb))
for r in res: print(r)
tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
cv2.imwrite(sys.argv[1], np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))
