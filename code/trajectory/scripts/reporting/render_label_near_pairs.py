"""Crop every label pair with centres under 3 m from the review renders, one tile per pair."""
import json, glob, math, sys
import cv2, numpy as np
out = sys.argv[1]; tiles = []
for c in ("HV_T_EW_1", "AV_T_WE_1", "AV_T_EW_7"):
    cal = json.load(open(f"outputs/object_detection/camera-data/{c}_r2lab/calibration_used.json"))
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    uv = lambda b: (K @ (M @ [b["x"], b["y"], b["z"] + b["h"] / 2, 1])[:3])
    for f in sorted(glob.glob(f"outputs/finetune/pseudo_labels_r2hb_v2/{c}/*_label.json")):
        o, fr = json.load(open(f)), f.split("/")[-1][:3]
        for i in range(len(o)):
            for j in range(i + 1, len(o)):
                d = math.hypot(o[i]["x"] - o[j]["x"], o[i]["y"] - o[j]["y"])
                if d >= 3: continue
                p = [uv(o[k]) for k in (i, j)]; p = np.mean([q[:2] / q[2] for q in p], 0).astype(int)
                img = cv2.imread(f"outputs/finetune/review/pseudo_labels_r2hb_v2/{c}/{fr}.jpg")
                x0, y0 = np.clip(p[0] - 110, 0, 1920 - 220), np.clip(p[1] - 75, 0, 1080 - 150)
                t = cv2.resize(img[y0:y0 + 150, x0:x0 + 220], (330, 225))
                cv2.rectangle(t, (0, 0), (330, 20), (0, 0, 0), -1)
                cv2.putText(t, f"{len(tiles)} {c[:9]} f{fr} {d:.1f}m x{o[i]['x']:.0f}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                tiles.append(t)
cols = 8; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))
print(len(tiles), out)
