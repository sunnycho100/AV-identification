"""Precision of low-score continuation (decision D28): are the weak boxes the tracker used vehicles?

Population: every state of the lo30k4 tracks (continuation 0.30, weak boxes at 4x noise)
whose detection scored under 0.45, on all graded clips. A fixed random sample (seed 0)
is cropped from the frames with each box's projected 3D outline, for labelling by eye.

    python scripts/evaluation/lowscore_precision.py sample      # writes the crops sheet and samples.json
    python scripts/evaluation/lowscore_precision.py score       # reads labels.json next to samples.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluators.result2kitti import get_lidar_3d_8points  # noqa: E402

CLIPS = ["HV_T_EW_1", "AV_T_WE_1", "AV_T_EW_3", "AV_T_WE_3", "AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5",
         "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]
TAG, DET, N = "lo30k4", "ft102feat25_lo10", 50
OUT = ROOT / "outputs/evaluation/lowscore_precision"


def population(tag=TAG):
    pop = []
    for clip in CLIPS:
        tracks = json.loads((ROOT / f"outputs/trajectories/{clip}_{tag}/tracks.json").read_text())["tracks"]
        for tid, st in tracks.items():
            pop += [(clip, tid, s["frame"], s["x_det"], s["y_det"], s["score"]) for s in st if "x_det" in s and s["score"] < 0.45]
    return pop


def box_for(clip, frame, x, y, score, det=DET):
    """The raw detection the tracker matched: same frame, nearest centre, score within 1e-4."""
    preds = json.loads((ROOT / f"outputs/object_detection/camera-data/{clip}_{det}/{frame:03d}_pred.json").read_text())
    cand = [b for b in preds if abs(b["score"] - score) < 1e-4] or preds
    return min(cand, key=lambda b: np.hypot(b["x"] - x, b["y"] - y))


def crop(clip, frame, b, det=DET):
    cal = json.loads((ROOT / f"outputs/object_detection/camera-data/{clip}_{det}/calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    c = np.asarray(get_lidar_3d_8points([b["l"], b["w"], b["h"]], b["yaw"], [b["x"], b["y"], b["z"] + b["h"] / 2]))
    q = (M @ np.c_[c, np.ones(8)].T)[:3]
    uv = (K @ q)[:2] / (K @ q)[2]
    img = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{frame:03d}.jpg"))
    x0, y0, x1, y1 = uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()
    m = max(x1 - x0, y1 - y0) * 0.6 + 20
    X0, Y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
    X1, Y1 = int(min(x1 + m, img.shape[1])), int(min(y1 + m, img.shape[0]))
    raw = img[Y0:Y1, X0:X1].copy()
    ann = raw.copy()
    p = (uv.T - [X0, Y0]).astype(int)
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        cv2.line(ann, tuple(p[i]), tuple(p[j]), (0, 255, 255), 1)
    return raw, ann


def tile(im, idx, label, size=220):
    h, w = im.shape[:2]
    s = size / max(h, w)
    im = cv2.resize(im, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_CUBIC)
    t = np.zeros((size + 24, size, 3), np.uint8)
    t[24:24 + im.shape[0], :im.shape[1]] = im
    cv2.putText(t, f"#{idx} {label}", (3, 17), 0, 0.5, (255, 255, 255), 1)
    return t


def sample(tag=TAG, det=DET, seed=0, out=OUT):
    pop = population(tag)
    rng = np.random.default_rng(seed)
    pick = [pop[i] for i in sorted(rng.choice(len(pop), N, replace=False))]
    out.mkdir(parents=True, exist_ok=True)
    rows, raws, anns = [], [], []
    for i, (clip, tid, f, x, y, sc) in enumerate(pick):
        b = box_for(clip, f, x, y, sc, det)
        raw, ann = crop(clip, f, b, det)
        lab = f"{clip[-6:]} f{f} {sc:.2f} {b['x']:.0f}m"
        raws.append(tile(raw, i, lab)); anns.append(tile(ann, i, lab))
        rows.append({"id": i, "clip": clip, "track": tid, "frame": f, "score": round(sc, 3), "x": round(b["x"], 1), "y": round(b["y"], 1)})
    grid = lambda ts: np.vstack([np.hstack(ts[r:r + 10]) for r in range(0, len(ts), 10)])
    cv2.imwrite(str(out / "samples_annotated.jpg"), grid(anns))
    cv2.imwrite(str(out / "samples_raw.jpg"), grid(raws))
    (out / "samples.json").write_text(json.dumps({"population": len(pop), "tag": tag, "seed": seed, "rows": rows}, indent=1))
    print(f"population {len(pop)} low-score states; sampled {N} -> {out}")


def score(out=OUT):
    s = json.loads((out / "samples.json").read_text())
    lab = json.loads((out / "labels.json").read_text())
    v = sum(1 for r in lab.values() if r["label"] == "vehicle")
    n = sum(1 for r in lab.values() if r["label"] == "not a vehicle")
    u = sum(1 for r in lab.values() if r["label"] == "unclear")
    off = sum(1 for r in lab.values() if r.get("off"))
    print(f"population {s['population']}; labelled {len(lab)}: vehicle {v}, not a vehicle {n}, unclear {u}; "
          f"precision {v / max(v + n, 1):.2f} (Wilson 95% interval below); displaced boxes {off}")
    z, p, k = 1.96, v / max(v + n, 1), v + n
    c = (p + z * z / (2 * k)) / (1 + z * z / k)
    hw = z * np.sqrt(p * (1 - p) / k + z * z / (4 * k * k)) / (1 + z * z / k)
    print(f"  95% interval {c - hw:.2f} to {c + hw:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "sample" and len(sys.argv) > 2:      # sample TAG DET SEED -> outputs/evaluation/lowscore_precision_TAG
        sample(sys.argv[2], sys.argv[3], int(sys.argv[4]), ROOT / f"outputs/evaluation/lowscore_precision_{sys.argv[2]}")
    elif sys.argv[1] == "score" and len(sys.argv) > 2:
        score(ROOT / f"outputs/evaluation/lowscore_precision_{sys.argv[2]}")
    else:
        {"sample": sample, "score": score}[sys.argv[1]]()
