"""Tier-1b target-vehicle ID: multi-scale template matching of the lab hood marker.

The instrumented car carries a white hood marker with red symbols (see
`Camera data/figures/car.png`, the dataset README's reference). Color
thresholds failed across lighting (attempts 1-4), but the marker as a
grayscale+color TEMPLATE survives: match it per frame at multiple scales,
threshold the correlation score, then keep only temporally COHERENT runs
(>=3 consecutive frames with small position steps) to kill isolated FPs.

Works on front-view (EW) clips only — the marker is on the hood.
Validated 2026-07-15/16: AV_T_EW_3 frames 243-288 tracked (1 FP, removed by
coherence); HV_T_EW_1 frames 199-268 tracked -> proved the HV run uses the
same marker-equipped car. HV_T_EW_2: no detection (car stays too far;
marker sub-10 px is unresolvable — method floor, not marker absence).

Run: .venv/bin/python scripts/tracking/match_marker_template.py \
       --frames-dir data/camera-data/<CLIP>/frames_all --out-dir outputs/target_id/<CLIP>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAR_PNG = ROOT / "Camera data/figures/car.png"
# marker patch inside the README figure's red box (1920x1032 screenshot coords)
TPL_BOX = (540, 562, 476, 510)  # y0,y1,x0,x1
SCALES = [0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.6, 2.0, 2.5]
SCORE_MIN = 0.80
PEAK_MIN = 0.87      # a run is the TARGET only if it peaks this high; license
                     # plates (white box + dark symbols) run coherently at .80-.85
RUN_MIN = 3          # frames a coherent run must span
STEP_MAX = 40.0      # px, max per-frame position jump within a run
ROI_TOP = 300        # skip sky/buildings


def build_template():
    img = cv2.imread(str(CAR_PNG))
    y0, y1, x0, x1 = TPL_BOX
    return img[y0:y1, x0:x1]


def match_frames(frames_dir, tpl):
    per_frame = {}
    for fp in sorted(Path(frames_dir).glob("*.jpg")):
        img = cv2.imread(str(fp))
        roi = img[ROI_TOP:, :]
        best = (-1.0, 0, 0, 0, 0, 1.0)
        for s in SCALES:
            t = cv2.resize(tpl, None, fx=s, fy=s)
            if t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
                continue
            res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, loc[0], loc[1] + ROI_TOP, t.shape[1], t.shape[0], s)
        score, x, y, w, h, s = best
        per_frame[int(fp.stem)] = {"score": round(float(score), 3),
                                   "x": x, "y": y, "w": w, "h": h, "scale": s}
    return per_frame


def coherent_runs(per_frame):
    """Group above-threshold frames into runs of small consecutive position steps."""
    hits = [(f, r) for f, r in sorted(per_frame.items()) if r["score"] >= SCORE_MIN]
    runs, cur = [], []
    for f, r in hits:
        if cur:
            pf, pr = cur[-1]
            gap = f - pf
            dist = np.hypot(r["x"] - pr["x"], r["y"] - pr["y"])
            if gap <= 3 and dist <= STEP_MAX * gap:
                cur.append((f, r))
                continue
            if len(cur) >= RUN_MIN:
                runs.append(cur)
            cur = []
        cur.append((f, r))
    if len(cur) >= RUN_MIN:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser(description="find the lab marker car in a front-view clip")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tpl = build_template()
    per_frame = match_frames(args.frames_dir, tpl)
    runs = coherent_runs(per_frame)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_runs = [r for r in runs if max(x[1]["score"] for x in r) >= PEAK_MIN]
    summary = {
        "frames": len(per_frame),
        "score_min": SCORE_MIN, "peak_min": PEAK_MIN,
        "target_runs": [{"start": r[0][0], "end": r[-1][0], "frames": len(r),
                         "peak_score": max(x[1]["score"] for x in r)} for r in target_runs],
        "rejected_runs": [{"start": r[0][0], "end": r[-1][0], "frames": len(r),
                           "peak_score": max(x[1]["score"] for x in r)}
                          for r in runs if r not in target_runs],
        "per_frame": {str(f): r for f, r in per_frame.items()},
    }
    (out_dir / "marker_matches.json").write_text(json.dumps(summary, indent=1))

    if target_runs:
        # annotate the peak frame of the longest target run as visual evidence
        run = max(target_runs, key=len)
        f, r = max(run, key=lambda x: x[1]["score"])
        img = cv2.imread(str(sorted(Path(args.frames_dir).glob("*.jpg"))[0].parent / f"{f:03d}.jpg"))
        cv2.rectangle(img, (r["x"], r["y"]), (r["x"] + r["w"], r["y"] + r["h"]), (0, 255, 255), 2)
        cv2.imwrite(str(out_dir / f"marker_peak_{f:03d}.jpg"), img)
        for rn in summary["target_runs"]:
            print(f"TARGET run: frames {rn['start']}-{rn['end']} "
                  f"({rn['frames']} hits, peak {rn['peak_score']:.2f})")
    else:
        print("no TARGET marker run (rear-view clip, or target too far); "
              f"{len(summary['rejected_runs'])} sub-peak runs rejected (plates etc.)")
    print(f"saved: {out_dir}/marker_matches.json")


if __name__ == "__main__":
    # ponytail: self-check on the clip the template came from — must find the known run
    import sys
    if "--frames-dir" not in " ".join(sys.argv):
        pf = match_frames(ROOT / "data/camera-data/AV_T_EW_3/frames_all", build_template())
        runs = coherent_runs(pf)
        assert runs and any(r[0][0] <= 250 <= r[-1][0] for r in runs), "known AV_T_EW_3 run not found"
        print("self-check OK: AV_T_EW_3 marker run recovered")
    else:
        main()
