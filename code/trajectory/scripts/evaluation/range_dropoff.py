"""Where the instrumented car's 3D detections disappear with range (GPS-free).

Reference: the car's track from the 2D baseline (Faster R-CNN plus ground plane, tag d2),
which holds the car out to 80 to 150 m. For every frame it observes the car, look for a
box within RADIUS of that position: in the fine-tuned head's raw output by score band,
after the road filter, in the pretrained head, and in the S4 and S4 low-score tracks.
Binned by distance along the road.

    python scripts/evaluation/range_dropoff.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ["AV_T_EW_3", "AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]
BINS = [(0, 40), (40, 60), (60, 80), (80, 100), (100, 200)]
RADIUS = 3.0                      # m: 2D bottom-centre bias plus box noise
DET = ROOT / "outputs/object_detection/camera-data"
TRAJ = ROOT / "outputs/trajectories"


def boxes(d, f, lo=0.0):
    p = d / f"{f:03d}_pred.json"
    return [(b["x"], b["y"]) for b in json.loads(p.read_text()) if b["score"] >= lo] if p.exists() else []


def near(bxs, xy):
    return any(np.hypot(x - xy[0], y - xy[1]) <= RADIUS for x, y in bxs)


def track_frames(clip, tag):
    inst = TRAJ / f"{clip}_{tag}" / "instrumented.json"
    vid = json.loads(inst.read_text()).get("vehicle_id") if inst.exists() else None
    if vid is None:
        return set()
    return {s["frame"] for s in json.loads((TRAJ / f"{clip}_{tag}" / "tracks.json").read_text())["tracks"][vid] if "x_det" in s}


def main():
    s0 = lambda c: "abl_s0" if c == "AV_T_EW_3" else "iv_s0"
    nv = "retrained" in sys.argv                                    # head trained without the test clip (D27)
    s4 = (lambda c: "nv45") if nv else (lambda c: "abl_s4" if c == "AV_T_EW_3" else "iv_s4")
    low, det = ("nv30k4", "nvfeat25_lo10") if nv else ("lo30k4", "ft102feat25_lo10")
    rows = {b: {k: 0 for k in ("frames", "raw45", "raw30", "raw10", "road45", "road30", "pre45", "s4", "s4low")} for b in BINS}
    for clip in CLIPS:
        ref = json.loads((TRAJ / f"{clip}_d2" / "instrumented.json").read_text()).get("vehicle_id")
        if ref is None:
            continue
        states = [s for s in json.loads((TRAJ / f"{clip}_d2" / "tracks.json").read_text())["tracks"][ref] if "x_det" in s]
        t4, t4l = track_frames(clip, s4(clip)), track_frames(clip, low)
        raw, road, pre = DET / f"{clip}_{det}", DET / f"{clip}_{det}_road", DET / f"{clip}_base102_h151_road"
        for s in states:
            xy, f = (s["x_det"], s["y_det"]), s["frame"]
            b = next((b for b in BINS if b[0] <= xy[0] < b[1]), None)
            if b is None:
                continue
            r = rows[b]
            r["frames"] += 1
            r["raw45"] += near(boxes(raw, f, 0.45), xy)
            r["raw30"] += near(boxes(raw, f, 0.30), xy)
            r["raw10"] += near(boxes(raw, f, 0.10), xy)
            r["road45"] += near(boxes(road, f, 0.45), xy)
            r["road30"] += near(boxes(road, f, 0.30), xy)
            r["pre45"] += near(boxes(pre, f), xy)
            r["s4"] += f in t4
            r["s4low"] += f in t4l
    print("| Range (m) | Frames | Fine-tuned raw >=0.45 | >=0.30 | >=0.10 | After road filter >=0.45 | >=0.30 | Pretrained >=0.45 (road) | S4 track | S4 low-score track |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for b, r in rows.items():
        n = max(r["frames"], 1)
        pc = lambda k: f"{100 * r[k] / n:.0f}%"
        print(f"| {b[0]}-{b[1]} | {r['frames']} | {pc('raw45')} | {pc('raw30')} | {pc('raw10')} | {pc('road45')} | {pc('road30')} | {pc('pre45')} | {pc('s4')} | {pc('s4low')} |")
    out = ROOT / f"outputs/evaluation/range_dropoff{'_retrained' if nv else ''}.json"
    out.write_text(json.dumps({f"{b[0]}-{b[1]}": r for b, r in rows.items()}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
