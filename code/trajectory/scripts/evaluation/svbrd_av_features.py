"""AV (Waymo) vs other vehicles on the SVBRD-LLM Austin clips, from our trajectories.

The dataset has no per-vehicle labels; like the paper (Waymo flagged by its roof
dome and livery, then checked by hand), Waymos are labelled by eye:
  --sheet  writes outputs/reports/svbrd/<clip>_<tag>_moving.jpg, one crop per
           moving track (median speed > 2 m/s, at least 3 s and 10 m of travel) with its id, to label from;
  labels   outputs/reports/svbrd/waymo_labels.json {clip: [track ids of LABEL_TAG]}.
Other detector tags reuse those labels: the track whose positions stay within
MATCH_M of the labelled track most often is the Waymo (image-only, as mark_instrumented).

Features per moving track, the paper's six minus lane changes (an intersection):
mean and std speed, mean and std acceleration, std jerk, from the smoothed track
by central differences over 0.5 s (trajectory_features.diff). Speeds in mph in the
table, accelerations in m/s^2, so they compare with the paper's 0.31 vs 0.53
(accel std) and 0.65 vs 1.22 (jerk std).

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/svbrd_av_features.py --sheet --tag det2d
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/svbrd_av_features.py --tags det2d base102
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/pipeline"))
import trajectory_features as tf   # noqa: E402

CLIPS = ("SV_355", "SV_366", "SV_373", "SV_382", "SV_407", "SV_411")
LABEL_TAG = "det2d"
OUT = ROOT / "outputs/reports/svbrd"
MIN_S, MOVING_MPS, MIN_TRAVEL_M, MATCH_M, MIN_MATCH, FPS = 3.0, 2.0, 10.0, 2.5, 15, 30.0
MPH = 2.23694


def tracks(clip, tag):
    return json.loads((ROOT / "outputs/trajectories" / f"{clip}_{tag}" / "tracks.json").read_text())["tracks"]


def moving(T):
    """Tracks of at least MIN_S that really travel: parked cars' jittering 2D boxes
    can read a few m/s, but they never get MIN_TRAVEL_M from where they started."""
    return {k: t for k, t in T.items() if len(t) >= MIN_S * FPS
            and np.median([s["speed_mps"] for s in t]) > MOVING_MPS
            and np.hypot(t[-1]["x"] - t[0]["x"], t[-1]["y"] - t[0]["y"]) > MIN_TRAVEL_M}


def sheet(clip, tag):
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{clip}_{tag}" / "calibration_used.json").read_text())
    K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
    tiles = []
    for tid, t in sorted(moving(tracks(clip, tag)).items(), key=lambda kv: int(kv[0])):
        s = t[len(t) // 2]
        im = cv2.imread(str(ROOT / f"data/camera-data/{clip}/frames_all/{s['frame']:03d}.jpg"))
        c = K @ (M[:3, :3] @ np.array([s["x"], s["y"], 0.8]) + M[:3, 3])
        u, v = int(c[0] / c[2]), int(c[1] / c[2])
        crop = im[max(v - 90, 0):v + 90, max(u - 120, 0):u + 120]
        tile = cv2.resize(crop, (240, 180)) if crop.size else np.zeros((180, 240, 3), np.uint8)
        cv2.putText(tile, tid, (5, 25), 0, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 8] + [np.zeros((180, 240, 3), np.uint8)] * (8 - len(tiles[i:i + 8])))
            for i in range(0, len(tiles), 8)]
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{clip}_{tag}_moving.jpg"), np.vstack(rows))
    print(f"{clip}: {len(tiles)} moving tracks -> {OUT / f'{clip}_{tag}_moving.jpg'}")


def features(t):
    tt = np.array([s.get("t_s", s["frame"] / FPS) for s in t])
    v = np.array([s["speed_mps"] for s in t])
    a = tf.diff(v, FPS, 0.5, tt)
    j = tf.diff(a, FPS, 0.5, tt)
    return {"mean_v_mph": v.mean() * MPH, "std_v_mph": v.std() * MPH, "mean_a": a.mean(), "std_a": a.std(),
            "std_jerk": j.std()}


def match(ref, T):
    """Track in T that sits within MATCH_M of ref in the most shared frames, or None
    when none does for MIN_MATCH frames (the detector missed that car)."""
    rf = {s["frame"]: (s["x"], s["y"]) for s in ref}
    n = {k: sum(1 for s in t if s["frame"] in rf and np.hypot(s["x"] - rf[s["frame"]][0], s["y"] - rf[s["frame"]][1]) < MATCH_M)
         for k, t in T.items()}
    k = max(n, key=n.get)
    return k if n[k] >= MIN_MATCH else None


def table(tags):
    labels = {k: v for k, v in json.loads((OUT / "waymo_labels.json").read_text()).items() if not k.startswith("_")}
    res = {}
    for tag in tags:
        rows = []
        for clip in CLIPS:
            T = tracks(clip, tag)
            av = set(labels.get(clip, []))
            if tag != LABEL_TAG:
                ref = tracks(clip, LABEL_TAG)
                av = {match(ref[i], T) for i in av} - {None}
            for k, t in moving(T).items():
                rows.append({"clip": clip, "id": k, "av": k in av, **features(t)})
        res[tag] = rows
    keys = ("mean_v_mph", "std_v_mph", "mean_a", "std_a", "std_jerk")
    L = ["| Detector | Group | Tracks | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 3)]
    for tag, rows in res.items():
        for name, g in (("Waymo", [r for r in rows if r["av"]]), ("other", [r for r in rows if not r["av"]])):
            L.append(f"| {tag} | {name} | {len(g)} | " + " | ".join(
                f"{np.median([r[k] for r in g]):.2f}" if g else "" for k in keys) + " |")
    L.append("| paper (SVBRD-LLM) | AV / HDV |  |  |  |  | 0.31 / 0.53 | 0.65 / 1.22 |")
    (OUT / "av_features.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "av_features.md").write_text("Medians over moving tracks (>= 3 s, median speed > 2 m/s, >= 10 m travelled)\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser("Waymo vs other vehicles on SVBRD clips")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--tag", default=LABEL_TAG)
    ap.add_argument("--tags", nargs="+", default=[LABEL_TAG])
    a = ap.parse_args()
    if a.sheet:
        for c in CLIPS:
            sheet(c, a.tag)
    else:
        table(a.tags)


if __name__ == "__main__":
    main()
