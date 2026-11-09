"""Scenes where a single-frame (untracked) label sits within 3 m of another label.

One tile per event (same clip, frames within 10 of each other, same spot on the road),
middle frame, cropped around the pair. Red: the untracked box the proposed rule would
drop. White: the label it collides with. Thin green: labels from smoothed tracks.
Thin cyan: other untracked labels.

    python3 scripts/reporting/render_untracked_conflicts.py LABEL_SET OUT.jpg
"""
import json, glob, math, sys
import cv2, numpy as np

E = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
CLIPS = ("HV_T_EW_1", "AV_T_WE_1", "AV_T_EW_7")


def corners(b):
    c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
    out = []
    for dz in (0, b["h"]):
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lx, ly = sx * b["l"] / 2, sy * b["w"] / 2
            out.append((b["x"] + c * lx - s * ly, b["y"] + s * lx + c * ly, b["z"] + dz))
    return np.array(out)


def project(b, K, M):
    q = (M @ np.c_[corners(b), np.ones(8)].T)[:3]
    if (q[2] <= 1).any():
        return None
    p = K @ q
    return (p[:2] / p[2]).T.astype(int)


def tracked(b):
    return b["yaw_source"] == "track_motion"


def conflicts(labels):
    """(frame, drop, other) for each untracked label within 3 m of another label."""
    out = []
    for f, o in labels:
        for i, a in enumerate(o):
            for j, b in enumerate(o):
                if i >= j or math.hypot(a["x"] - b["x"], a["y"] - b["y"]) >= 3:
                    continue
                if tracked(a) and tracked(b):
                    continue
                drop, keep = (a, b) if (not tracked(a) and (tracked(b) or a["score"] < b["score"])) else (b, a)
                out.append((f, drop, keep))
    return out


def main():
    lab_set, out = sys.argv[1], sys.argv[2]
    tiles = []
    for c in CLIPS:
        cal = json.load(open(f"outputs/object_detection/camera-data/{c}_r2lab/calibration_used.json"))
        K, M = np.array(cal["K"]), np.array(cal["lidar2cam"])
        labels = [(int(p.split("/")[-1][:3]), json.load(open(p)))
                  for p in sorted(glob.glob(f"outputs/finetune/{lab_set}/{c}/*_label.json"))]
        by_frame = dict(labels)
        hits = conflicts(labels)
        events = []
        for h in hits:
            for ev in events:
                f0, d0, _ = ev[-1]
                if h[0] - f0 <= 10 and math.hypot(h[1]["x"] - d0["x"], h[1]["y"] - d0["y"]) < 15:
                    ev.append(h)
                    break
            else:
                events.append([h])
        for ev in events:
            f, drop, keep = ev[len(ev) // 2]
            img = cv2.imread(f"data/camera-data/{c}/frames_all/{f:03d}.jpg")
            for b in by_frame[f]:
                uv = project(b, K, M)
                if uv is not None:
                    for i, j in E:
                        cv2.line(img, tuple(uv[i]), tuple(uv[j]), (0, 200, 0) if tracked(b) else (230, 200, 0), 1)
            centre = []
            for b, col in ((keep, (255, 255, 255)), (drop, (0, 0, 255))):
                uv = project(b, K, M)
                if uv is not None:
                    for i, j in E:
                        cv2.line(img, tuple(uv[i]), tuple(uv[j]), col, 3)
                    centre.append(uv.mean(0))
            cx, cy = np.mean(centre, 0).astype(int) if centre else (960, 540)
            x0, y0 = int(np.clip(cx - 320, 0, 1920 - 640)), int(np.clip(cy - 180, 0, 1080 - 360))
            t = img[y0:y0 + 360, x0:x0 + 640].copy()
            cv2.rectangle(t, (0, 0), (640, 26), (0, 0, 0), -1)
            cv2.putText(t, f"{len(tiles) + 1}. {c} frames {ev[0][0]}-{ev[-1][0]} ({len(ev)} pairs), {drop['x']:.0f} m out",
                        (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tiles.append(t)
    cols = 4
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(out, np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]))
    print(len(tiles), "tiles,", out)


def _selfcheck():
    t = {"yaw_source": "track_motion", "score": 0.5, "x": 0, "y": 0}
    u = {"yaw_source": "lane_flow", "score": 0.9, "x": 2, "y": 0}
    far = {"yaw_source": "lane_flow", "score": 0.9, "x": 9, "y": 0}
    r = conflicts([(1, [t, u, far])])
    assert len(r) == 1 and r[0][1] is u and r[0][2] is t, r          # untracked dropped even with a higher score
    assert conflicts([(1, [t, {**t, "x": 1}])]) == []                # two tracked labels are never touched


if __name__ == "__main__":
    _selfcheck()
    main()
