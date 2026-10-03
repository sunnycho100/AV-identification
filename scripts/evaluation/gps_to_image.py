"""Evaluation only: put a clip's GPS track into the camera road frame and into pixels.

The road frame (x along the road away from the camera, y left, origin under the
camera) comes from each clip's own vanishing-point extrinsic. The GPS-to-road
transform is one 2D rotation and translation for the whole site, no scale, and
it is set ONLY from the two declared calibration clips (CAL_CLIPS), never from
the clip being graded:

  rotation      road bearing = median GPS velocity heading of the calibration
                cars (a car on a straight road drives along it). The camera's x
                axis is the lane vanishing direction, so no camera data enters.
                Which way +x points (with or against travel) comes from the sign
                of the image track's velocity.
  translation   camera ground point C in site ENU (origin: first GPS row of the
                first calibration clip, from lat/lon, so every clip shares it).
                Per calibration clip C_k = median(g - Rot q) over the
                image-identified car; C is the mean of the two. One clip drives
                each way, so a common GPS-video time offset, and the antenna's
                forward and sideways lever arm, flip sign between them and cancel
                in the mean. Half their difference is reported.

What this can and cannot determine. Across the road the transform is timing-free
on a straight road, so a graded clip's lateral offset is a real held-out test.
Along the road it is not: at steady speed v a GPS-video time offset tau and an
along-road shift v*tau are the same thing, and the lab's video_time_sec is an
estimate. The along-road anchor is therefore only as good as the calibration
clips' timing, and a graded clip's along-road error at the lab timing is mostly
that clip's own timing error. gps_benchmark.py fits the one constant tau per
clip and reports it, instead of hiding it in a per-clip translation.

Frame times come from data/camera-data/<clip>/frame_times.json (true
presentation times, frozen copies marked) when present, else frame / 30.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/gps_to_image.py --clip AV_T_EW_3
    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/gps_to_image.py --selfcheck
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
from site_error_model import enu_from_latlon, gps_at, load_gps, rot, wrap  # noqa: E402

CAL_CLIPS = ("HV_T_EW_1", "AV_T_WE_1")    # the only clips whose GPS may set calibration
ROAD_Z = -1.73                             # road plane in the detector output frame
L, W, H = 4.6, 1.9, 1.5                    # drawn car size, as render_trajectories.py
RED, MAGENTA, GREEN, ORANGE = (0, 0, 255), (255, 0, 255), (0, 220, 0), (0, 140, 255)


def site_origin():
    with open(ROOT / "Camera data" / f"{CAL_CLIPS[0]}_trajectory.csv") as f:
        r = next(csv.DictReader(f))
    return float(r["latitude"]), float(r["longitude"])


def frame_times(clip, n):
    """(t_s, copy) arrays for frames 0..n-1."""
    p = ROOT / f"data/camera-data/{clip}/frame_times.json"
    if p.exists():
        fr = {f["frame"]: f for f in json.loads(p.read_text())["frames"]}
        return (np.array([fr[i]["t_s"] for i in range(n)]), np.array([fr[i]["copy"] for i in range(n)]))
    return np.arange(n) / 30.0, np.zeros(n, bool)


def load_clip(clip, tag="ft102"):
    """GPS in site ENU, the image-identified car (or None), frame times, projection."""
    run = ROOT / "outputs/trajectories" / f"{clip}_{tag}"
    tracks = json.loads((run / "tracks.json").read_text())["tracks"]
    inst = run / "instrumented.json"
    vid = json.loads(inst.read_text())["vehicle_id"] if inst.exists() else None
    n = len(list((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg")))
    t, copy = frame_times(clip, n)
    cal = json.loads((ROOT / "outputs/object_detection/camera-data" / f"{clip}_{tag}" / "calibration_used.json").read_text())
    car = None
    if vid is not None:
        st = [s for s in tracks[vid] if not copy[s["frame"]]]
        car = {"id": vid, "frame": np.array([s["frame"] for s in st]),
               "t": np.array([t[s["frame"]] for s in st]),
               "q": np.array([[s["x"], s["y"]] for s in st]),
               "v": np.array([[s["vx"], s["vy"]] for s in st]),
               "speed": np.array([s["speed_mps"] for s in st])}
    return {"clip": clip, "gps": load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv", *site_origin()),
            "car": car, "tracks": tracks, "t": t, "copy": copy,
            "K": np.array(cal["K"]), "l2c": np.array(cal["lidar2cam"])}


def road_bearing(clips):
    """ENU angle of road +x: median GPS heading of each car, flipped when the image
    track drives against +x, then the circular mean over clips."""
    a = []
    for c in clips:
        _, psi, _ = gps_at(c["gps"], c["car"]["t"])
        a.append(float(np.median(psi)) + (0.0 if np.median(c["car"]["v"][:, 0]) > 0 else math.pi))
    return math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))


def site_transform(clips):
    """phi, C and per-clip diagnostics from calibration clips (lab timing as given)."""
    phi = road_bearing(clips)
    per = {}
    for c in clips:
        g, _, _ = gps_at(c["gps"], c["car"]["t"])
        Ck = np.median(g - c["car"]["q"] @ rot(phi).T, axis=0)
        per[c["clip"]] = Ck
    C = np.mean(list(per.values()), axis=0)
    d = {k: (rot(phi).T @ (v - C)).tolist() for k, v in per.items()}   # road frame, along/left
    return {"phi": phi, "C": C, "per_clip_offset_road_m": d, "clips": [c["clip"] for c in clips]}


def gps_road(gps, T, t):
    """GPS at video times t in the road frame: xy (N,2), heading (N,), speed (N,)."""
    g, psi, v = gps_at(gps, t)
    return (g - T["C"]) @ rot(T["phi"]), wrap(psi - T["phi"]), v


def to_px(xyz, K, l2c):
    """Road-frame points (N,3) to pixels (N,2); NaN behind the camera."""
    c = (l2c @ np.c_[xyz, np.ones(len(xyz))].T)[:3]
    uv = (K @ c)[:2] / c[2]
    uv[:, c[2] <= 1e-6] = np.nan
    return uv.T


def load_site():
    return site_transform([load_clip(c) for c in CAL_CLIPS])


def render(clip, T, offset=None, tag="ft102"):
    """Overlay video: tracked boxes (orange = image-picked car), GPS path (timing-free,
    red line), GPS car at the lab timing (red box) and, if given, at lab + offset s
    (magenta box; that offset was fitted on the picked car, so it lines up along the
    road by construction and only the across-road agreement is a check there)."""
    import cv2
    sys.path.insert(0, str(ROOT))
    from evaluators.result2kitti import get_lidar_3d_8points
    from scripts.data_converter.visual_utils import draw_box_3d, project_to_image
    c = load_clip(clip, tag)
    k34 = np.c_[c["K"], np.zeros(3)]
    gt = c["gps"]["t"]
    path, _, _ = gps_road(c["gps"], T, gt[::5])
    path_px = to_px(np.c_[path, np.full(len(path), ROAD_Z)], c["K"], c["l2c"])
    by_frame = {}
    for tid, states in c["tracks"].items():
        for s in states:
            by_frame.setdefault(s["frame"], []).append((tid, s))
    mark = c["car"]["id"] if c["car"] else None
    out = ROOT / "outputs/reports/gps_overlay" / clip
    out.mkdir(parents=True, exist_ok=True)

    def box(img, x, y, yaw, col, label):
        b = get_lidar_3d_8points([L, W, H], yaw, [x, y, ROAD_Z + H / 2])
        cc = (c["l2c"] @ np.c_[b, np.ones(8)].T).T[:, :3]
        if np.sum(cc[:, 2] > 1e-6) < 8:
            return
        pts = project_to_image(cc, k34)
        draw_box_3d(img, pts, c=col)
        u, v = int(pts[:, 0].min()), int(pts[:, 1].max()) + 20
        cv2.putText(img, label, (u, v), 0, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (u, v), 0, 0.6, col, 2)

    frames = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))
    for i, f in enumerate(frames):
        img = cv2.imread(str(f))
        ok = np.isfinite(path_px).all(1) & (path_px[:, 1] > 0)
        cv2.polylines(img, [path_px[ok].astype(np.int32)], False, RED, 1, cv2.LINE_AA)
        for tid, s in by_frame.get(i, []):
            yaw = math.atan2(s["vy"], s["vx"]) if s["speed_mps"] > 1 else s["yaw"]
            box(img, s["x"], s["y"], yaw, ORANGE if tid == mark else GREEN, f"{tid}" + (" picked" if tid == mark else ""))
        ti = c["t"][i]
        for dt, col, name in [(0.0, RED, "GPS lab time")] + ([(offset, MAGENTA, f"GPS {offset:+.2f} s")] if offset else []):
            if gt[0] <= ti + dt <= gt[-1]:
                xy, hd, _ = gps_road(c["gps"], T, np.array([ti + dt]))
                box(img, xy[0, 0], xy[0, 1], hd[0], col, f"{name} x={xy[0, 0]:.0f} m")
        tag_txt = "  FROZEN COPY" if c["copy"][i] else ""
        cv2.putText(img, f"{clip} frame {i} t={ti:.2f} s  picked car: {mark or 'none'}{tag_txt}",
                    (20, 40), 0, 1.0, (255, 255, 255), 3)
        cv2.imwrite(str(out / f"{i:03d}.jpg"), img)
    mp4 = out / f"{clip}_gps_overlay.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(out / "%03d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1280:-2", str(mp4)], check=True)
    return mp4


def _selfcheck():
    """Two synthetic calibration cars, one each way, both with the same 0.4 s timing
    error and a 0.8 m forward antenna offset: the mean recovers C exactly, the
    bearing is recovered, and the WE clip sits lever - v*tau along from the mean."""
    phi, C, v, tau, lever = math.radians(-2.7), np.array([-115.0, 17.0]), 28.0, 0.4, 0.8
    t = np.arange(0, 10, 1 / 125)
    clips = []
    for name, sgn, y in (("EW", -1, -5.2), ("WE", 1, -27.8)):
        x = (20 if sgn > 0 else 110) + sgn * v * t                  # true road x of the car
        xa = x + sgn * lever                                         # antenna ahead of centre
        g = C + np.c_[xa, np.full_like(xa, y)] @ rot(phi).T
        psi = np.full_like(t, phi + (0 if sgn > 0 else math.pi))
        gps = {"t": t + tau, "e": g[:, 0], "n": g[:, 1], "hx": np.cos(psi), "hn": np.sin(psi), "v": np.full_like(t, v)}
        tf = np.arange(30, 270) / 30.0
        q = np.c_[np.interp(tf, t, x), np.full_like(tf, y)]
        clips.append({"clip": name, "gps": gps, "car": {"t": tf, "q": q, "v": np.c_[np.full_like(tf, sgn * v), 0 * tf]}})
    T = site_transform(clips)
    assert abs(wrap(T["phi"] - phi)) < 1e-9, T["phi"]
    assert np.allclose(T["C"], C, atol=1e-6), T["C"]
    half = T["per_clip_offset_road_m"]["WE"][0]
    assert abs(half - (lever - v * tau)) < 1e-6, half
    xy, hd, _ = gps_road(clips[0]["gps"], T, np.array([5.0 + tau]))
    assert np.allclose(xy[0], [110 - v * 5.0 - lever, -5.2], atol=1e-6), xy
    assert abs(abs(hd[0]) - math.pi) < 1e-9
    K = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]])
    l2c = np.eye(4)[[1, 2, 0, 3]] * np.array([[-1], [-1], [1], [1]])   # x forward -> camera z
    assert np.allclose(to_px(np.array([[10.0, 0, 0]]), K, l2c), [[960, 540]])
    print("selfcheck ok (bearing, camera point, offset split and projection recovered)")


def main():
    ap = argparse.ArgumentParser("GPS track into the road frame and the image")
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--no-offset", action="store_true", help="skip the benchmark's fitted-offset box")
    ap.add_argument("--offset", type=float, default=None,
                    help="draw the magenta box at this offset instead (e.g. one borrowed from another clip)")
    a = ap.parse_args()
    T = load_site()
    print(f"site: bearing {math.degrees(T['phi']):.3f} deg, per-clip camera point vs mean (road along, left): "
          + ", ".join(f"{k} {v[0]:+.2f} {v[1]:+.2f} m" for k, v in T["per_clip_offset_road_m"].items()))
    for clip in a.clip:
        bj = ROOT / "outputs/evaluation/gps_benchmark" / f"{clip}.json"
        off = None
        if a.offset is not None:
            off = a.offset
        elif bj.exists() and not a.no_offset:
            off = (json.loads(bj.read_text()).get("timing") or {}).get("best_offset_s")
        print(clip, "->", render(clip, T, off))


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
