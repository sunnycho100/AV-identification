# Site Error Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-clip rigid-fit GPS grader with a site-level error model whose parameters are shared physical quantities, fitted jointly across clips and evaluated leave-one-clip-out.

**Architecture:** One script holds the model (GPS loading in a right-handed site frame, the residual function, a robust least-squares fit, leave-one-clip-out evaluation, and fit-free speed and acceleration metrics). One JSON config lists the site's clips. One helper renders track IDs on a frame so the target vehicle on rear-view clips can be picked by eye without GPS. Plain-assert tests on synthetic data prove the model recovers known parameters and exposes its own degeneracies.

**Tech Stack:** Python 3 under `.venv` (numpy, scipy 1.18, opencv already installed). No new dependencies. Tests are plain `assert` functions run with the interpreter, no pytest.

## Global Constraints

- Run everything with `.venv/bin/python` from the repository root `BEVHeights-mac/`.
- GPS trajectory CSVs under `Camera data/` are held-out evaluation ground truth. They are read only by the new grader, never by any calibration, detection, tracking, or target-identification step.
- The target vehicle must be chosen from the image only. Never choose a track by comparing to GPS.
- Commit messages: short subject line, details in the body, no `Co-Authored-By` trailer, no em dashes anywhere.
- Do not modify `scripts/tracking/grade_target_vs_gps.py` or `scripts/reporting/compare_trajectories.py`. The new grader lives beside them and they stay as the historical baseline.
- Do not add new pip dependencies.

---

## Why this model (read before building)

The current grader fits a rotation and translation per clip (2D Kabsch), then reports the residual as "position RMSE". Three problems:

1. **It grades away the calibration.** Rotation and translation between the camera frame and the world are exactly what a georeferenced calibration should supply. Fitting them per clip means the number reported is agreement in path shape, not position accuracy.
2. **It cannot represent the handedness mismatch.** The lab's GPS frame is (east, south, up), which is left-handed. The camera frame is (forward, left, up), right-handed. A proper rotation cannot map one onto the other. The fit works today only because every graded path is straight, and a straight segment is its own mirror image. On a curve it would grade a left turn against a right turn.
3. **The scale error is already visible in the residual and was fed back into calibration.** The along-road error trend (+1.5 m at 25 m, -1 m at 80 m) is linear in range and crosses zero at the centroid, which is what a range-scale error looks like once translation is absorbed. The similarity fit on AV_T_EW_3 gives scale 1.045 and drops RMSE from 0.99 m to 0.36 m. The 16.22 m "GPS-scale height" used to set the site height came from this same kind of fit, so calibration was tuned on the evaluation data.

The replacement has **no per-clip parameters**. Seven physical quantities are shared across all clips of a site:

```
theta = [C_e, C_n, beta_P0, beta_P1, s, l_x, l_y]

camera prediction   P_i = C + s * Rot(beta_p) q_i        (q_i = camera-frame track position)
GPS prediction      G_i = g_i + Rot(psi_i) l             (g_i = GPS position, psi_i = vehicle heading)
residual            r_i = P_i - G_i
```

| Parameter | Meaning | Why shared |
|---|---|---|
| `C_e, C_n` | camera ground point in site ENU | one physical camera |
| `beta_P0, beta_P1` | bearing of the camera-frame x axis, one per PTZ pose | `site_extrinsic.py` already gates clips to the same pose within 2 degrees, and the dry run found the two front-view clips' bearings agree to 0.3 degrees. All site T clips are pose P0; a re-pointed clip gets P1 |
| `s` | range scale about the camera | ray-plane range is proportional to camera height, so `s = h_true / h_used` to first order. `s * 16.26` is an implied camera height to compare against the OTC3D estimate of 14.9 to 15.6 m |
| `l_x, l_y` | GPS antenna to box-centre offset in the vehicle body frame (forward, left) | one physical antenna position on one instrumented vehicle |

Each clip is graded with parameters fitted on the **other** clips (leave-one-clip-out), so no clip's own GPS influences its own score. Fitting all clips together is reported too, as the in-sample number.

**Time offset is deliberately not a parameter.** All clips run at constant speed (25 to 30 m/s, varying under 3 m/s within a clip). A time offset shifts the GPS position along the travel direction by speed times offset; the lever arm `l_x` shifts it along the same direction. On constant-speed data these are indistinguishable, so one must be fixed from outside. The lab's frame sync (residual under 4 ms) is trusted and dt = 0.

**Why both drive directions matter.** On a single direction, `l_x` is indistinguishable from the camera's along-road position and `l_y` from its across-road position, because the heading is constant and the lever arm becomes a constant shift. When the vehicle drives the other way the lever-arm shift flips sign while the camera position does not, so the two separate. `free_mask` detects this from the clips' headings and holds the lever arm at zero until both directions are present. The plan reports the Jacobian singular values and the least-observable parameter direction so that any remaining degeneracy is printed rather than silently absorbed.

**What the dry run already showed.** Running the Task 1 code on the two existing front-view clips fits each clip alone to 0.3 to 0.6 m with `s` of 0.955 to 0.973 (implied camera height 15.2 to 15.8 m, matching OTC3D), but places the camera 75 m apart along the road between the two clips. That is a time-sync error, a wrong target vehicle, or a different pole, and Task 2 Step 3 says how to find out. The old grader could not see it.

**Fit-free metrics.** Speed and longitudinal acceleration from the camera track need no alignment at all: they are invariant to translation, rotation, and handedness. The ratio of mean GPS speed to mean camera speed is a direct estimate of `s` with no least squares involved, and must agree with the fitted `s`. These are also the quantities the downstream AV-versus-human classifier will consume.

---

## File Structure

- Create `scripts/evaluation/site_error_model.py`: model, loaders, fit, evaluation, CLI. Single responsibility: grade tracks against GPS under the shared model.
- Create `scripts/evaluation/test_site_error_model.py`: synthetic-data self-checks.
- Create `scripts/evaluation/site_T.json`: clip list for the Todd Drive site (the `_T_` clips).
- Create `scripts/evaluation/label_tracks_on_frame.py`: renders track IDs on one frame for image-only target picking on rear-view clips.
- Output (gitignored): `outputs/evaluation/site_T_report.json`.

Data formats consumed (already exist, do not change):

- `Camera data/<CLIP>_trajectory.csv`: columns used are `latitude`, `longitude`, `east_velocity`, `north_velocity`, `processed_heading_x`, `processed_heading_y` (south component), `video_time_sec`, `nearest_frame_index` (blank when the row is outside the video). About 125 Hz. `nearest_frame_time_sec` equals `frame / 30` exactly.
- `outputs/tracking/camera-data/<CLIP>_phase1/target_track.json`: `{"target_track_id": "137", "states": [{"frame", "x", "y", "z", "yaw", "vx", "vy", "vz", "speed_mps", "score"}, ...]}`. Exists for AV_T_EW_3 and HV_T_EW_1 (front view, identified by hood marker).
- `outputs/tracking/camera-data/<CLIP>_phase1/tracks.json`: `{"meta": {...}, "tracks": {"<id>": [states...]}}`. AV_T_WE_1 has this but no `target_track.json` (rear view, no hood marker visible).
- `outputs/object_detection/camera-data/<CLIP>_phase1/calibration_used.json`: `K` (3x3) and `lidar2cam` (4x4) exactly as consumed by the detector; track `x, y, z` are in the frame `lidar2cam` expects.
- Camera frame: x along the road pointing away from the camera, y left, z up, origin on the road below the camera. Tracker coasted frames repeat `vx` and `score` verbatim from the previous state.

---

### Task 1: Model, loaders, and synthetic self-checks

**Files:**
- Create: `scripts/evaluation/site_error_model.py`
- Create: `scripts/evaluation/test_site_error_model.py`

**Interfaces:**
- Produces: `enu_from_latlon(lat, lon, lat0, lon0) -> (e, n)` arrays in metres; `load_gps(csv_path, lat0, lon0) -> dict`; `gps_at(gps, t) -> (pos (N,2), psi (N,), v (N,))`; `load_track(path, track_id=None) -> (frames, xy, yaw, coasted)`; `build_clip(name, pose, gps, frames, xy, yaw, coasted) -> clip dict`; `predict(theta, clip) -> (p_cam, p_gps)`; `residuals(theta, clips) -> (2N,)`; `free_mask(clips) -> (7,) bool`; `fit(clips, theta0=None, free=None) -> {"theta", "sv", "null_dir", "free", "n"}`; `standalone(clip, lat0, lon0) -> dict`; `evaluate(theta, clip, mask) -> dict`; `kinematics(clip) -> dict`; `loco(clips) -> dict`. Constants `FPS = 30.0`, `POSES = ("P0", "P1")`, `NAMES`, `P_S = 4`, `P_LX = 5`.

- [ ] **Step 1: Write the failing tests**

Create `scripts/evaluation/test_site_error_model.py`:

```python
"""Self-checks for site_error_model on synthetic data.

    .venv/bin/python scripts/evaluation/test_site_error_model.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from site_error_model import (FPS, POSES, P_LX, P_S, enu_from_latlon, fit, free_mask,
                              gps_at, load_track, rot, wrap)

# C=(40,-15) m; camera x axis bearing 0.05 rad for pose P0 and 0.02 for pose P1;
# s=0.957 (16.26 m used, 15.56 m true); lever arm 1.2 m forward, 0.1 m left.
# A clip driving toward the camera has heading beta+pi, one driving away has heading
# beta: the heading flip is what separates the lever arm from the camera position.
TRUE = np.array([40.0, -15.0, 0.05, 0.02, 0.957, 1.2, 0.1])


def synth(pose, theta, toward_camera, n=80):
    """Straight constant-speed run, camera x from 25 to 90 m, exact under the model."""
    beta, C, s, l = theta[2 + POSES.index(pose)], theta[:2], theta[P_S], theta[P_LX:]
    x = np.linspace(90, 25, n) if toward_camera else np.linspace(25, 90, n)
    q = np.stack([x, np.full(n, -5.0)], 1)
    psi = beta + (np.pi if toward_camera else 0.0)
    p = C + s * (q @ rot(beta).T)
    g = p - rot(psi) @ l
    return {"name": pose, "pose": pose, "t": np.arange(n) / FPS, "q": q,
            "yaw": np.full(n, psi - beta), "g": g, "psi": np.full(n, psi),
            "v": np.full(n, abs(x[1] - x[0]) * FPS * s), "coasted": np.zeros(n, bool)}


def test_recovers_parameters_two_poses_both_directions():
    r = fit([synth("P0", TRUE, True), synth("P1", TRUE, False)])
    th = r["theta"]
    assert np.allclose(wrap(th[2:4] - TRUE[2:4]), 0, atol=1e-5), th
    assert np.allclose(np.r_[th[:2], th[4:]], np.r_[TRUE[:2], TRUE[4:]], atol=1e-4), th
    assert r["sv"][-1] / r["sv"][0] > 1e-6, r["sv"]


def test_recovers_parameters_one_pose_both_directions():
    # the site T case: one PTZ pose, clips driving both ways
    clips = [synth("P0", TRUE, True), synth("P0", TRUE, False)]
    m = free_mask(clips)
    assert m.tolist() == [True, True, True, False, True, True, True], m
    th = fit(clips, free=m)["theta"]
    assert abs(wrap(th[2] - TRUE[2])) < 1e-5, th
    assert np.allclose(np.r_[th[:2], th[4:]], np.r_[TRUE[:2], TRUE[4:]], atol=1e-4), th


def test_one_direction_is_degenerate():
    # one clip, all parameters free: beta_P1 unobservable, l_x vs C along-road,
    # l_y vs C across-road. Three near-zero singular values; observable ones sit near 1e-2.
    c = synth("P0", TRUE, True)
    r = fit([c])
    assert r["sv"][-3] / r["sv"][0] < 1e-6, r["sv"]
    # free_mask fixes exactly those three; two clips the same way stay degenerate
    assert free_mask([c]).tolist() == [True, True, True, False, True, False, False]
    assert free_mask([c, synth("P0", TRUE, True)]).tolist() == [True, True, True, False, True, False, False]


def test_enu_and_heading_are_right_handed():
    e, n = enu_from_latlon(43.001, -89.0, 43.0, -89.0)
    assert abs(n - 111.3) < 0.2 and abs(e) < 1e-9, (e, n)
    gps = {"t": np.array([0.0, 1.0]), "e": np.zeros(2), "n": np.zeros(2),
           "hx": np.array([1.0, 0.0]), "hn": np.array([0.0, -1.0]),   # east, then south
           "v": np.ones(2)}
    _, psi, _ = gps_at(gps, np.array([0.0, 1.0]))
    assert np.allclose(psi, [0.0, -np.pi / 2]), psi


def test_coasted_frames_flagged():
    st = [{"frame": 0, "x": 1, "y": 0, "yaw": 0, "vx": 0.5, "score": 0.7},
          {"frame": 1, "x": 2, "y": 0, "yaw": 0, "vx": 0.6, "score": 0.8},
          {"frame": 2, "x": 3, "y": 0, "yaw": 0, "vx": 0.6, "score": 0.8}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"states": st}, f)
    _, _, _, coasted = load_track(f.name)
    assert coasted.tolist() == [False, False, True], coasted


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python scripts/evaluation/test_site_error_model.py`
Expected: `ModuleNotFoundError: No module named 'site_error_model'`

- [ ] **Step 3: Write the model**

Create `scripts/evaluation/site_error_model.py`:

```python
"""Site error model: grade camera tracks against GPS with shared physical parameters.

Why. scripts/tracking/grade_target_vs_gps.py fits a rotation and translation per
clip and reports the residual. Those are what a georeferenced calibration should
supply, so the per-clip fit hides calibration error and grades path shape only.
It also cannot represent the mirror between the GPS frame (east, south) and the
camera frame (forward, left): harmless on straight paths, wrong on curves.

This model has NO per-clip parameters. Every parameter is a physical quantity
shared across all clips of a site and each clip is graded with parameters
fitted on the OTHER clips (leave-one-clip-out):

    theta = [C_e, C_n, beta_P0, beta_P1, s, l_x, l_y]
    P_i = C + s * Rot(beta_p) q_i          camera prediction of the box centre
    G_i = g_i + Rot(psi_i) l               GPS prediction of the box centre
    r_i = P_i - G_i

  C       camera ground point, site ENU (m)
  beta_p  bearing of the camera-frame x axis, one per PTZ pose; clips that pass the
          2-degree pose gate in site_extrinsic.py share a pose
  s       range scale about the camera; s = h_true / h_used to first order
  l       GPS antenna to box centre, vehicle body frame (forward, left)
  q_i     camera-frame track position (x forward, y left)
  g_i     GPS position in site ENU at the frame time; psi_i vehicle heading,
          ENU, counter-clockwise from east

Time offset is not a parameter: on constant-speed clips it is indistinguishable
from l_x. The lab's frame sync is trusted (dt = 0). Jacobian singular values are
reported so unobservable directions are printed, not absorbed.

GPS is read only here and only for grading. It never feeds calibration.

    .venv/bin/python scripts/evaluation/site_error_model.py \
        --out outputs/evaluation/site_T_report.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
R_EARTH = 6378137.0
FPS = 30.0
POSES = ("P0", "P1")
NAMES = ["C_e", "C_n", "beta_P0", "beta_P1", "s", "l_x", "l_y"]
P_S, P_LX = 4, 5


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fold(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def enu_from_latlon(lat, lon, lat0, lon0):
    """Local tangent plane in metres. Paths are under 300 m, so this is exact enough."""
    e = R_EARTH * np.radians(np.asarray(lon) - lon0) * math.cos(math.radians(lat0))
    n = R_EARTH * np.radians(np.asarray(lat) - lat0)
    return e, n


def load_gps(csv_path, lat0, lon0):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["nearest_frame_index"]]
    col = lambda k: np.array([float(r[k]) for r in rows])
    e, n = enu_from_latlon(col("latitude"), col("longitude"), lat0, lon0)
    # processed_heading_y is the SOUTH component; negate to make the frame right-handed
    return {"t": col("video_time_sec"), "e": e, "n": n,
            "hx": col("processed_heading_x"), "hn": -col("processed_heading_y"),
            "v": np.hypot(col("east_velocity"), col("north_velocity"))}


def gps_at(gps, t):
    """Position (N,2), heading psi (N,), speed (N,) interpolated at video times t."""
    i = lambda k: np.interp(t, gps["t"], gps[k])
    return np.stack([i("e"), i("n")], 1), np.arctan2(i("hn"), i("hx")), i("v")


def load_track(path, track_id=None):
    d = json.load(open(path))
    st = d["states"] if "states" in d else d["tracks"][str(track_id)]
    fr = np.array([s["frame"] for s in st], float)
    xy = np.array([[s["x"], s["y"]] for s in st], float)
    yaw = np.array([s["yaw"] for s in st], float)
    sc = np.array([s["score"] for s in st], float)
    vx = np.array([s["vx"] for s in st], float)
    # AB3DMOT repeats the previous score and velocity verbatim on frames with no detection
    coasted = np.r_[False, (sc[1:] == sc[:-1]) & (vx[1:] == vx[:-1])]
    return fr, xy, yaw, coasted


def build_clip(name, pose, gps, fr, xy, yaw, coasted):
    g, psi, v = gps_at(gps, fr / FPS)
    return {"name": name, "pose": pose, "t": fr / FPS, "q": xy, "yaw": yaw,
            "g": g, "psi": psi, "v": v, "coasted": coasted}


def predict(theta, c):
    beta = theta[2 + POSES.index(c["pose"])]
    p_cam = theta[:2] + theta[P_S] * (c["q"] @ rot(beta).T)
    lx, ly = theta[P_LX], theta[P_LX + 1]
    cp, sp = np.cos(c["psi"]), np.sin(c["psi"])
    p_gps = c["g"] + np.stack([cp * lx - sp * ly, sp * lx + cp * ly], 1)
    return p_cam, p_gps


def residuals(theta, clips):
    out = []
    for c in clips:
        p_cam, p_gps = predict(theta, c)
        out.append((p_cam - p_gps)[~c["coasted"]].ravel())
    return np.concatenate(out)


def initial_theta(clips):
    theta = np.zeros(7)
    theta[P_S] = 1.0
    for pi_, pz in enumerate(POSES):
        cs = [c for c in clips if c["pose"] == pz]
        if not cs:
            continue
        c0 = cs[0]                            # one clip fixes the branch; all share the pose
        psi = float(np.median(c0["psi"]))
        q, gg = c0["q"], c0["g"]
        best = None
        for beta in (psi, psi + np.pi):      # camera x axis runs along or against travel
            th = theta.copy()
            th[2 + pi_] = beta
            th[:2] = gg.mean(0) - (q @ rot(beta).T).mean(0)
            cost = float(np.sum(residuals(th, [c0]) ** 2))
            if best is None or cost < best[0]:
                best = (cost, th)
        theta = best[1]
    return theta


def free_mask(clips):
    """Which parameters the data can constrain. A pose with no clips has no beta.
    The lever arm separates from the camera position only when the vehicle
    headings span both drive directions: the heading flip changes the sign of
    Rot(psi) l while C stays put."""
    poses = {c["pose"] for c in clips}
    m = np.ones(7, bool)
    for pi_, pz in enumerate(POSES):
        m[2 + pi_] = pz in poses
    heads = [float(np.median(c["psi"])) for c in clips]
    m[P_LX:] = any(abs(wrap(a - b)) > np.pi / 2 for a in heads for b in heads)
    return m


def fit(clips, theta0=None, free=None):
    """Robust least squares over the parameters marked free; the rest stay at theta0.
    Returns the singular values of the Jacobian over the free parameters and the
    least-observable direction, so a degenerate fit is visible instead of drifting."""
    theta0 = initial_theta(clips) if theta0 is None else np.asarray(theta0, float)
    free = np.ones(7, bool) if free is None else np.asarray(free, bool)

    def full(p):
        th = theta0.copy()
        th[free] = p
        return th

    # s > 0 removes the exact symmetry (s, beta) == (-s, beta + pi)
    lo, hi = np.full(7, -np.inf), np.full(7, np.inf)
    lo[P_S], hi[P_S] = 0.5, 2.0
    res = least_squares(lambda p: residuals(full(p), clips), theta0[free],
                        bounds=(lo[free], hi[free]), loss="soft_l1", f_scale=1.0)
    _, sv, vt = np.linalg.svd(res.jac)
    null = np.zeros(7)
    null[free] = vt[-1]
    return {"theta": full(res.x), "sv": sv, "null_dir": null, "free": free,
            "n": res.fun.size // 2}


def evaluate(theta, c, mask):
    """Residual statistics on the frames selected by mask. Sign: camera minus GPS,
    along positive means the camera places the car ahead along its travel direction,
    across positive means to the left of travel."""
    p_cam, p_gps = predict(theta, c)
    r = (p_cam - p_gps)[mask]
    d = np.stack([np.cos(c["psi"]), np.sin(c["psi"])], 1)[mask]
    along = (r * d).sum(1)
    across = (r * np.stack([-d[:, 1], d[:, 0]], 1)).sum(1)
    e = np.linalg.norm(r, axis=1)
    beta = theta[2 + POSES.index(c["pose"])]
    dh = wrap(c["yaw"][mask] + beta - c["psi"][mask])
    return {"n": int(mask.sum()),
            "rmse_m": rms(e), "median_m": float(np.median(e)),
            "p95_m": float(np.percentile(e, 95)),
            "along_bias_m": float(along.mean()), "along_rms_m": rms(along),
            "across_bias_m": float(across.mean()), "across_rms_m": rms(across),
            "heading_bias_deg": float(np.degrees(np.median(dh))),
            "heading_fold_mae_deg": float(np.degrees(np.abs(fold(dh)).mean())),
            "n_heading_flipped": int((np.abs(dh) > np.pi / 2).sum())}


def kinematics(c, window=31):
    """Speed and longitudinal acceleration from the camera track alone: no
    alignment, no fitted parameters. speed_ratio is a direct estimate of s.
    Per-frame box positions are noisy, so the derivative window is 1 s; the
    acceleration number is mostly a noise floor on these constant-speed clips."""
    window = window if len(c["q"]) >= window else 15
    if len(c["q"]) < window:
        return {}
    vel = savgol_filter(c["q"], window, 2, deriv=1, delta=1 / FPS, axis=0)
    acc = savgol_filter(c["q"], window, 2, deriv=2, delta=1 / FPS, axis=0)
    v_cam = np.linalg.norm(vel, axis=1)
    a_cam = (vel * acc).sum(1) / np.maximum(v_cam, 1e-6)
    a_gps = np.gradient(c["v"], c["t"])
    m = ~c["coasted"]
    return {"speed_ratio_gps_over_cam": float(c["v"][m].mean() / v_cam[m].mean()),
            "speed_rmse_mps": rms(v_cam[m] - c["v"][m]),
            "accel_rmse_mps2": rms(a_cam[m] - a_gps[m])}


def standalone(c, lat0, lon0):
    """One clip alone: C, beta, s with the lever arm fixed at zero. NOT a grade (it
    fits the clip it scores). A consistency check: every clip from one camera must
    put C at the same place. Disagreement along the road with across-road agreement
    means a time-sync error in that clip's CSV or the wrong target vehicle in the
    same lane; the camera coordinates are returned so a map can arbitrate."""
    m = free_mask([c])
    m[P_LX:] = False
    f = fit([c], free=m)
    th = f["theta"]
    lat = lat0 + math.degrees(th[1] / R_EARTH)
    lon = lon0 + math.degrees(th[0] / (R_EARTH * math.cos(math.radians(lat0))))
    ev = evaluate(th, c, ~c["coasted"])
    return {"C_e": float(th[0]), "C_n": float(th[1]),
            "camera_lat": float(lat), "camera_lon": float(lon),
            "beta_deg": float(np.degrees(wrap(th[2 + POSES.index(c["pose"])]))),
            "s": float(th[P_S]), "rmse_m": ev["rmse_m"],
            "along_rms_m": ev["along_rms_m"], "across_rms_m": ev["across_rms_m"],
            "speed_ratio_gps_over_cam": kinematics(c).get("speed_ratio_gps_over_cam")}


def loco(clips):
    out = {}
    for c in clips:
        others = [o for o in clips if o is not c]
        if not others:
            continue
        f = fit(others, free=free_mask(others))
        out[c["name"]] = {
            "theta_from_other_clips": dict(zip(NAMES, f["theta"].tolist())),
            "sv_min_over_max": float(f["sv"][-1] / f["sv"][0]),
            "all_frames": evaluate(f["theta"], c, np.ones(len(c["q"]), bool)),
            "excluding_coasted": evaluate(f["theta"], c, ~c["coasted"]),
            "fit_free": kinematics(c)}
    return out


def load_clips(cfg):
    clips = []
    for c in cfg["clips"]:
        if c.get("track_id") == "PICK_FROM_IMAGE":
            print(f"skipping {c['name']}: track_id not chosen yet")
            continue
        gps = load_gps(ROOT / "Camera data" / f"{c['name']}_trajectory.csv",
                       cfg["lat0"], cfg["lon0"])
        clips.append(build_clip(c["name"], c["pose"], gps,
                                *load_track(ROOT / c["track"], c.get("track_id"))))
    return clips


def main():
    ap = argparse.ArgumentParser("Shared-parameter site error model, leave-one-clip-out")
    ap.add_argument("--config", default=str(ROOT / "scripts/evaluation/site_T.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    clips = load_clips(cfg)
    free = free_mask(clips)
    f = fit(clips, free=free)
    report = {"site": cfg["site"], "clips": [c["name"] for c in clips],
              "per_clip_standalone": {c["name"]: standalone(c, cfg["lat0"], cfg["lon0"])
                                      for c in clips},
              "free_parameters": [n for n, k in zip(NAMES, free) if k],
              "theta_all_clips": dict(zip(NAMES, f["theta"].tolist())),
              "implied_camera_height_m": float(f["theta"][P_S] * cfg["height_used_m"]),
              "jacobian_singular_values": f["sv"].tolist(),
              "least_observable_direction": dict(zip(NAMES, f["null_dir"].tolist())),
              "in_sample": {c["name"]: evaluate(f["theta"], c, ~c["coasted"]) for c in clips},
              "leave_one_clip_out": loco(clips)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```


- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python scripts/evaluation/test_site_error_model.py`
Expected:
```
ok test_recovers_parameters_with_both_directions
ok test_one_direction_is_degenerate
ok test_enu_and_heading_are_right_handed
ok test_coasted_frames_flagged
```

If `test_recovers_parameters_with_both_directions` fails on tolerance only, print `r["theta"] - TRUE` and check it is under 1e-3 everywhere; if so, loosen `atol` to 1e-3 and note it in the commit body. If a parameter is off by more than that, the bug is in `predict` or `synth`, not in the tolerance.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/site_error_model.py scripts/evaluation/test_site_error_model.py
git commit -m "Add shared-parameter site error model for GPS grading" -m "Replaces per-clip rigid alignment with seven physical parameters shared across a site (camera position, per-direction bearing, range scale, antenna lever arm), fitted jointly and evaluated leave-one-clip-out. GPS frame is converted to right-handed ENU before use. Synthetic self-checks cover parameter recovery, single-direction degeneracy, handedness, and coasted-frame flagging."
```

---

### Task 2: Site config and first run on the two front-view clips

**Files:**
- Create: `scripts/evaluation/site_T.json`
- Output: `outputs/evaluation/site_T_report.json` (gitignored)

**Interfaces:**
- Consumes: `main()` from Task 1 with `--config` and `--out`.
- Produces: the config schema `{"site", "lat0", "lon0", "height_used_m", "clips": [{"name", "pose", "track", "track_id"?, "identified_from"?}]}` that Task 3 fills in. `pose` is a PTZ pose label; clips whose `metric_extrinsic_site.json` passed the 2-degree reference gate share one label.

- [ ] **Step 1: Write the config**

Create `scripts/evaluation/site_T.json`. `lat0, lon0` is a fixed site origin near the Todd Drive camera (the clips' own GPS origins are each clip's first row, about 350 m apart, so a shared origin is needed). `height_used_m` is the site height the detector ran with.

```json
{
  "site": "T",
  "lat0": 43.0353,
  "lon0": -89.4215,
  "height_used_m": 16.26,
  "clips": [
    {"name": "AV_T_EW_3", "pose": "P0",
     "track": "outputs/tracking/camera-data/AV_T_EW_3_phase1/target_track.json"},
    {"name": "HV_T_EW_1", "pose": "P0",
     "track": "outputs/tracking/camera-data/HV_T_EW_1_phase1/target_track.json"},
    {"name": "AV_T_WE_1", "pose": "P0",
     "track": "outputs/tracking/camera-data/AV_T_WE_1_phase1/tracks.json",
     "track_id": "PICK_FROM_IMAGE",
     "identified_from": "manual pick from label_tracks_on_frame.py render, image only, GPS never read"}
  ]
}
```

- [ ] **Step 2: Run on the two front-view clips**

Run: `.venv/bin/python scripts/evaluation/site_error_model.py --out outputs/evaluation/site_T_report.json`
Expected: prints `skipping AV_T_WE_1: track_id not chosen yet`, then a JSON report with `clips: ["AV_T_EW_3", "HV_T_EW_1"]` and `free_parameters: ["C_e", "C_n", "beta_P0", "s"]` (both clips drive west, so the lever arm is held at zero; no clip has pose P1, so `beta_P1` is held too).

- [ ] **Step 3: Read `per_clip_standalone` first, before any other number**

This block fits each clip on its own (camera position, bearing, scale; lever arm zero). It is not a grade. It is the consistency check that the old per-clip rigid fit could never perform, because that fit absorbed the camera position.

A dry run of this exact code on 2026-09-13 produced:

| clip | C_e (m) | C_n (m) | camera lat, lon | beta (deg) | s | rmse (m) | across rms (m) | speed ratio |
|---|---|---|---|---|---|---|---|---|
| AV_T_EW_3 | -93.7 | 15.8 | 43.03540, -89.42270 | -2.46 | 0.955 | 0.34 | 0.10 | 0.962 |
| HV_T_EW_1 | -168.4 | 19.5 | 43.03550, -89.42360 | -2.14 | 0.973 | 0.63 | 0.18 | 1.013 |

Read it this way:

- Each clip alone fits to well under a metre, with across-road agreement of 0.1 to 0.2 m. The bearing agrees to 0.3 degrees. The scale `s` is 0.955 to 0.973 in both, and the implied camera height (`s` times 16.26 m) is 15.2 to 15.8 m, inside the OTC3D range of 14.9 to 15.6 m. The range-dependent depth bias in the README is therefore consistent with a calibration height error, not model depth behaviour.
- The two clips put the same camera 75 m apart **along the road** while agreeing across the road. A camera does not move 75 m along a highway. Exactly one of these is true and the report cannot tell which: (a) one clip's `video_time_sec` column is offset by about 2.7 s (75 m at 28 m/s), (b) one clip's target track is a different vehicle in the same lane as the instrumented car, (c) the two clips come from different camera poles. The old grader hid this completely, because a per-clip translation absorbs any of the three.
- The leave-one-clip-out numbers (`rmse_m` near 74 m) are meaningless until this is resolved. Do not report them.

Resolve it in this order, and record the outcome in the commit body of Step 4:

1. **Map check (decisive, image-free, GPS-free).** Open both `camera_lat, camera_lon` pairs in Google Earth or Google Maps satellite view and find the WisDOT camera pole on the Beltline (US 12/18) at the Todd Drive interchange, Madison WI. The pair within about 10 m of the pole belongs to the consistent clip. If the pole sits near neither, hypothesis (c) is wrong and the true camera position is a third point; note it and continue.
2. **Target identity on the inconsistent clip.** Run the existing `scripts/reporting/render_target_id.py --clip <CLIP>` and step through the frames. The orange box must stay on the marker car for the whole track, including frames before and after the marker window. If it jumps to a neighbouring car in the same lane, the identification is wrong; fix it in the image (choose the correct ID from `label_tracks_on_frame.py` renders, Task 3 Step 1) and write the corrected ID and frames used into the config as a `track_id` override with `identified_from` stating it was a manual image pick.
3. **Time alignment on the inconsistent clip.** If the target is right, the CSV's `video_time_sec` is suspect by `disagreement_along_m / mean_speed` seconds (75 m at 28 m/s is 2.7 s). Ask the lab how `video_time_sec` was derived for that clip. Do not correct it by fitting; a corrected value must come from the lab's timestamps. If they supply one, add `"dt_s": <value>` to that clip's config entry and apply it in `build_clip` as `gps_at(gps, fr / FPS + dt_s)`.

Only when the standalone camera positions agree to within a few metres is the shared model valid and `leave_one_clip_out` reportable. Then check:

- `implied_camera_height_m` against OTC3D (14.9 to 15.6 m).
- `fit_free.speed_ratio_gps_over_cam` per clip against `s` (should agree to about 0.02; on the dry run they did for AV_T_EW_3 at 0.962 vs 0.955, and disagreed for HV_T_EW_1 at 1.013 vs 0.973, which is itself weak evidence that the HV_T_EW_1 track is the odd one).
- `excluding_coasted.across_rms_m` small (0.2 to 0.7 m), `along_rms_m` well under the old 0.9 m once scale is modelled.
- `heading_bias_deg` near 174 to 175 in both clips: after the 180-degree front/back flip, the detector's yaw carries a constant offset of about 5 degrees. This is systematic, not noise, and worth a line in the README.

- [ ] **Step 4: Commit the config**

```bash
git add scripts/evaluation/site_T.json
git commit -m "Add Todd Drive site config for the error model"
```

---

### Task 3: Image-only target pick on the rear-view clip, which breaks the degeneracy

**Files:**
- Create: `scripts/evaluation/label_tracks_on_frame.py`
- Modify: `scripts/evaluation/site_T.json` (fill `track_id` for AV_T_WE_1)

**Interfaces:**
- Consumes: `outputs/object_detection/camera-data/AV_T_WE_1_phase1/calibration_used.json` (`K`, `lidar2cam`), `outputs/tracking/camera-data/AV_T_WE_1_phase1/tracks.json`, `data/camera-data/AV_T_WE_1/frames_all/NNN.jpg`.
- Produces: `outputs/tracking/camera-data/AV_T_WE_1_phase1_tracks_frameNNN.jpg` with track IDs drawn at each track's projected position.

- [ ] **Step 1: Write the renderer**

Create `scripts/evaluation/label_tracks_on_frame.py`:

```python
"""Draw every track alive at one frame with its ID, so the instrumented car can be
picked by eye against Camera data/figures/car.png. Image only; GPS is never read.

    .venv/bin/python scripts/evaluation/label_tracks_on_frame.py --clip AV_T_WE_1 --frame 150
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--run", default="phase1")
    a = ap.parse_args()
    det = ROOT / f"outputs/object_detection/camera-data/{a.clip}_{a.run}"
    cal = json.load(open(det / "calibration_used.json"))
    K, l2c = np.array(cal["K"], float), np.array(cal["lidar2cam"], float)
    tracks = json.load(open(ROOT / f"outputs/tracking/camera-data/{a.clip}_{a.run}/tracks.json"))["tracks"]
    img = cv2.imread(str(ROOT / f"data/camera-data/{a.clip}/frames_all/{a.frame:03d}.jpg"))
    for tid, states in tracks.items():
        s = next((s for s in states if s["frame"] == a.frame), None)
        if s is None:
            continue
        pc = l2c @ np.array([s["x"], s["y"], s["z"], 1.0])
        if pc[2] <= 0:
            continue
        u, v, w = K @ pc[:3]
        u, v = int(u / w), int(v / w)
        cv2.circle(img, (u, v), 6, (0, 140, 255), -1)
        cv2.putText(img, tid, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
    out = det.parent / f"{a.clip}_{a.run}_tracks_frame{a.frame:03d}.jpg"
    cv2.imwrite(str(out), img)
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the projection lands on cars**

Run: `.venv/bin/python scripts/evaluation/label_tracks_on_frame.py --clip AV_T_WE_1 --frame 150`
Open the printed JPG next to `outputs/object_detection/camera-data/AV_T_WE_1_phase1/150_annotated.jpg`. Every orange dot must sit on a car that has a box in the annotated image. If the dots are offset by a constant amount, `lidar2cam` in the sidecar expects a different z convention than the tracks carry: try `s["z"] + cal["ground_shift_applied_m"]` and `s["z"] - cal["ground_shift_applied_m"]`, keep whichever puts dots on cars, and record which in the commit body.

- [ ] **Step 3: Pick the target track**

Render three frames spread across the clip (for example 60, 150, 240). Identify the instrumented vehicle by appearance against `Camera data/figures/car.png` and pick the track ID that follows it. The longest tracks in this clip are IDs 8 (299 frames) and 6 (288 frames); the target is usually one of the long tracks but confirm by appearance, not length. Write the ID into `scripts/evaluation/site_T.json` replacing `"PICK_FROM_IMAGE"`, and extend `identified_from` with the frames used, for example `"manual pick, frames 60/150/240, image only, GPS never read"`.

- [ ] **Step 4: Run the full model**

Run: `.venv/bin/python scripts/evaluation/site_error_model.py --out outputs/evaluation/site_T_report.json`
Expected: `clips` lists all three; `free_parameters` now includes `l_x` and `l_y` (the WE clip drives east, the others west); `jacobian_singular_values` has no near-zero entries (the smallest over the largest should exceed about 1e-4). Check the lever arm physically: `l_x` should be a plausible antenna-to-centre offset for a passenger car (magnitude under 2.5 m), `l_y` under 1 m. If `l_x` comes out at several metres, the WE target pick is wrong, or the WE clip does not share pose P0 (check its `metric_extrinsic_site.json` diagnostics against the reference pose; if it was re-pointed, give it `"pose": "P1"`, at which point it cannot be graded leave-one-out until a second P1 clip exists). Also read `per_clip_standalone` for the new clip: its camera position is a third vote on the 75 m disagreement from Task 2.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluation/label_tracks_on_frame.py scripts/evaluation/site_T.json
git commit -m "Add image-only track labelling and rear-view target for site T" -m "The rear-view clip AV_T_WE_1 has no hood marker, so its target track is picked by eye from a frame render with track IDs. Adding a second drive direction makes the antenna lever arm separable from the camera position."
```

---

### Task 4: Interpretation write-up

**Files:**
- Modify: `README.md` (the "Trajectory accuracy vs GPS" row of the status table and the "Range-dependent depth bias" paragraph under known issues)

- [ ] **Step 1: Read the report and fill the table**

From `outputs/evaluation/site_T_report.json` record, per held-out clip, from `leave_one_clip_out.<clip>.excluding_coasted`: `rmse_m`, `along_bias_m`, `along_rms_m`, `across_rms_m`, `heading_bias_deg`; and from `fit_free`: `speed_ratio_gps_over_cam`, `speed_rmse_mps`, `accel_rmse_mps2`. From the top level: `theta_all_clips`, `implied_camera_height_m`, `least_observable_direction`.

- [ ] **Step 2: Update README**

Replace the "Trajectory accuracy vs GPS" status row text with the leave-one-clip-out RMSE range and add one sentence: "Graded with a site error model whose parameters are shared across clips and fitted on the other clips; no per-clip alignment." In the "Range-dependent depth bias" paragraph, replace the claim that the bias is model depth behaviour with the finding from `s`: state the implied camera height and whether it agrees with the OTC3D estimate (14.9 to 15.6 m). If it agrees, the paragraph should say the bias is a calibration height error, and the next action is to re-run detection with the implied height and re-grade. Keep both edits under five sentences total.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Report GPS grading from the site error model"
```

---

## What this plan does not do (on purpose)

- It does not re-run detection with a corrected height. That is the natural next step if `implied_camera_height_m` agrees with OTC3D, but it changes the detector inputs and belongs in its own plan.
- It does not fit a time offset. See the reasoning section: not identifiable on constant-speed clips. If a clip with braking is ever recorded, add `dt` per clip then.
- It does not smooth the tracks before grading. The classifier's smoothing belongs in the tracking stage, and this grader should see what the tracker emits.
- It does not touch the V and W sites. Once site T works, copy `site_T.json` to `site_V.json` and `site_W.json` with their own `lat0, lon0` and clips; the code is site-agnostic.
- It supports at most two PTZ poses per site (`POSES`). Add a third label if a site ever needs one.

---

## Execution notes (2026-09-13)

Executed in this repo after two external reviews. Changes from the plan as written above:

- `loco` skips a hold-out whose pose or drive direction has no training clip, with a reason, instead of reporting a meaningless score (perfect synthetic data gave 2.41 m error otherwise).
- Singular values come from a plain central-difference Jacobian, not the robust-loss-weighted one `least_squares` returns.
- Per-clip frame rate is recovered from the CSV (`nearest_frame_index / nearest_frame_time_sec`): AV_T_WE_3 is 31 fps, HV_T_EW_2 is 29.97 fps.
- Frames byte-identical to the previous frame (constant-rate extraction over drops) are masked together with tracker-coasted frames as `invalid`.
- The GPS-derived-height re-run in Task 4 is dropped. Any re-run uses the OTC3D height.
- Two more clips were tracked (HV_T_EW_2, AV_T_WE_3). AV_T_WE_1 = track 28, AV_T_WE_3 = track 17, both picked from the image. HV_T_EW_2 is excluded as unidentifiable.
- The `site_extrinsic.py` note string no longer claims the height was established GPS-free.

Result: four standalone camera positions spread 75 m along the road. The shared fit and leave-one-clip-out are not reportable until the per-clip time alignment is resolved with the lab. README left unchanged for that reason. Full numbers in `outputs/evaluation/site_T_report.json` and the vault note.
