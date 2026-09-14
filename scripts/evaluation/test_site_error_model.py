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
                              gps_at, kinematics, load_gps, load_track, loco, rot, wrap)

# C=(40,-15) m; camera x axis bearing 0.05 rad for pose P0 and 0.02 for pose P1;
# s=0.957; lever arm 1.2 m forward, 0.1 m left. A clip driving toward the camera has
# heading beta+pi, one driving away has heading beta: the heading flip is what
# separates the lever arm from the camera position.
TRUE = np.array([40.0, -15.0, 0.05, 0.02, 0.957, 1.2, 0.1])


def synth(pose, theta, toward_camera, n=80, name=None):
    """Straight constant-speed run, camera x from 25 to 90 m, exact under the model."""
    beta, C, s, l = theta[2 + POSES.index(pose)], theta[:2], theta[P_S], theta[P_LX:]
    x = np.linspace(90, 25, n) if toward_camera else np.linspace(25, 90, n)
    q = np.stack([x, np.full(n, -5.0)], 1)
    psi = beta + (np.pi if toward_camera else 0.0)
    p = C + s * (q @ rot(beta).T)
    g = p - rot(psi) @ l
    return {"name": name or pose, "pose": pose, "t": np.arange(n) / FPS, "q": q,
            "yaw": np.full(n, psi - beta), "g": g, "psi": np.full(n, psi),
            "v": np.full(n, abs(x[1] - x[0]) * FPS * s), "invalid": np.zeros(n, bool)}


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
    assert free_mask([c]).tolist() == [True, True, True, False, True, False, False]
    assert free_mask([c, synth("P0", TRUE, True)]).tolist() == [True, True, True, False, True, False, False]


def test_loco_skips_unidentifiable_holdouts():
    # one clip per direction: each hold-out leaves no clip in its own direction,
    # so the lever arm would be absorbed (2.4 m error on perfect data). Must skip.
    a, b = synth("P0", TRUE, True, name="toward"), synth("P0", TRUE, False, name="away")
    r = loco([a, b])
    assert "skipped" in r["toward"] and "skipped" in r["away"], r
    # two per direction: every hold-out is identifiable and exact
    r = loco([a, b, synth("P0", TRUE, True, name="toward2"), synth("P0", TRUE, False, name="away2")])
    assert all("skipped" not in v for v in r.values()), r
    assert max(v["excluding_invalid"]["rmse_m"] for v in r.values()) < 1e-3, r


def test_kinematics_skips_frame_gaps():
    c = synth("P0", TRUE, True)
    assert "skipped" not in kinematics(c)
    c["t"] = np.r_[c["t"][:40], c["t"][40:] + 0.5]
    assert "skipped" in kinematics(c)


def test_enu_and_heading_are_right_handed():
    e, n = enu_from_latlon(43.001, -89.0, 43.0, -89.0)
    assert abs(n - 111.3) < 0.2 and abs(e) < 1e-9, (e, n)
    gps = {"t": np.array([0.0, 1.0]), "e": np.zeros(2), "n": np.zeros(2),
           "hx": np.array([1.0, 0.0]), "hn": np.array([0.0, -1.0]),   # east, then south
           "v": np.ones(2)}
    _, psi, _ = gps_at(gps, np.array([0.0, 1.0]))
    assert np.allclose(psi, [0.0, -np.pi / 2]), psi


def test_fps_recovered_from_csv():
    hdr = ["latitude", "longitude", "east_velocity", "north_velocity", "processed_heading_x",
           "processed_heading_y", "video_time_sec", "nearest_frame_index", "nearest_frame_time_sec"]
    rows = [[43.0, -89.0, 1, 0, 1, 0, 0.0, 0, 0.0], [43.0, -89.0, 1, 0, 1, 0, 1.0, 31, 1.0]]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write(",".join(hdr) + "\n" + "\n".join(",".join(map(str, r)) for r in rows))
    assert abs(load_gps(f.name, 43.0, -89.0)["fps"] - 31.0) < 1e-9


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
