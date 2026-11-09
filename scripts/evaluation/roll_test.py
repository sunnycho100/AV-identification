"""Does the camera roll measured from building edges (+0.7 deg, 2026 pose, GPS-free; D26) remove the
direction-dependent speed and lateral bias? First look, no GPU (overnight E7).

Each round-2 detection is projected to its pixel with the current calibration and cast
back onto the road plane through a calibration rolled about the optical axis
(R' = rot_z(phi) R, camera centre kept, pitch and yaw untouched: sweep_roll.py arm A).
Exact for the ground contact point if the detector's image position is right; a
full re-detection with the rolled extrinsic is the follow-up if this helps. Decided
before running: +0.7 deg is the candidate (measured sign); -0.7 deg is a control.
Added 2026-11-08: +1.88 deg, OpenTrafficCam3D's 2026 roll against the local highway deck (camera
right axis 1.79 deg above the deck, mean of three 2026 frames; 1.79 / cos(17.5 deg pitch)), fixed
before running and reported only.

    python scripts/evaluation/roll_test.py make
    python scripts/evaluation/roll_test.py run
    python scripts/evaluation/roll_test.py score
"""
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
CLIPS = ["AV_T_EW_6", "AV_T_WE_4", "AV_T_WE_5", "HV_T_EW_5", "HV_T_EW_6", "HV_T_WE_5", "HV_T_WE_6"]   # 2026 pose
ROLLS = {"rollp07": 0.7, "rollm07": -0.7, "rollp19": 1.88}
CAM = ROOT / "outputs/object_detection/camera-data"


def rot_z(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def recast(p, K, L_old, L_new):
    """Point (x, y, z) in the output frame -> its pixel under L_old -> the plane z under L_new."""
    q = L_old[:3, :3] @ p + L_old[:3, 3]
    d = np.linalg.inv(K) @ (K @ q / q[2])
    R, t = L_new[:3, :3], L_new[:3, 3]
    C, dw = -R.T @ t, R.T @ d
    return C + (p[2] - C[2]) / dw[2] * dw


def make():
    for c in CLIPS:
        src = CAM / f"{c}_r2full_lo10"
        cal = json.loads((src / "calibration_used.json").read_text())
        K, L = np.array(cal["K"]), np.array(cal["lidar2cam"])
        E = json.loads((ROOT / cal["extrinsic_json"]).read_text())
        R, t = np.array(E["rotation"]), np.array(E["translation"])
        E4 = np.eye(4); E4[:3, :3], E4[:3, 3] = R, t
        S = np.linalg.inv(E4) @ L                                   # output frame shift (z + 1.73)
        Cw = -R.T @ t
        for tag, deg in ROLLS.items():
            R2 = rot_z(math.radians(deg)) @ R
            E2 = np.eye(4); E2[:3, :3], E2[:3, 3] = R2, -R2 @ Cw
            L2 = E2 @ S
            out = CAM / f"{c}_r2full_lo10_{tag}"
            shutil.rmtree(out, ignore_errors=True); out.mkdir()
            for p in src.glob("*_pred.json"):
                boxes = []
                for b in json.loads(p.read_text()):
                    a = recast(np.array([b["x"], b["y"], b["z"]]), K, L, L2)
                    f = recast(np.array([b["x"] + 2 * math.cos(b["yaw"]), b["y"] + 2 * math.sin(b["yaw"]), b["z"]]), K, L, L2)
                    boxes.append({**b, "x": float(a[0]), "y": float(a[1]), "yaw": math.atan2(f[1] - a[1], f[0] - a[0])})
                (out / p.name).write_text(json.dumps(boxes))
            (out / "calibration_used.json").write_text(json.dumps({**cal, "lidar2cam": L2.tolist(),
                                                                    "roll_test_deg": deg, "note": "roll_test.py recast of r2full_lo10"}))
            print(c, tag, "done")


def run():
    import lowscore_sweep as ls
    for c in CLIPS:
        for tag in ROLLS:
            g = f"r2fullhb_{tag}"
            if not (ls.TRAJ / f"{c}_{g}" / "tracks.json").exists():
                ls.sh([sys.executable, "scripts/pipeline/extract_trajectories.py", "--clip", c, "--tag", g, "--det-tag", f"r2full_lo10_{tag}",
                       "--track-low-thresh", 0.30, "--low-noise", 4, "--jumps", "remove", "--smooth", "rts", "--q-vel", 0.1,
                       "--low-filter", "confirm", "--confirm-floor", 0.30])
            print(c, g, "car", ls.mark(c, g))


def _selfcheck():
    K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])
    L = np.eye(4); L[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]) @ np.array(
        [[math.cos(0.3), 0, math.sin(0.3)], [0, 1, 0], [-math.sin(0.3), 0, math.cos(0.3)]]); L[:3, 3] = -L[:3, :3] @ [0, 0, 15.1]
    p = np.array([40.0, -10.0, -1.73])
    assert np.allclose(recast(p, K, L, L), p, atol=1e-6)            # same calibration: point unchanged


if __name__ == "__main__":
    _selfcheck()
    {"make": make, "run": run}[sys.argv[1]]()
