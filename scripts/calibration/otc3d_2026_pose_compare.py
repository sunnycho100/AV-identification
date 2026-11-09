"""Runs on the lab server from the OpenTrafficCam3D root, after pipeline_localization.py with
configs/gsv_loc_madison_todd_2026.yml (a copy of the July madison_todd folder plus 2025 and 2026 query frames).

Pose of each localized query relative to the July local highway plane (scale audit), GPS-free.

The resumed COLMAP mapper keeps the July frame (database cameras move a median 0.02 units), and the
July latest_metric_model is that frame scaled by transform_scale_mat (8.6119), so query poses map to
the audit's metric frame by that scale alone. Angles: pitch = forward axis below the deck plane,
roll = camera right axis above the plane, pan = heading change on the plane from the 2025 frame.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "trafficreloc"))
from utils.read_write_model import qvec2rotmat, read_images_binary  # noqa: E402

SFM = "database/recon_output/sfm_superpoint+superglue"
P = ROOT / "sample_data/processed_data"
QUERIES = ["y2025_AV_T_WE_1_150.jpg", "y2026_AV_T_EW_6_background.jpg", "y2026_AV_T_WE_4_020.jpg", "y2026_HV_T_WE_5_100.jpg"]


def poses(path):
    out = {}
    for im in read_images_binary(path).values():
        R = qvec2rotmat(im.qvec)
        out[im.name] = (R, -R.T @ im.tvec)
    return out


new = poses(P / "madison_todd_2026" / SFM / "latest_model/images.bin")
july = poses(P / "madison_todd" / SFM / "latest_model/images.bin")
s = float((P / "madison_todd" / SFM / "transform_scale_mat.txt").read_text().split()[0])

shared = [n for n in july if n in new and n != "AV_T_WE_1_150.jpg"]
d = np.array([np.linalg.norm(new[n][1] - july[n][1]) for n in shared]) * s
print(f"database cameras, new vs July frame: median {np.median(d):.2f} m, p90 {np.percentile(d, 90):.2f} m, "
      f"{(d > 1).sum()} of {len(d)} moved more than 1 m")
Rj, Cj = july["AV_T_WE_1_150.jpg"]
Rq, Cq = new[QUERIES[0]]
ang = np.degrees(np.arccos(np.clip((np.trace(Rq @ Rj.T) - 1) / 2, -1, 1)))
print(f"2025 frame relocalized vs July: centre {s * np.linalg.norm(Cq - Cj):.2f} m apart, rotation {ang:.3f} deg apart")

audit = json.loads((Path.home() / "roadside-camera/BEVHeights/outputs/calibration/camera-data/AV_T_WE_1/otc3d_scale_audit.json").read_text())
abcd = np.array(audit["method"]["local_highway_plane"]["equation_abcd"])
n, d0 = abcd[:3] / np.linalg.norm(abcd[:3]), abcd[3] / np.linalg.norm(abcd[:3])

out = {}
for q in QUERIES:
    if q not in new:
        print(q, "not registered")
        continue
    R, C = new[q]
    Cm = s * C
    h = n @ Cm + d0
    u = n * np.sign(h)
    right, _, fwd = R
    fp = fwd - (fwd @ u) * u
    out[q] = {"pitch_deg": float(np.degrees(np.arcsin(-fwd @ u))), "roll_deg": float(np.degrees(np.arcsin(right @ u))),
              "height_above_deck_m": float(abs(h)), "_fwd_on_plane": (fp / np.linalg.norm(fp)).tolist(), "_u": u.tolist()}
f0 = np.array(out[QUERIES[0]]["_fwd_on_plane"])
for q, v in out.items():
    f1, u = np.array(v["_fwd_on_plane"]), np.array(v["_u"])
    v["pan_from_2025_deg"] = float(np.degrees(np.arctan2(np.cross(f0, f1) @ u, f0 @ f1)))
    print(q, {k: round(x, 3) for k, x in v.items() if not k.startswith("_")})
json.dump(out, open(ROOT / "y2026_pose_compare.json", "w"), indent=1)
