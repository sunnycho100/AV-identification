"""Write a calibration variant: the clip's VP rotation with a pitch (and roll)
correction and a chosen height, in the same JSON schema as site_extrinsic.py.

The corrections are applied in the camera frame exactly as
fit_height_from_gps_distances.ground does (R' = Rz(roll) Rx(pitch) R), so a
fitted (h, dp) carries over unchanged.

    .venv/bin/python scripts/calibration/make_extrinsic_variant.py --clip AV_T_EW_3 \
        --height 15.1 --dpitch-deg -0.31 --out-name metric_extrinsic_h151_dp031.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_height_from_gps_distances import rot_x, rot_z   # noqa: E402


def variant(R, height, dpitch_deg=0.0, droll_deg=0.0):
    Rc = rot_z(math.radians(droll_deg)) @ rot_x(math.radians(dpitch_deg)) @ R
    return Rc, -Rc @ np.array([0.0, 0.0, height])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--dpitch-deg", type=float, default=0.0)
    ap.add_argument("--droll-deg", type=float, default=0.0)
    ap.add_argument("--base", default="metric_extrinsic_site.json")
    ap.add_argument("--out-name", required=True)
    a = ap.parse_args()
    cal = ROOT / "outputs/calibration/camera-data" / a.clip
    base = json.loads((cal / a.base).read_text())
    R, t = variant(np.array(base["rotation"]), a.height, a.dpitch_deg, a.droll_deg)
    (cal / a.out_name).write_text(json.dumps({
        "rotation": R.tolist(), "translation": t.tolist(),
        "note": f"{a.base} rotation with pitch {a.dpitch_deg:+.3f} deg, roll {a.droll_deg:+.3f} deg "
                f"(camera frame), height {a.height} m. Calibration-loop variant, see "
                f"docs/superpowers/plans/2026-09-30-site-calibration-loop.md",
        "diagnostics": {"height_m": a.height, "dpitch_deg": a.dpitch_deg, "droll_deg": a.droll_deg,
                        "base": a.base}}, indent=2))
    print(f"wrote {cal / a.out_name}")


def _selfcheck():
    """Camera height must come out as asked, and dp must match the fit's ground()."""
    from fit_height_from_gps_distances import ground
    sys.path.insert(0, str(ROOT))
    from scripts.calibration.vp_extrinsic_from_frame import solve_pose
    K = np.array([[1532.2, 0, 961.7], [0, 1514.3, 539.7], [0, 0, 1]])
    R, _, _, _ = solve_pose(K, (156.6, 21.9), 16.26)
    Rc, t = variant(R, 15.1, -0.31)
    assert abs((-Rc.T @ t)[2] - 15.1) < 1e-9
    p = np.array([60.0, -5.0, 0.0])
    uv = K @ (Rc @ p + t)
    g = ground((uv[:2] / uv[2])[None], K, R, 15.1, math.radians(-0.31))[0]
    assert np.allclose(g, p[:2], atol=1e-6), g
    print("selfcheck ok (height, and the same pitch convention as the GPS-distance fit)")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else main()
