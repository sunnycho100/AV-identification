"""Grade detector runs of one clip against GPS on the image-identified target.

One line per run suffix: speed ratio, chord ratio, rigid RMSE, along-ray error
by range band, heading axis error. Appends to the calibration-loop ledger
(outputs/evaluation/calibration_loop_ledger.csv) so results survive a context
reset. See docs/superpowers/plans/2026-09-30-site-calibration-loop.md.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/grade_calibration_candidate.py \
        --clip AV_T_EW_3 --runs phase1 h156 trafcam trafcam_h156 --label "h15.1 dp-0.31"
"""
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/evaluation"), str(ROOT / "scripts/tracking")]
import compare_height_runs as C   # noqa: E402
import grade_target_vs_gps as g   # noqa: E402
import score_heading as sh        # noqa: E402

LEDGER = ROOT / "outputs/evaluation/calibration_loop_ledger.csv"
COLS = ["date", "clip", "run", "note", "n", "speed_ratio", "chord", "rigid_rmse_m",
        "err_0_40", "err_40_60", "err_60_80", "err_80_120", "heading_axis_deg"]


def grade(clip, run):
    ref_run, ref_tid, _ = C.REFERENCE[clip]
    ref_px = C.to_px(clip, ref_run, C.load_track(clip, ref_run, ref_tid))
    tid, _ = C.pick_target(clip, run, ref_px)
    tracks = json.loads((ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}" / "tracks.json").read_text())
    r = g.score(tracks["tracks"][tid], g.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv"))
    hd = sh.score_clip(sh.load_tracks(ROOT / "outputs/tracking/camera-data" / f"{clip}_{run}" / "tracks.json"),
                       None, str(ROOT / "outputs/object_detection/camera-data" / f"{clip}_{run}"))
    b = {k: v["depth_err_median_m"] for k, v in r["depth_err_by_range"].items()}
    return {"n": r["n"], "speed_ratio": round(r["speed_ratio"], 3), "chord": round(r["scale_chord"], 3),
            "rigid_rmse_m": round(r["rmse_rigid_m"], 2),
            **{f"err_{k.replace('-', '_').rstrip('m')}": round(b[k], 2) if k in b else "" for k in
               ("0-40m", "40-60m", "60-80m", "80-120m")},
            "heading_axis_deg": round(hd["median_folded_deg"], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()
    new = not LEDGER.exists()
    for run in a.runs:
        r = grade(a.clip, run)
        print(f"{run:22s} n={r['n']:3d} speed {r['speed_ratio']:.3f} chord {r['chord']:.3f} "
              f"RMSE {r['rigid_rmse_m']:.2f} | along-ray {r['err_0_40']}/{r['err_40_60']}/{r['err_60_80']}/"
              f"{r['err_80_120']} | heading {r['heading_axis_deg']:.1f}")
        if not a.no_ledger:
            with open(LEDGER, "a", newline="") as f:
                w = csv.DictWriter(f, COLS)
                if new:
                    w.writeheader()
                    new = False
                w.writerow({"date": datetime.date.today().isoformat(), "clip": a.clip, "run": run,
                            "note": a.note, **r})


if __name__ == "__main__":
    main()
