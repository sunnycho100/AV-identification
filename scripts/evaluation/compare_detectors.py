"""One table comparing detectors run through the same pipeline (evaluation only).

Per detector tag, three parts:
  coverage   per range bin of the GPS car's distance along the road, the share of
             frames where a road-masked detection lies within 2 m of the GPS car,
             and the median along-road error (detection x minus GPS x) of those
             matches. A bias that grows with range is what a road that is not flat
             (Hang's concern for BEVHeight) or a wrong pitch would produce. GPS
             timing is the ft102 benchmark's fitted offset per clip, the same for
             every detector, so timing cannot favour one of them.
  benchmark  gps_benchmark.py on the image-identified car (held out AV_T_EW_3 row)
  scorecard  scorecard.py mean over the 5 clips (GPS-free)
Writes outputs/reports/monouni_compare/summary_table.md and .json.

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/evaluation/compare_detectors.py \
        --tags ft102 det2d monouni_A monouni_B
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
import gps_to_image as g2i   # noqa: E402

GRADED = ("AV_T_EW_3", "HV_T_EW_1", "AV_T_WE_1", "AV_T_WE_3")   # HV_T_EW_2: GPS car inside the video hole
BINS = ((15, 60), (60, 100), (100, 140), (140, 200))
MATCH_M = 2.0
PY = sys.executable


def coverage(clip, tag, T, tau):
    t, copy = g2i.frame_times(clip, len(list((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))))
    gps = g2i.load_gps(ROOT / "Camera data" / f"{clip}_trajectory.csv", *g2i.site_origin())
    det = ROOT / "outputs/object_detection/camera-data" / f"{clip}_{tag}_road"
    rows = []
    for f in np.flatnonzero(~copy):
        tt = t[f] + tau
        if tt < gps["t"][0] or tt > gps["t"][-1]:
            continue
        xy, _, _ = g2i.gps_road(gps, T, np.array([tt]))
        p = det / f"{f:03d}_pred.json"
        boxes = [b for b in json.loads(p.read_text()) if b["class_name"] == "car"] if p.exists() else []
        d = [np.hypot(b["x"] - xy[0, 0], b["y"] - xy[0, 1]) for b in boxes]
        k = int(np.argmin(d)) if d else None
        hit = d and d[k] < MATCH_M
        rows.append((xy[0, 0], bool(hit), boxes[k]["x"] - xy[0, 0] if hit else np.nan))
    return rows


def summarise(rows):
    a = np.array(rows, dtype=float).reshape(-1, 3)
    out = {}
    for lo, hi in BINS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() < 5:
            out[f"{lo}-{hi}"] = None
            continue
        hit = a[m, 1] > 0
        out[f"{lo}-{hi}"] = {"frames": int(m.sum()), "share_within_2m": round(float(hit.mean()), 3),
                             "along_err_median_m": round(float(np.nanmedian(a[m, 2])), 2) if hit.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser("Compare detectors through the same pipeline")
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    T = g2i.load_site()
    tau = {c: json.loads((ROOT / "outputs/evaluation/gps_benchmark" / f"{c}.json").read_text())["timing"]["best_offset_s"]
           for c in GRADED}
    res = {}
    for tag in a.tags:
        cov = {c: coverage(c, tag, T, tau[c]) for c in GRADED}
        bdir = ROOT / "outputs/evaluation" / ("gps_benchmark" if tag == "ft102" else f"gps_benchmark_{tag}")
        bench = {c: json.loads((bdir / f"{c}.json").read_text()) for c in GRADED if (bdir / f"{c}.json").exists()}
        sc = json.loads(subprocess.run([PY, "scripts/evaluation/scorecard.py", "--tag", tag, "--json"], cwd=ROOT,
                                       capture_output=True, text=True, check=True).stdout)
        res[tag] = {"coverage_all_graded": summarise([r for c in GRADED for r in cov[c]]),
                    "coverage_held_out": summarise(cov["AV_T_EW_3"]),
                    "benchmark": {c: {"vehicle": b.get("vehicle"), **b.get("timing_free", {}),
                                      **((b.get("timing") or {}).get("at_best_offset") or {})} for c, b in bench.items()},
                    "scorecard_mean": sc["mean"], "scorecard_rows": sc["rows"]}
    out = ROOT / "outputs/reports/monouni_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_table.json").write_text(json.dumps({"gps_offset_s_from_ft102": tau, "results": res}, indent=2))
    f = lambda v, k="share_within_2m": "" if v is None else (f"{v[k]:.0%}" if k == "share_within_2m" else f"{v[k]:+.1f}" if v[k] is not None else "")
    L = ["Coverage: share of GPS-car frames with a box within 2 m (all 4 graded clips), by distance along the road",
         "", "| Detector | " + " | ".join(f"{lo}-{hi} m" for lo, hi in BINS) + " | along error " +
         " | ".join(f"{lo}-{hi}" for lo, hi in BINS) + " |", "|" + "---|" * (2 * len(BINS) + 1)]
    for tag, r in res.items():
        c = r["coverage_all_graded"]
        L.append(f"| {tag} | " + " | ".join(f(c[f'{lo}-{hi}']) for lo, hi in BINS) + " | "
                 + " | ".join(f(c[f'{lo}-{hi}'], 'along_err_median_m') for lo, hi in BINS) + " |")
    L += ["", "Held-out AV_T_EW_3, image-identified car (gps_benchmark.py, at the fitted offset)", "",
          "| Detector | Car | Speed ratio | Speed MAE (mph) | Lateral bias (m) | Heading MAE (deg) | Accel rms cam / GPS / diff | Along rms (m) |",
          "|" + "---|" * 8]
    for tag, r in res.items():
        b = r["benchmark"].get("AV_T_EW_3")
        if not b or "speed_ratio" not in b:
            L.append(f"| {tag} | not identified |" + " |" * 6)
            continue
        L.append(f"| {tag} | {b['vehicle']} | {b['speed_ratio']} | {b['speed_mae_mps'] * 2.23694:.2f} | {b['lateral_bias_m']:+.2f} "
                 f"| {b['heading_mae_deg']:.2f} | {b['accel_rms_camera']:.2f} / {b['accel_rms_gps']:.2f} / {b['accel_rms_diff']:.2f} "
                 f"| {b['along_rms_m']:.2f} |")
    keys = ("frag_1s", "accel_steady", "lat_std", "follow_mph", "resid_m", "resid_ac1", "frozen")
    L += ["", "GPS-free scorecard, mean of 5 clips (scorecard.py)", "", "| Detector | " + " | ".join(keys) + " |",
          "|" + "---|" * (len(keys) + 1)]
    for tag, r in res.items():
        L.append(f"| {tag} | " + " | ".join(str(r["scorecard_mean"][k]) for k in keys) + " |")
    (out / "summary_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
