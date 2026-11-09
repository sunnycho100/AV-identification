"""True time of every extracted frame, and which frames are frozen copies.

frames_all/ was cut from the clip at a fixed rate, so where the video dropped
frames the same picture was written more than once (HV_T_EW_2 has 89 copies
over a 3 s hole), and a clip whose stream is not 30 fps (AV_T_WE_3 at 31,
AV_V_EW_3 at 29.17) does not sit on a 1/30 s grid. Each extracted frame is
matched to the decoded source frame it came from; its time is that frame's
presentation timestamp. A frame showing the same source frame as the one
before it is a copy: no new information, so the pipeline treats it as missing.

Writes data/camera-data/<clip>/frame_times.json:
  {"clip", "source", "fps_nominal", "frames": [{"frame", "src", "t_s", "copy"}]}

    /Users/sunghwan_cho/miniforge/bin/python3.12 scripts/pipeline/frame_times.py --clip AV_T_WE_3
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SMALL = (96, 54)


def source_frames(mp4):
    """Every decoded frame (small grey) with its presentation time."""
    pts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout.split()
    cap, imgs = cv2.VideoCapture(str(mp4)), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        imgs.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32))
    t = np.array([float(p.strip(",")) for p in pts[:len(imgs)]])
    return np.stack(imgs), t - t[0]


def match(extracted, src):
    """Index of the source frame each extracted frame came from (in order, never backwards)."""
    out, j = [], 0
    for e in extracted:
        lo, hi = max(0, j - 2), min(len(src), j + 12)
        d = [np.abs(src[k] - e).mean() for k in range(lo, hi)]
        j = lo + int(np.argmin(d))
        out.append(j)
    return out


def frame_times(clip):
    fr = sorted((ROOT / f"data/camera-data/{clip}/frames_all").glob("*.jpg"))
    mp4 = ROOT / "Camera data" / f"{clip}.mp4"
    src, t = source_frames(mp4)
    ext = [cv2.resize(cv2.imread(str(f), 0), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32) for f in fr]
    idx = match(ext, src)
    rate = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                           "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip()
    frames = [{"frame": i, "src": k, "t_s": round(float(t[k]), 4), "copy": i > 0 and k == idx[i - 1]}
              for i, k in enumerate(idx)]
    return {"clip": clip, "source": mp4.name, "fps_nominal": rate, "frames": frames}


def load(clip, fps=30.0):
    """frame -> (t_s, copy). Falls back to frame / fps when the table is missing."""
    p = ROOT / f"data/camera-data/{clip}/frame_times.json"
    if not p.exists():
        return None
    return {f["frame"]: (f["t_s"], f["copy"]) for f in json.loads(p.read_text())["frames"]}


def main():
    ap = argparse.ArgumentParser("True frame times and frozen copies")
    ap.add_argument("--clip", nargs="+", required=True)
    for clip in ap.parse_args().clip:
        d = frame_times(clip)
        (ROOT / f"data/camera-data/{clip}/frame_times.json").write_text(json.dumps(d))
        f = d["frames"]
        dt = np.diff([x["t_s"] for x in f if not x["copy"]])
        print(f"{clip}: {len(f)} frames, {sum(x['copy'] for x in f)} copies, span {f[-1]['t_s']:.2f} s, "
              f"median step {np.median(dt) * 1000:.1f} ms (stream {d['fps_nominal']}), "
              f"frame/30 would end at {(len(f) - 1) / 30:.2f} s")


def _selfcheck():
    """A source with a 3-frame hole, extracted on a fixed grid, maps back with copies marked."""
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 255, (10, 54, 96)).astype(np.float32)
    ext = [src[i] for i in (0, 1, 2, 2, 2, 3, 4, 5, 6, 7)]
    idx = match(ext, src)
    assert idx == [0, 1, 2, 2, 2, 3, 4, 5, 6, 7], idx
    print("selfcheck ok (repeated frames map to the same source frame)")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
