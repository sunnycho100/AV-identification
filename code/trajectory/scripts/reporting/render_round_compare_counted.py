"""Round 1 vs round 2 side by side, with the frames each side is credited for marked.

A frame is credited when the marked Lincoln track has a detection in it (the "observed"
frames that seconds tracked counts; frames where the tracker only predicts are drawn orange
by render_trajectories.py but not counted). Each half's bottom bar says whether that frame
counts. A timeline under the video shows the whole pass, one row per round (orange = counted),
with the coverage window and a cursor at the current frame; `sheet` stacks the 9 timelines.

    python3 scripts/reporting/render_round_compare_counted.py          # video
    python3 scripts/reporting/render_round_compare_counted.py sheet    # timelines only
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/evaluation")]
import heldout_eval as he           # noqa: E402
import round2_same_frames as sf     # noqa: E402
from gps_to_image import load_clip, load_site   # noqa: E402

TAGS = (("hb30", "Round 1"), ("r2fullhb", "Round 2"))
VID = ROOT / "outputs/reports/2026-11-04_new_todd/video"
SHEET = ROOT / "outputs/reports/2026-11-04_new_todd/annotated/round1_vs_round2_timelines.jpg"
WIN = ROOT / "outputs/evaluation/coverage_window/windows.json"
ORANGE, GRAY = (0, 140, 255), (160, 160, 160)
X0, X1 = 170, 1900


def counted(clip, T):
    gps = load_clip(clip, he.tag(clip, "S4"))["gps"]
    out = []
    for tag, _ in TAGS:
        d = sf.load(clip, tag, T, gps)
        out.append(set() if d is None else set(d["frame"][d["observed"]].tolist()))
    return out


def strip(cnt, n, win, cur=None):
    """1920 x 80 timeline: one row per round, orange where counted, window edges in white."""
    img = np.zeros((80, 1920, 3), np.uint8)
    x = lambda f: int(X0 + (X1 - X0) * f / n)
    for r, ((_, name), c) in enumerate(zip(TAGS, cnt)):
        y = 12 + r * 30
        cv2.putText(img, name, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(img, (X0, y), (X1, y + 22), (60, 60, 60), -1)
        for f in c:
            cv2.rectangle(img, (x(f), y), (x(f + 1), y + 22), ORANGE, -1)
    for f in (win["entry"], win["exit"] + 1):
        cv2.line(img, (x(f), 4), (x(f), 76), (255, 255, 255), 2)
    if cur is not None:
        cv2.line(img, (x(cur), 0), (x(cur), 79), (0, 255, 255), 3)
    return img


def frame(clip, i, cnt, n, win):
    tiles = []
    for (tag, name), c in zip(TAGS, cnt):
        im = cv2.resize(cv2.imread(str(ROOT / f"outputs/reports/trajectories/{clip}_{tag}/{i:03d}.jpg")), (960, 540))
        cv2.rectangle(im, (0, 490), (960, 540), (0, 0, 0), -1)
        on = i in c
        cv2.putText(im, f"{name}: Lincoln {'counted' if on else 'not counted'}", (12, 524),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ORANGE if on else GRAY, 2)
        tiles.append(im)
    return np.vstack([np.hstack(tiles), strip(cnt, n, win, i)])


def sheet(T, W):
    rows = []
    for clip in he.HELD:
        n = len(list((ROOT / f"outputs/reports/trajectories/{clip}_hb30").glob("*.jpg")))
        title = np.zeros((36, 1920, 3), np.uint8)
        cv2.putText(title, f"{clip}: window = between the white lines", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        rows += [title, strip(counted(clip, T), n, W[clip])]
    cv2.imwrite(str(SHEET), np.vstack(rows))
    print(SHEET)


def main():
    T, W = load_site(), json.loads(WIN.read_text())
    if sys.argv[1:] == ["sheet"]:
        return sheet(T, W)
    tmp = VID / "_counted_frames"
    tmp.mkdir(exist_ok=True)
    k = 0
    for clip in he.HELD:
        cnt = counted(clip, T)
        n = len(list((ROOT / f"outputs/reports/trajectories/{clip}_hb30").glob("*.jpg")))
        for i in range(n):
            cv2.imwrite(str(tmp / f"{k:05d}.jpg"), frame(clip, i, cnt, n, W[clip]))
            k += 1
        print(clip, f"round 1 {len(cnt[0])} frames, round 2 {len(cnt[1])}, round 2 only {len(cnt[1] - cnt[0])}, round 1 only {len(cnt[0] - cnt[1])}")
    out = VID / "all9_round1_vs_round2_counted.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp / "%05d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out)], check=True)
    for p in tmp.glob("*.jpg"):
        p.unlink()
    tmp.rmdir()
    print(out)


if __name__ == "__main__":
    main()
