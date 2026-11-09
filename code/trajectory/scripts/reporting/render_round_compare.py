"""Side-by-side tracks of two runs on one clip (made by render_trajectories.py), with titles.

    python3 scripts/reporting/render_round_compare.py CLIP TAG_LEFT "TITLE LEFT" TAG_RIGHT "TITLE RIGHT" OUT.mp4
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    clip, tl, title_l, tr, title_r, out = sys.argv[1:7]
    out = ROOT / out
    tmp = out.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    dl, dr = (ROOT / "outputs/reports/trajectories" / f"{clip}_{t}" for t in (tl, tr))
    frames = sorted(dl.glob("*.jpg"))
    for f in frames:
        tiles = []
        for d, title in ((dl, title_l), (dr, title_r)):
            im = cv2.resize(cv2.imread(str(d / f.name)), (960, 540))
            cv2.rectangle(im, (0, 490), (960, 540), (0, 0, 0), -1)
            cv2.putText(im, title, (12, 524), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(im)
        cv2.imwrite(str(tmp / f.name), np.hstack(tiles))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(tmp / "%03d.jpg"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", str(out)], check=True)
    keep = sorted(tmp.glob("*.jpg"))
    cv2.imwrite(str(out.with_name(out.stem + "_still.jpg")), cv2.imread(str(keep[len(keep) * 3 // 5])))
    for p in keep:
        p.unlink()
    tmp.rmdir()
    print(out)


if __name__ == "__main__":
    main()
