"""Re-solve yaw alone by aligning the projected box wireframe to image gradients.

Why. BEVHeight gives a full 3D box per frame, and x, y, z, l, w, h are believable;
the one number the roadside domain gap hurts is yaw. So keep the box and re-fit
that single parameter against the image the detection came from. For each
detection the box is rotated through the semicircle around its own axis, the 12
wireframe edges are projected with K and lidar2cam, and each candidate yaw is
scored by the mean Sobel gradient sampled along those edges. The four
ground-contact edges are weighted `ground_w` because the tire line and the near
shadow edge are the sharp, unambiguous edges at this camera height; the roof and
vertical edges smear.

Two objectives. `magnitude` is the recipe's own, the mean gradient magnitude,
which counts any strong gradient under the edge whatever its direction.
`normal` is the absolute gradient component perpendicular to the projected edge,
so a silhouette boundary, whose gradient points across the boundary, scores and
body texture or a road marking running along the edge does not.

Sign. The sweep is over the axis only, offsets in [-90, +90) degrees around the
detection's own yaw, so the answer is always within 90 degrees of what BEVHeight
predicted and the front-to-back sign is whatever the detector already chose.
A 180 degree sweep would happily report the flipped solution, and the frozen
score would then show a folded win paid for with a worse flip fraction. This
candidate does not touch flips.

Confidence is best peak over second-best local maximum of the score curve, which
is near 1 when the gradient landscape is flat (far range, small box) and large
when one axis clearly wins. `min_conf` drops the ambiguous fits back to the raw
yaw, `max_range` does the same past a given x, since the landscape is expected to
flatten out around 70 m.

Measured, and it does not work. On the five held-out clips with tracks on disk
the mean median folded error goes 8.884 to 19.321 degrees, worst in the bin the
recipe expected it to own: 0 to 40 m goes 9.70 to 26.65 degrees. Median
confidence is 1.03, so the score curve is essentially flat and the argmax is
noise; the chosen offset from the raw yaw has a median magnitude of 33 degrees
inside 40 m. Mean gradient magnitude does not separate a car's silhouette from
its own body texture and the road markings under it at these box sizes (40 to 90
px wide). `objective=normal` was added to test exactly that and it does not
rescue the method: 8.884 to 16.356 degrees, still worse in every bin, still
median confidence 1.04. Perpendicular gradient is a better objective than
magnitude, by about 3 degrees of mean median folded error, and the gap to the
raw detector yaw is far larger than the gap between the two objectives. The
projected box is 40 to 90 px wide here and nothing at that scale separates the
silhouette from the clutter under it, so the next thing worth trying is a
segmentation mask, not another edge score.

Corners follow evaluators.result2kitti.get_lidar_3d_8points exactly (bottom face
0-3 at z, top face 4-7 at z + h, x along l, y along w), built inline and
vectorised over the yaw sweep; that module is not importable here because it
pulls in numba. The test asserts the two agree.

    cfg: objective (magnitude or normal), step_deg (2), ground_w (3),
         min_conf (1.0, off), max_range (1e9, off), samples (20 points per edge)
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import score_heading as sh

OBJECTIVE = "magnitude"
STEP_DEG = 2.0
GROUND_W = 3.0
MIN_CONF = 1.0
MAX_RANGE = 1e9
SAMPLES = 20

# unit box, centre at (0, 0) in x/y, base at z = 0, scaled by (l, w, h)
CORNER_UNITS = np.array([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0],
                         [-0.5, -0.5, 0.0], [-0.5, 0.5, 0.0],
                         [0.5, 0.5, 1.0], [0.5, -0.5, 1.0],
                         [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0]])
# the four ground-contact edges first, then roof, then verticals
EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)])
N_GROUND = 4


@lru_cache(maxsize=None)
def load_boxes(det_dir, frame):
    """One frame of raw detections as (N, 7): x, y, z, l, w, h, yaw.

    Same file and same order as score_heading.load_dets, so an index from the
    match below picks exactly the detection the frozen score is grading.
    """
    p = Path(det_dir) / f"{int(frame):03d}_pred.json"
    if not p.exists():
        return np.zeros((0, 7))
    d = json.load(open(p))
    return np.array([[s["x"], s["y"], s["z"], s["l"], s["w"], s["h"], s["yaw"]]
                     for s in d], float).reshape(-1, 7)


def match_indices(track, det_dir):
    """Detection index per state, -1 where none is within score_heading.MATCH_M.

    This is score_heading.detection_yaw's matching rule, returning the index
    instead of the yaw so the rest of the box comes along.
    """
    out = np.full(len(track), -1, int)
    for i, s in enumerate(track):
        dets = sh.load_dets(str(det_dir), s["frame"])
        if not len(dets):
            continue
        d = np.hypot(dets[:, 0] - s["x"], dets[:, 1] - s["y"])
        j = int(np.argmin(d))
        if d[j] <= sh.MATCH_M:
            out[i] = j
    return out


def corners(box, yaws):
    """(A, 8, 3) box corners in the ground frame, one set per yaw in `yaws`."""
    x, y, z, l, w, h = box[:6]
    p = CORNER_UNITS * np.array([l, w, h])
    c, s = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
    return np.stack([c * p[:, 0] - s * p[:, 1] + x,
                     s * p[:, 0] + c * p[:, 1] + y,
                     np.broadcast_to(p[:, 2] + z, (len(yaws), 8))], axis=-1)


def edge_scores(box, yaws, grad, K, lidar2cam, samples=SAMPLES,
                objective=OBJECTIVE):
    """Mean gradient response along each projected edge, (A, 12), or None.

    `grad` is (H, W, 2), the Sobel gx and gy. None when any corner falls behind
    the camera at any yaw: the projection is meaningless there and the caller
    falls back to the raw yaw.
    """
    pts = corners(box, yaws)
    cam = pts @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
    if not (cam[..., 2] > 1e-6).all():
        return None
    uv = cam @ K.T
    uv = uv[..., :2] / uv[..., 2:]
    a, b = uv[:, EDGES[:, 0]], uv[:, EDGES[:, 1]]
    t = np.linspace(0.0, 1.0, samples)[None, None, :, None]
    p = a[:, :, None, :] + (b - a)[:, :, None, :] * t
    h, w = grad.shape[:2]
    # ponytail: nearest-pixel sampling, bilinear buys nothing on a blurred
    # gradient field; swap it in if sub-pixel edges ever matter
    u = np.rint(p[..., 0]).astype(int)
    v = np.rint(p[..., 1]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    g = grad[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)]
    if objective == "normal":
        d = b - a
        n = np.stack([-d[..., 1], d[..., 0]], -1) / np.maximum(
            np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
        val = np.abs((g * n[:, :, None, :]).sum(-1))
    elif objective == "magnitude":
        val = np.hypot(g[..., 0], g[..., 1])
    else:
        raise ValueError(f"objective must be magnitude or normal, not {objective!r}")
    return (val * ok).sum(-1) / np.maximum(ok.sum(-1), 1)


def solve_yaw(box, grad, K, lidar2cam, step_deg=STEP_DEG, ground_w=GROUND_W,
              samples=SAMPLES, objective=OBJECTIVE):
    """(yaw, confidence) for one box, or (None, 0.0) if it cannot be projected.

    The sweep is the semicircle around the box's own yaw, so the returned angle
    keeps the detection's front-to-back sign.
    """
    offsets = np.deg2rad(np.arange(-90.0, 90.0, step_deg))
    yaw0 = float(box[6])
    scores = edge_scores(box, yaw0 + offsets, grad, K, lidar2cam, samples, objective)
    if scores is None:
        return None, 0.0
    weight = np.where(np.arange(len(EDGES)) < N_GROUND, ground_w, 1.0)
    s = scores @ weight / weight.sum()

    n = len(s)
    i = int(np.argmax(s))
    lo, hi = s[(i - 1) % n], s[(i + 1) % n]
    denom = lo - 2 * s[i] + hi
    shift = 0.5 * (lo - hi) / denom if denom != 0 else 0.0
    yaw = yaw0 + np.deg2rad((-90.0 + (i + np.clip(shift, -1, 1)) * step_deg))

    peaks = s[(s >= np.roll(s, 1)) & (s >= np.roll(s, -1))]
    peaks = np.sort(peaks)[::-1]
    conf = float(peaks[0] / peaks[1]) if len(peaks) > 1 and peaks[1] > 0 else np.inf
    return float(yaw), conf


@lru_cache(maxsize=2)
def frame_grad(clip, frame):
    """(H, W, 2) Sobel gx and gy of one frame, or None when the frame is missing.

    Cached at size 2 only: states are visited frame by frame, and a full clip of
    float32 gradients would be gigabytes.
    """
    p = ROOT / "data/camera-data" / clip / "frames_all" / f"{int(frame):03d}.jpg"
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return np.stack([cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3),
                     cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)], -1)


def run(clip, det_dir, tracks, cfg):
    objective = str(cfg.get("objective", OBJECTIVE))
    step_deg = float(cfg.get("step_deg", STEP_DEG))
    ground_w = float(cfg.get("ground_w", GROUND_W))
    min_conf = float(cfg.get("min_conf", MIN_CONF))
    max_range = float(cfg.get("max_range", MAX_RANGE))
    samples = int(cfg.get("samples", SAMPLES))

    cal = json.load(open(Path(det_dir) / "calibration_used.json"))
    K = np.array(cal["K"], float)
    lidar2cam = np.array(cal["lidar2cam"], float)

    # one pass per frame, because the gradient image is the expensive part
    per_frame = {}
    for tid, track in tracks.items():
        for s, j in zip(track, match_indices(track, det_dir)):
            if j >= 0:
                per_frame.setdefault(s["frame"], []).append((tid, j))

    out = {tid: {} for tid in tracks}
    n_state = n_far = n_lowconf = n_unprojectable = 0
    confs = []
    for frame in sorted(per_frame):
        grad = frame_grad(clip, frame)
        if grad is None:
            continue
        boxes = load_boxes(str(det_dir), frame)
        for tid, j in per_frame[frame]:
            n_state += 1
            box = boxes[j]
            if box[0] > max_range:
                n_far += 1
                continue
            yaw, conf = solve_yaw(box, grad, K, lidar2cam, step_deg, ground_w,
                                  samples, objective)
            if yaw is None:
                n_unprojectable += 1
                continue
            confs.append(min(conf, 10.0))
            if conf < min_conf:
                n_lowconf += 1
                continue
            out[tid][frame] = yaw
    kept = sum(len(v) for v in out.values())
    print(f"  {clip} [{objective}]: {kept}/{n_state} states re-solved, "
          f"{n_far} past max_range, {n_lowconf} under min_conf, "
          f"{n_unprojectable} unprojectable, "
          f"median conf {np.median(confs) if confs else float('nan'):.3f}")
    return out
