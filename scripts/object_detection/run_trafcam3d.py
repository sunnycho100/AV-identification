"""Run trafcam_3d (Zhu et al., arXiv:2103.15293) on a clip with our calibration.

Why. BEVHeight's heading on this footage follows its own ego x axis rather than
the image (see the camalign test). trafcam_3d is a detector built for high
traffic cameras: the frame is warped onto the road plane, a YOLOv3 predicts
rotated boxes in that top-down image, and a "tail" marks each box's front, so
heading is a full 360 deg read from the image.

This driver replaces their dataset loader, which only knows their own datasets:
the road-plane homography comes from our K and site extrinsic (road at z = 0),
the top-down image is 8 px per metre like their BrnoCompSpeed setup, and the
same letterbox and multi-scale homographies are rebuilt for the dual-view model.
Boxes are written as <frame>_pred.json in the road frame, in the same schema as
run_bevheight_generic.py, so tracking and the heading score run unchanged.

Runs in trafcam_3d's own venv (their code is 2021-era):
    third_party/trafcam_3d/.venv/bin/python scripts/object_detection/run_trafcam3d.py \
        --clip AV_T_EW_3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAFCAM = ROOT / "third_party/trafcam_3d"
PX_PER_M = 8.0                       # their BrnoCompSpeed BEV scale
X_RANGE = (12.0, 108.0)              # along the road, 96 m like their widest Brno view
Y_RANGE = (-44.0, 12.0)              # across; covers both carriageways on Todd Drive
DAIR_GROUND_Z = -1.73                # our other detectors write the road at this z
DEFAULT_H = 1.5                      # the model has no height; a car-sized placeholder


def road_homography(K, lidar2cam):
    """Image pixel -> road (x, y) for the z = 0 plane, as a 3x3."""
    H_img_world = K @ np.column_stack([lidar2cam[:3, 0], lidar2cam[:3, 1], lidar2cam[:3, 3]])
    return np.linalg.inv(H_img_world)


FLIP = False   # --flip-bev: turn the top-down image 180 deg (far at the bottom)


def bev_axes():
    return ("y", "x") if FLIP else ("-y", "-x")


def bev_from_world():
    """Road (x, y) -> top-down pixel (u, v): far is up, the car's left is left
    (or both reversed under FLIP, which is the same picture turned 180 deg)."""
    if FLIP:
        return np.array([[0.0, PX_PER_M, -Y_RANGE[0] * PX_PER_M],
                         [PX_PER_M, 0.0, -X_RANGE[0] * PX_PER_M],
                         [0.0, 0.0, 1.0]])
    return np.array([[0.0, -PX_PER_M, Y_RANGE[1] * PX_PER_M],
                     [-PX_PER_M, 0.0, X_RANGE[1] * PX_PER_M],
                     [0.0, 0.0, 1.0]])


def bev_size():
    return (int(round((Y_RANGE[1] - Y_RANGE[0]) * PX_PER_M)),
            int(round((X_RANGE[1] - X_RANGE[0]) * PX_PER_M)))


def to_world(H_world_bev, uv):
    p = H_world_bev @ np.array([uv[0], uv[1], 1.0])
    return p[:2] / p[2]


def boxes_to_preds(det, H_world_bev):
    """(N, 9) x, y, w, h, r, tail_dx, tail_dy, conf, cls in BEV px -> our schema.

    Heading: the axis comes from the rotated box angle, and the tail only picks
    which end is the front. Their yaw2mat rotates by -r, so the long side points
    along -(r + 90 deg) in BEV pixels. On AV_T_EW_3 that axis is 2.5 deg from
    motion, and 5.3 with the ego frame turned 20 deg (so it follows the car, not
    the image layout); +(r + 90) looked equally good unrotated but read 43.6
    turned, and the tail alone gave 18.9.
    yaw_tail keeps the tail-only heading for comparison.
    """
    preds = []
    for x, y, w, h, r, tdx, tdy, conf, _ in det:
        c = to_world(H_world_bev, (x, y))
        front = to_world(H_world_bev, (x + tdx, y + tdy)) - c
        a = -(r + math.pi / 2)
        along = to_world(H_world_bev, (x + 10 * math.cos(a), y + 10 * math.sin(a))) - c
        if along @ front < 0:
            along = -along
        preds.append({"class_id": 0, "class_name": "car", "score": float(conf),
                      "x": float(c[0]), "y": float(c[1]), "z": DAIR_GROUND_Z,
                      "l": float(max(w, h) / PX_PER_M), "w": float(min(w, h) / PX_PER_M),
                      "h": DEFAULT_H,
                      "yaw": float(math.atan2(along[1], along[0])),
                      "yaw_tail": float(math.atan2(front[1], front[0])),
                      "bev": [float(v) for v in (x, y, w, h, r, tdx, tdy)]})
    return preds


def rotate_preds(preds, theta):
    """Model-frame boxes to the road frame when the ego frame was turned by theta."""
    c, s = math.cos(theta), math.sin(theta)
    wrap = lambda a: math.atan2(math.sin(a), math.cos(a))
    for p in preds:
        p["x"], p["y"] = c * p["x"] - s * p["y"], s * p["x"] + c * p["y"]
        p["yaw"], p["yaw_tail"] = wrap(p["yaw"] + theta), wrap(p["yaw_tail"] + theta)
    return preds


def footprint(p):
    """Four road-plane corners (x, y) of a pred and its front-centre point."""
    c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
    fwd, left = np.array([c, s]) * p["l"] / 2, np.array([-s, c]) * p["w"] / 2
    ctr = np.array([p["x"], p["y"]])
    return np.array([ctr + fwd + left, ctr + fwd - left, ctr - fwd - left, ctr - fwd + left]), ctr + fwd


def render(ori0, bev0, preds, K, lidar2cam):
    """Camera view and top-down view side by side; red edge and dot mark the front."""
    ori, bev = ori0.copy(), bev0.copy()
    Hbw = bev_from_world()

    def to_bev(pts):
        q = (Hbw @ np.c_[pts, np.ones(len(pts))].T).T
        return (q[:, :2] / q[:, 2:]).astype(int)

    def to_img(pts):
        q = (lidar2cam[:3, :3] @ np.c_[pts, np.zeros(len(pts))].T).T + lidar2cam[:3, 3]
        q = (K @ q.T).T
        return (q[:, :2] / q[:, 2:]).astype(int)

    for p in preds:
        corners, front = footprint(p)
        for img, f in ((bev, to_bev), (ori, to_img)):
            pc, pf = f(corners), f(front[None])[0]
            cv2.polylines(img, [pc.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.line(img, tuple(pc[0]), tuple(pc[1]), (0, 0, 255), 3)
            cv2.circle(img, tuple(pf), 4, (0, 0, 255), -1)
        cv2.putText(ori, f"{p['score']:.2f}", tuple(to_img(corners)[2]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    scale = ori.shape[0] / bev.shape[0]
    bev = cv2.resize(bev, (int(bev.shape[1] * scale), ori.shape[0]))
    return np.hstack([ori, bev])


def main():
    sys.path[:0] = [str(Path(__file__).resolve().parent / "trafcam_shims"),
                    str(TRAFCAM / "yolov3"), str(TRAFCAM / "bev")]
    import torch
    from models import Darknet
    from utils.datasets import letterbox
    from utils.utils import non_max_suppression
    from bev.bev import BEVWorldSpec
    from bev.calib import Calib

    ap = argparse.ArgumentParser("trafcam_3d on one clip with our calibration")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--suffix", default="trafcam")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.2)   # their run_detect.sh
    ap.add_argument("--iou-thres", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--extrinsic", default="metric_extrinsic_site.json",
                    help="extrinsic file under outputs/calibration/camera-data/<clip>/")
    ap.add_argument("--ego-rot-deg", type=float, default=0.0,
                    help="turn the ego frame so the road runs diagonally in the "
                         "top-down image: tests whether the axis follows the car or "
                         "the image layout. Boxes are written back in the road frame")
    ap.add_argument("--y-range", type=float, nargs=2, default=None)
    ap.add_argument("--flip-bev", action="store_true",
                    help="feed the top-down image turned 180 deg: tests whether the "
                         "front the model reports follows the car or the image layout")
    args = ap.parse_args()
    global FLIP, Y_RANGE
    FLIP = args.flip_bev
    if args.y_range:
        Y_RANGE = tuple(args.y_range)

    cal = ROOT / "outputs/calibration/camera-data" / args.clip
    K = np.array(json.loads(next(cal.glob("*_anycalib_pinhole_pinhole.json")).read_text())
                 ["prediction"]["intrinsics"][:4])
    K = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
    ext = json.loads((cal / args.extrinsic).read_text())
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3], lidar2cam[:3, 3] = ext["rotation"], np.ravel(ext["translation"])
    theta = math.radians(args.ego_rot_deg)
    Rz = np.eye(4)
    Rz[:2, :2] = [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    lidar2cam = lidar2cam @ Rz          # model frame: p_road = Rz p_model

    H_world_img = road_homography(K, lidar2cam)
    H_bev_world = bev_from_world()
    H_bev_img = H_bev_world @ H_world_img
    H_world_bev = np.linalg.inv(H_bev_world)
    bw, bh = bev_size()

    weights = TRAFCAM / "yolov3/weights/best_ra_tail_180_dv.pt"
    cfg = TRAFCAM / "yolov3/cfg/yolov3-spp-rboxttdv180.cfg"
    model = Darknet(str(cfg), args.img_size, rotated=True, half_angle=True, tail=True,
                    tail_inv=False, rotated_anchor=True, dual_view=True)
    # the 2021 checkpoint pickles a few numpy scalars next to the tensors; allow
    # just those rather than turning the safe loader off
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype,
                                           type(np.dtype(np.float64))]):
        state = torch.load(weights, map_location="cpu")["model"]
    # thop's profiler (run by their model_info) adds total_ops and total_params
    # counters to every module; they are not weights, so only those may be missing
    res = model.load_state_dict(state, strict=False)
    real_missing = [k for k in res.missing_keys if not k.endswith(("total_ops", "total_params"))]
    assert not real_missing and not res.unexpected_keys, (real_missing, res.unexpected_keys)
    model.eval()

    out_dir = ROOT / "outputs/object_detection/camera-data" / f"{args.clip}_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_used.json").write_text(json.dumps({
        "detector": "trafcam_3d best_ra_tail_180_dv.pt", "px_per_m": PX_PER_M,
        "x_range": X_RANGE, "y_range": Y_RANGE, "img_size": args.img_size,
        "conf_thres": args.conf_thres, "K": K.tolist(), "lidar2cam": lidar2cam.tolist(),
        "road_plane_z_in_output": DAIR_GROUND_Z}, indent=2))

    frames = sorted((ROOT / "data/camera-data" / args.clip / "frames_all").glob("*.jpg"))
    frames = frames[:args.limit] if args.limit else frames
    for f in frames:
        ori0 = cv2.imread(str(f))
        bev0 = cv2.warpPerspective(ori0, H_bev_img, (bw, bh), flags=cv2.INTER_LINEAR)
        bev, rb, pb = letterbox(bev0, new_shape=args.img_size)
        ori, ro, po = letterbox(ori0, new_shape=args.img_size)

        # the loader's three homographies (full, 1/8, 1/16), built with their own
        # Calib and BEVWorldSpec so the pixel conventions match training exactly
        calib = Calib(K=K, T=lidar2cam, u_size=ori0.shape[1], v_size=ori0.shape[0])
        calib = calib.scale(align_corners=False, scale_ratio_u=ro[0], scale_ratio_v=ro[1])
        calib = calib.pad(po[0], po[1], po[0], po[1])
        bspec = BEVWorldSpec(u_size=bw, v_size=bh, u_axis=bev_axes()[0], v_axis=bev_axes()[1],
                             x_min=X_RANGE[0], x_max=X_RANGE[1], y_min=Y_RANGE[0], y_max=Y_RANGE[1])
        bspec = bspec.scale(align_corners=False, scale_ratio_u=rb[0], scale_ratio_v=rb[1])
        bspec = bspec.pad(pb[0], pb[1], pb[0], pb[1])
        H = []
        for c, b in ((calib, bspec),
                     (calib.scale(align_corners=False, scale_ratio_u=1 / 8, scale_ratio_v=1 / 8),
                      bspec.scale(align_corners=False, scale_ratio_u=1 / 8, scale_ratio_v=1 / 8)),
                     (calib.scale(align_corners=False, scale_ratio_u=1 / 16, scale_ratio_v=1 / 16),
                      bspec.scale(align_corners=False, scale_ratio_u=1 / 16, scale_ratio_v=1 / 16))):
            H.append(np.linalg.inv(c.gen_H_world_img()) @ b.gen_H_world_bev())
        H_img_bev = torch.from_numpy(np.stack(H)).float()[None]

        to_t = lambda im: torch.from_numpy(np.ascontiguousarray(
            im[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255.0
        with torch.no_grad():
            pred = model(to_t(bev), x_sec_view=to_t(ori), H_img_bev=H_img_bev)[0]
        det = non_max_suppression(pred, args.conf_thres, args.iou_thres, multi_label=False,
                                  rotated=True, rotated_anchor=True, tail=True,
                                  invalid_masks=[None])[0]
        det = np.zeros((0, 9)) if det is None else det.numpy()
        # letterboxed bev px -> original bev px
        det[:, [0, 1]] = (det[:, [0, 1]] - np.array(pb)) / rb[0]
        det[:, [2, 3, 5, 6]] /= rb[0]
        preds = boxes_to_preds(det, H_world_bev)
        cv2.imwrite(str(out_dir / f"{f.stem}_vis.jpg"), render(ori0, bev0, preds, K, lidar2cam))
        rotate_preds(preds, theta)       # render draws in the model frame; json is road frame
        (out_dir / f"{f.stem}_pred.json").write_text(json.dumps(preds, indent=2))
        print(f"{f.name}: {len(preds)} cars")
    print(f"saved to {out_dir}")


def _selfcheck():
    """The BEV map must place road points where the ranges say, and a box
    written in BEV px must come back at its world position and heading."""
    Hbw = bev_from_world()
    Hwb = np.linalg.inv(Hbw)
    far_left = Hbw @ np.array([X_RANGE[1], Y_RANGE[1], 1.0])
    assert np.allclose(far_left[:2] / far_left[2], [0, 0]), far_left
    near_right = Hbw @ np.array([X_RANGE[0], Y_RANGE[0], 1.0])
    assert np.allclose(near_right[:2] / near_right[2], bev_size()), near_right
    # a car at (60, -5) facing +x (away): its front is up in the BEV image
    c = Hbw @ np.array([60.0, -5.0, 1.0])
    c = c[:2] / c[2]
    det = np.array([[c[0], c[1], 14.0, 36.0, 0.0, 0.0, -18.0, 0.9, 0.0]])
    p = boxes_to_preds(det, Hwb)[0]
    assert abs(p["x"] - 60) < 1e-9 and abs(p["y"] + 5) < 1e-9, p
    assert abs(p["yaw"]) < 1e-9 and abs(p["l"] - 4.5) < 1e-9 and abs(p["w"] - 1.75) < 1e-9, p
    # an ego frame turned by 20 deg comes back to the road frame exactly
    t = math.radians(20)
    road = np.array([60.0, -5.0])
    model = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]]) @ road
    back = rotate_preds([dict(p, x=model[0], y=model[1], yaw=-t, yaw_tail=-t)], t)[0]
    assert np.allclose([back["x"], back["y"]], road) and abs(back["yaw"]) < 1e-9, back
    print("selfcheck ok (BEV corners, box position, heading from box angle and tail, "
          "size, ego-rotation round trip)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
