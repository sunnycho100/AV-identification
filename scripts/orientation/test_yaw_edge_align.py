"""Self-check for the yaw_edge_align candidate.

Two things have to hold. The inline corner builder must reproduce
evaluators.result2kitti.get_lidar_3d_8points, which the candidate copies rather
than imports (that module pulls in numba, which is not installed here), and the
sweep must actually find an edge. The second test draws one box's wireframe at a
known yaw into a blank image, adds noise, and starts the candidate 20 degrees
off; it has to come back within 3 degrees.

    .venv/bin/python scripts/orientation/test_yaw_edge_align.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "candidates"))

import yaw_edge_align as ya


def reference_corners(obj_size, yaw_lidar, center_lidar):
    """evaluators.result2kitti.get_lidar_3d_8points, transcribed."""
    center_lidar = list(map(float, center_lidar))
    lidar_r = np.matrix([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
                         [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
                         [0, 0, 1]])
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix([
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [0, 0, 0, 0, h, h, h, h]])
    return np.asarray((lidar_r * corners_3d_lidar + np.matrix(center_lidar).T).T)


def test_corners_match_reference():
    box = np.array([41.0, -5.5, -1.7, 4.4, 1.8, 1.45, 0.37])
    x, y, z, l, w, h = box[:6]
    # render_annotated passes z + h/2 and the reference subtracts h/2 again
    ref = reference_corners([l, w, h], box[6], [x, y, z + h / 2])
    got = ya.corners(box, np.array([box[6]]))[0]
    assert np.allclose(got, ref, atol=1e-9), (got, ref)


def synthetic_camera():
    """K and lidar2cam for a 16 m roadside camera looking down the x axis."""
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    base = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    pitch = math.radians(12.0)
    tilt = np.array([[1.0, 0.0, 0.0],
                     [0.0, math.cos(pitch), -math.sin(pitch)],
                     [0.0, math.sin(pitch), math.cos(pitch)]])
    R = tilt @ base
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = R
    lidar2cam[:3, 3] = -R @ np.array([0.0, 0.0, 16.0])
    return K, lidar2cam


def draw_box(box, yaw, K, lidar2cam, shape=(1080, 1920)):
    """A noisy grey image holding this box's solid silhouette at `yaw`.

    Solid, not a wireframe: a drawn line is a ridge whose Sobel magnitude is
    zero along its own centre, so a wireframe would score the true yaw at a
    minimum. A vehicle against the road is a step edge, and the silhouette
    reproduces that.
    """
    pts = ya.corners(box, np.array([yaw]))[0]
    cam = pts @ lidar2cam[:3, :3].T + lidar2cam[:3, 3]
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    img = np.full(shape, 90, np.uint8)
    cv2.fillConvexPoly(img, cv2.convexHull(np.rint(uv).astype(np.int32)), 200)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0.0, 6.0, shape), 0, 255).astype(np.uint8)


def test_recovers_yaw_from_20_deg_off():
    K, lidar2cam = synthetic_camera()
    true_yaw = 0.35
    box = np.array([38.0, -4.0, 0.0, 4.4, 1.8, 1.45, true_yaw])
    img = draw_box(box, true_yaw, K, lidar2cam)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    start = box.copy()
    start[6] = true_yaw + math.radians(20.0)
    yaw, conf = ya.solve_yaw(start, grad, K, lidar2cam)
    err = abs(math.degrees(ya.sh.fold(yaw - true_yaw)))
    assert yaw is not None
    assert err < 3.0, f"recovered {math.degrees(yaw):.2f} deg, {err:.2f} deg off"
    assert conf > 1.0, conf
    print(f"recovered within {err:.2f} deg, confidence {conf:.2f}")


def test_keeps_the_detection_sign():
    """The answer never crosses 90 degrees from where the detection put it."""
    K, lidar2cam = synthetic_camera()
    true_yaw = 0.35
    box = np.array([38.0, -4.0, 0.0, 4.4, 1.8, 1.45, true_yaw])
    img = draw_box(box, true_yaw, K, lidar2cam)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    flipped = box.copy()
    flipped[6] = true_yaw + math.pi
    yaw, _ = ya.solve_yaw(flipped, grad, K, lidar2cam)
    assert abs(math.degrees(ya.sh.wrap(yaw - flipped[6]))) <= 90.0, yaw


if __name__ == "__main__":
    test_corners_match_reference()
    test_recovers_yaw_from_20_deg_off()
    test_keeps_the_detection_sign()
    print("ok")
