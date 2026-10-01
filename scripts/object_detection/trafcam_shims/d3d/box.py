"""Stand-in for the two d3d.box calls trafcam_3d's inference NMS makes.

d3d (cmpute/d3d) is a compiled CUDA package that does not build on the Mac.
Inference only needs rotated-box IoU and NMS, so these use OpenCV's exact
rotated-rectangle intersection. Boxes are (N, 5) x, y, w, h, angle in radians.

ponytail: O(n^2) Python loop, fine for tens of boxes per frame; swap in a
vectorised rotated IoU if it ever runs on dense scenes.
"""
import math

import cv2
import numpy as np
import torch


def _rect(b):
    return ((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), math.degrees(float(b[4])))


def _iou(a, b):
    inter = cv2.rotatedRectangleIntersection(_rect(a), _rect(b))[1]
    if inter is None or len(inter) < 3:
        return 0.0
    i = cv2.contourArea(cv2.convexHull(inter))
    u = float(a[2] * a[3] + b[2] * b[3]) - i
    return i / u if u > 0 else 0.0


def box2d_iou(boxes1, boxes2, method="rbox"):
    a, b = boxes1.detach().cpu().numpy(), boxes2.detach().cpu().numpy()
    out = np.array([[_iou(x, y) for y in b] for x in a], np.float32).reshape(len(a), len(b))
    return torch.from_numpy(out).to(boxes1.device)


def box2d_nms(boxes, scores, iou_method="rbox", iou_threshold=0.5):
    b = boxes.detach().cpu().numpy()
    keep = np.zeros(len(b), bool)
    for i in np.argsort(-scores.detach().cpu().numpy()):
        if not any(_iou(b[i], b[j]) > iou_threshold for j in np.flatnonzero(keep)):
            keep[i] = True
    return torch.from_numpy(keep).to(boxes.device)


if __name__ == "__main__":
    s = torch.tensor([[0, 0, 4, 2, 0.0], [0, 0, 4, 2, 0.0], [0, 0, 4, 2, math.pi / 2], [50, 50, 4, 2, 0.3]])
    iou = box2d_iou(s, s)
    assert abs(iou[0, 1] - 1) < 1e-5 and abs(iou[0, 2] - 4 / 12) < 1e-4 and iou[0, 3] == 0, iou
    keep = box2d_nms(s, torch.tensor([0.9, 0.8, 0.7, 0.6]), iou_threshold=0.4)
    assert keep.tolist() == [True, False, True, True], keep
    print("d3d shim ok")
