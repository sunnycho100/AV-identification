"""Road vanishing point from independently fitted line segments (GT-free).

Replaces the Hough + RANSAC estimator in road_line_calibrate_v3, which biased the
VP by ~1.3 deg on the Beltline clips. Two differences matter:

1. LSD gives sub-pixel segment endpoints; Hough quantises to the accumulator grid.
2. Each road line is fitted from its own pixels with two free parameters, then the
   VP is the least-squares common point of those lines. The old estimator scored
   candidate VPs directly, so short mis-detected segments could drag it.

Two diagnostics come back with the VP. `residual_median_px` is the real validity
gate: above MAX_RESIDUAL_PX the fitted lines share no common point and the frame is
rejected rather than returned. `ramp_corr` correlates signed residual against line
position, the pattern that exposed the original bias; treat it as advisory only,
since with ~20 lines |r| up to about 0.45 is not distinguishable from noise.

    from scripts.calibration.lsd_vanishing_point import lsd_vanishing_point
    vp, lines, diag = lsd_vanishing_point(img)
"""
import sys

import cv2
import numpy as np

MIN_SEG_LEN = 22.0        # px; below this LSD direction estimates are unreliable
MIN_LINE_LEN = 200.0      # px; total segment length needed to accept a road line
MIN_Y_SPAN = 90.0         # px; a line must extend in depth, not be a short blob
COLLINEAR_PX = 3.0        # px; perpendicular distance for grouping segments
COLLINEAR_DEG = 2.5       # deg; direction agreement for grouping segments
MAX_RESIDUAL_PX = 15.0    # px; above this the lines do not share a VP at all
INLIER_PX = 4.0           # px; line-to-VP distance counted as one family


def _tls_line(pts, w):
    """Length-weighted total least squares. Returns unit normal n and offset d."""
    mu = np.average(pts, axis=0, weights=w)
    q = (pts - mu) * np.sqrt(w)[:, None]
    n = np.linalg.svd(q, full_matrices=False)[2][-1]
    n = n / np.linalg.norm(n)
    return n, float(n @ mu)


def fit_road_lines(img, roi_top=420.0):
    """Group LSD segments into road lines, one free 2-DOF fit each.

    Returns a list of (normal, offset, total_px_length); the line is {p : n.p = d}.
    """
    det = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    found = det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0]
    if found is None:
        return []
    segs = found.reshape(-1, 4)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mid = np.c_[(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2]
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1],
                                segs[:, 2] - segs[:, 0])) % 180
    # drop near-horizontal and near-vertical: neither can be a receding road line
    keep = ((mid[:, 1] > roi_top) & (length > MIN_SEG_LEN) &
            (((ang > 8) & (ang < 82)) | ((ang > 98) & (ang < 172))))
    segs, length, ang, mid = segs[keep], length[keep], ang[keep], mid[keep]
    if len(segs) < 4:
        return []

    used = np.zeros(len(segs), bool)
    lines = []
    for i in np.argsort(-length):
        if used[i]:
            continue
        p1, p2 = segs[i, :2], segs[i, 2:]
        dv = (p2 - p1) / np.linalg.norm(p2 - p1)
        n, d = np.array([-dv[1], dv[0]]), np.array([-dv[1], dv[0]]) @ p1
        da = np.minimum(np.abs(ang - ang[i]), 180 - np.abs(ang - ang[i]))
        m = (~used) & (np.abs(mid @ n - d) < COLLINEAR_PX) & (da < COLLINEAR_DEG)
        if length[m].sum() < MIN_LINE_LEN * 0.75:
            continue
        for _ in range(3):                       # re-fit on the group, re-gather
            pts = np.r_[segs[m][:, :2], segs[m][:, 2:]]
            n, d = _tls_line(pts, np.r_[length[m], length[m]])
            m = (~used) & (np.abs(mid @ n - d) < COLLINEAR_PX) & (da < COLLINEAR_DEG)
        used |= m
        span = np.r_[segs[m][:, 1], segs[m][:, 3]]
        if length[m].sum() >= MIN_LINE_LEN and span.max() - span.min() >= MIN_Y_SPAN:
            lines.append((n, d, float(length[m].sum())))
    return lines


def lsd_vanishing_point(img, roi_top=420.0):
    """Least-squares common point of the independently fitted road lines.

    Returns (vp_xy, lines, diagnostics) or (None, [], {}) if too few lines.
    """
    lines = fit_road_lines(img, roi_top)
    if len(lines) < 4:
        return None, [], {}
    homog = np.array([np.r_[n, -d] for n, d, _ in lines])
    weight = np.array([w for _, _, w in lines])

    # A scene can hold more than one family of parallel lines: at Whitney Way the
    # cross street converges somewhere else entirely, and a plain least-squares
    # point splits the difference and fits neither. Take the family carrying the
    # most line length, then refit on it alone.
    best, rng = None, np.random.default_rng(0)
    n_iter = min(2000, max(200, len(lines) * (len(lines) - 1) // 2 * 4))
    for _ in range(n_iter):
        i, j = rng.choice(len(lines), 2, replace=False)
        p = np.cross(homog[i], homog[j])
        if abs(p[2]) < 1e-12:
            continue
        p = p[:2] / p[2]
        inl = np.abs(homog @ np.r_[p, 1.0]) < INLIER_PX
        if best is None or weight[inl].sum() > best[0]:
            best = (weight[inl].sum(), inl)
    if best is None or best[1].sum() < 4:
        return None, lines, {"n_lines": len(lines), "reject": "no consistent family"}

    inl = best[1]
    for _ in range(4):                          # refit on inliers, re-gather
        v = np.linalg.svd(homog[inl])[2][-1]
        if abs(v[2]) < 1e-12:
            return None, lines, {"n_lines": len(lines), "reject": "degenerate fit"}
        vp = v[:2] / v[2]
        inl = np.abs(homog @ np.r_[vp, 1.0]) < INLIER_PX
        if inl.sum() < 4:
            return None, lines, {"n_lines": len(lines), "reject": "family collapsed"}

    lines = [ln for ln, k in zip(lines, inl) if k]
    homog = homog[inl]
    signed = homog @ np.r_[vp, 1.0]
    pos = np.array([(d - n[1] * 700.0) / n[0] for n, d, _ in lines])   # x at y=700
    ramp = float(np.corrcoef(pos, signed)[0, 1]) if len(pos) > 2 else 0.0
    diag = {"n_lines": len(lines), "n_lines_total": int(len(inl)),
            "residual_median_px": float(np.median(np.abs(signed))),
            "residual_max_px": float(np.abs(signed).max()),
            "ramp_corr": ramp}
    if diag["residual_median_px"] > MAX_RESIDUAL_PX:
        # the fitted lines do not share a vanishing point: the frame is not a
        # clean straight-road view (PTZ re-point, dusk, heavy occlusion)
        diag["reject"] = (f"residual {diag['residual_median_px']:.1f} px "
                          f"> {MAX_RESIDUAL_PX} px")
        return None, lines, diag
    return vp, lines, diag


def _draw_family(img, vp, slopes, width=3):
    for k in slopes:
        c = vp[0] - k * vp[1]
        cv2.line(img, (int(k * 430 + c), 430), (int(k * 1070 + c), 1070),
                 (255, 255, 255), width, cv2.LINE_AA)


def _selfcheck():
    """Algebra is exact; rendered recovery is within a few px; and when two
    families of parallel lines are present the dominant one must win."""
    truth = np.array([200.0, 60.0])
    ln = []
    for k in (0.3, 0.8, 1.4, 2.2, 3.0):
        c = truth[0] - k * truth[1]
        dv = np.array([k, 1.0]) / np.hypot(k, 1.0)
        n = np.array([-dv[1], dv[0]])
        ln.append((n, float(n @ np.array([k * 700 + c, 700.0])), 500.0))
    homog = np.array([np.r_[n, -d] for n, d, _ in ln])
    v = np.linalg.svd(homog)[2][-1]
    got = v[:2] / v[2]
    assert np.allclose(got, truth, atol=1e-6), got
    assert np.abs(homog @ np.r_[got, 1.0]).max() < 1e-6

    img = np.zeros((1080, 1920, 3), np.uint8)
    _draw_family(img, truth, (0.5, 1.2, 2.0, 2.9))
    vp, lines, diag = lsd_vanishing_point(img)
    assert vp is not None and len(lines) >= 4, (vp, len(lines))
    assert np.linalg.norm(vp - truth) < 12.0, (vp, truth)
    single = np.linalg.norm(vp - truth)

    # dominant family (6 long lines) vs a decoy converging elsewhere (2 lines).
    # A plain least-squares point would land between them; RANSAC must not.
    decoy = np.array([1700.0, 40.0])
    img2 = np.zeros((1080, 1920, 3), np.uint8)
    _draw_family(img2, truth, (0.4, 0.8, 1.2, 1.7, 2.3, 2.9))
    _draw_family(img2, decoy, (-2.4, -1.6))
    vp2, lines2, diag2 = lsd_vanishing_point(img2)
    assert vp2 is not None, diag2
    assert np.linalg.norm(vp2 - truth) < np.linalg.norm(vp2 - decoy), (vp2, truth, decoy)
    assert np.linalg.norm(vp2 - truth) < 15.0, (vp2, truth)
    print(f"selfcheck ok (single family off by {single:.1f} px; with a decoy "
          f"family off by {np.linalg.norm(vp2 - truth):.1f} px, "
          f"{diag2['n_lines']}/{diag2['n_lines_total']} lines kept)")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit("run with --selfcheck, or import lsd_vanishing_point()")
