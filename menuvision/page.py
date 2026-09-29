"""Find the printed page in a photo and flatten it."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass
class Page:
    image: np.ndarray
    quad: np.ndarray | None  # corners in the photo (tl, tr, br, bl); None when not found
    coverage: float  # page area / photo area; 0 when not found


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def find_page(photo: np.ndarray, min_coverage: float = 0.15, joined: bool = False) -> Page:
    """Largest convex quadrilateral in the photo, warped to a flat page.

    When no quadrilateral covers at least `min_coverage` of the photo, the photo is
    returned as is with `quad=None`, so the caller can ask for a better shot.

    `joined` is the second attempt for a sheet that runs out of the frame: the frame cuts
    its outline into separate pieces (the left edge, the rest), so no single piece is a
    quadrilateral. The hull of all long edge pieces together is the visible part of the
    sheet, simplified until four corners remain.
    """
    gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(cv.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv.dilate(edges, np.ones((5, 5), np.uint8))
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    area = photo.shape[0] * photo.shape[1]
    if joined:
        long = [c for c in contours if cv.arcLength(c, False) > 0.1 * min(photo.shape[:2])]
        if not long:
            return Page(photo, None, 0.0)
        hulls, tolerances = [cv.convexHull(np.vstack(long))], (0.02, 0.03, 0.04, 0.06, 0.08)
    else:
        # Hulls, not raw contours: a page edge broken by glare or a thumb leaves an open
        # contour with almost no area, while its hull is still the page outline.
        hulls = sorted((cv.convexHull(c) for c in contours), key=cv.contourArea, reverse=True)[:5]
        tolerances = (0.02,)
    for hull in hulls:
        for tolerance in tolerances:
            approx = cv.approxPolyDP(hull, tolerance * cv.arcLength(hull, True), True)
            if len(approx) <= 4:
                break
        coverage = cv.contourArea(approx) / area
        if len(approx) != 4 or not cv.isContourConvex(approx) or coverage < min_coverage:
            continue
        quad = order_corners(approx)
        w = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        h = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        target = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        flat = cv.warpPerspective(photo, cv.getPerspectiveTransform(quad, target), (w, h))
        return Page(flat, quad, float(coverage))
    return Page(photo, None, 0.0)
