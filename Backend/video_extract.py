from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _laplacian_sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _yellow_score(img_bgr: np.ndarray) -> float:
    # Heuristic: ARK cryo stats use bright yellow text. Score proportion of yellowish bright pixels in central region.
    h, w = img_bgr.shape[:2]
    x1, x2 = int(w * 0.30), int(w * 0.78)
    y1, y2 = int(h * 0.18), int(h * 0.85)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 80, 130], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)


def _yellow_bbox(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Find a broad bbox around yellow stats text (in full frame coordinates)."""
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return None

    # Search area that usually contains the creature stat panel in ARK UI.
    sx1, sx2 = int(w * 0.18), int(w * 0.97)
    sy1, sy2 = int(h * 0.08), int(h * 0.95)
    roi = img_bgr[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([15, 70, 120], dtype=np.uint8)
    upper = np.array([42, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask (keep character strokes / clusters)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) < 40:
        return None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # Reject tiny boxes (noise / highlights)
    if (x2 - x1) < max(40, int(roi.shape[1] * 0.08)) or (y2 - y1) < max(20, int(roi.shape[0] * 0.04)):
        return None

    return sx1 + x1, sy1 + y1, sx1 + x2, sy1 + y2


def _crop_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Return a high-detail crop centered on the stat panel, with safe margins."""
    h, w = img_bgr.shape[:2]
    bbox = _yellow_bbox(img_bgr)

    if bbox is None:
        # Safe fallback: keep a large center-right UI zone (includes name/level/stats panel on most captures)
        x1, x2 = int(w * 0.22), int(w * 0.95)
        y1, y2 = int(h * 0.08), int(h * 0.93)
        crop = img_bgr[y1:y2, x1:x2]
    else:
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Expand a lot to include species/sex/level labels and the full stat rows
        pad_l = int(max(140, bw * 0.35))
        pad_r = int(max(220, bw * 0.55))
        pad_t = int(max(140, bh * 0.85))
        pad_b = int(max(120, bh * 0.45))

        X1 = max(0, x1 - pad_l)
        Y1 = max(0, y1 - pad_t)
        X2 = min(w, x2 + pad_r)
        Y2 = min(h, y2 + pad_b)
        crop = img_bgr[Y1:Y2, X1:X2]

        # If crop still too small, fallback to larger central-right zone
        ch, cw = crop.shape[:2]
        if cw < int(w * 0.38) or ch < int(h * 0.30):
            x1, x2 = int(w * 0.20), int(w * 0.96)
            y1, y2 = int(h * 0.06), int(h * 0.94)
            crop = img_bgr[y1:y2, x1:x2]

    # Upscale moderately if needed to help OCR/vision on small digits
    ch, cw = crop.shape[:2]
    target_w = 1280
    if cw > 0 and cw < target_w:
        scale = target_w / float(cw)
        # cap to avoid huge memory usage
        scale = min(scale, 2.2)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)

    return crop


def _encode_preview(img_bgr: np.ndarray, max_w: int = 1280) -> bytes:
    # Crop around the probable stats panel + upscale for readability
    img_bgr = _crop_for_ocr(img_bgr)

    h, w = img_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return bytes(buf) if ok else b''


def extract_candidate_frames(video_path: str, frame_step_sec: float = 0.5, max_frames: int = 120) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps and frame_count else 0.0
    step_frames = max(1, int(round(frame_step_sec * fps)))

    frames: List[Dict[str, Any]] = []
    idx = 0
    sampled = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step_frames != 0:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            idx += 1
            continue
        sharp = _laplacian_sharpness(frame)
        yscore = _yellow_score(frame)
        confidence = min(1.0, (yscore * 18.0) + (min(sharp, 500.0) / 1000.0))
        is_candidate = (yscore >= 0.006) and (sharp >= 12)
        frames.append({
            'index': int(idx),
            'time_sec': float(idx / fps),
            'sharpness': float(sharp),
            'yellow_score': float(yscore),
            'confidence': float(confidence),
            'is_candidate': bool(is_candidate),
            'jpeg_bytes': _encode_preview(frame),
        })
        sampled += 1
        if sampled >= max_frames:
            break
        idx += 1

    cap.release()

    # Keep candidates first, but preserve some context frames.
    frames.sort(key=lambda f: (f['is_candidate'], f['confidence'], f['sharpness']), reverse=True)
    return frames
