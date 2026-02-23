from __future__ import annotations

import math
from typing import Any, Dict, List

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
    # yellow range broad + bright
    lower = np.array([15, 80, 130], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(mask.mean() / 255.0)
    return ratio


def _encode_preview(img_bgr: np.ndarray, max_w: int = 420) -> bytes:
    h, w = img_bgr.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
