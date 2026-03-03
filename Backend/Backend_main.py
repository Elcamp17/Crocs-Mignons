from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import logging
import traceback
from typing import Any, Dict

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

try:
    # Exécution en package (Backend/)
    from .video_extract import extract_candidate_frames
    from .analyzer import analyze_candidate_frames, AnalyzerNotConfiguredError

    # Compat: le fichier s'appelle aujourd'hui repro.py (préféré),
    # mais on garde le support de repro_logic.py si présent.
    try:
        from .repro import build_repro_report  # type: ignore
    except ImportError:
        from .repro_logic import build_repro_report  # type: ignore

except ImportError:  # exécution directe dans /app (sans package)
    from video_extract import extract_candidate_frames
    from analyzer import analyze_candidate_frames, AnalyzerNotConfiguredError

    try:
        from repro import build_repro_report  # type: ignore
    except ImportError:
        from repro_logic import build_repro_report  # type: ignore


APP_MODE = os.getenv("APP_MODE", "dev")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "250"))

app = FastAPI(title="ARK Repro Video Analyzer", version="0.1.0")
logger = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_here = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(_here, "frontend")
if not os.path.isdir(FRONTEND_DIR):
    # repo mode: backend/ + frontend/ as siblings
    FRONTEND_DIR = os.path.normpath(os.path.join(_here, "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/")
def root_ui():
    if os.path.isdir(FRONTEND_DIR):
        return FileResponse(os.path.join(FRONTEND_DIR, "repro.html"))
    return {"ok": True, "message": "Frontend non embarqué"}


@app.get("/repro")
def repro_ui():
    if os.path.isdir(FRONTEND_DIR):
        return FileResponse(os.path.join(FRONTEND_DIR, "repro.html"))
    raise HTTPException(status_code=404, detail="Frontend non embarqué")


@app.get("/repro.html")
def repro_html_ui():
    return repro_ui()


@app.get("/checklist.html")
def checklist_ui():
    if os.path.isdir(FRONTEND_DIR):
        return FileResponse(os.path.join(FRONTEND_DIR, "checklist.html"))
    raise HTTPException(status_code=404, detail="Frontend non embarqué")


@app.get("/scan.html")
def scan_ui():
    if os.path.isdir(FRONTEND_DIR):
        return FileResponse(os.path.join(FRONTEND_DIR, "scan.html"))
    raise HTTPException(status_code=404, detail="Frontend non embarqué")


def _analyzer_mode() -> str:
    if os.getenv("ARK_VISION_PROVIDER", "").strip():
        return os.getenv("ARK_VISION_PROVIDER", "vision-plugin")
    return "frames-only (configure analyzer)"


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": _analyzer_mode(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "app_mode": APP_MODE,
    }


@app.post("/analyze-repro-video")
async def analyze_repro_video(
    file: UploadFile = File(...),
    min_confidence_percent: float = Form(72),
    frame_step_sec: float = Form(0.5),
    max_frames: int = Form(120),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    ctype = (file.content_type or "").lower()
    if "video" not in ctype and not file.filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Fichier vidéo attendu (MP4 de préférence)")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (> {MAX_UPLOAD_MB} Mo)")

    logger.info(
        "[SCAN] route hit file=%s content_type=%s size_bytes=%s provider=%s",
        file.filename,
        file.content_type,
        len(raw),
        _analyzer_mode(),
    )

    min_confidence = max(0.0, min(1.0, float(min_confidence_percent) / 100.0))
    frame_step_sec = max(0.1, min(2.0, float(frame_step_sec)))
    max_frames = max(10, min(300, int(max_frames)))

    with tempfile.TemporaryDirectory(prefix="arkrepro_") as td:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        video_path = os.path.join(td, f"upload{ext}")
        with open(video_path, "wb") as f:
            f.write(raw)

        try:
            logger.info(
                "[SCAN] extraction start path=%s frame_step_sec=%s max_frames=%s min_confidence=%.3f",
                video_path,
                frame_step_sec,
                max_frames,
                min_confidence,
            )
            frames = extract_candidate_frames(video_path, frame_step_sec=frame_step_sec, max_frames=max_frames)
            logger.info(
                "[SCAN] extraction done frames_total=%s candidates=%s",
                len(frames),
                sum(1 for fr in frames if fr.get("is_candidate", True)),
            )
        except Exception as e:
            logger.exception("Extraction vidéo impossible")
            raise HTTPException(status_code=500, detail=f"Extraction vidéo impossible: {e}") from e

        warning = None
        specimens = []
        analyzer_meta: Dict[str, Any] = {"analyzer_mode": _analyzer_mode()}

        try:
            logger.info("[SCAN] analyzer start frames_total=%s", len(frames))
            specimens, analyzer_meta = analyze_candidate_frames(frames, min_confidence=min_confidence)
            logger.info(
                "[SCAN] analyzer done specimens=%s analyzer_meta=%s",
                len(specimens or []),
                json.dumps(analyzer_meta, ensure_ascii=False),
            )
        except AnalyzerNotConfiguredError as e:
            logger.warning("Analyzer non configuré: %s", e)
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            logger.exception("Erreur analyse_candidate_frames")
            # Si c'est une exception OpenAI avec réponse HTTP, on essaie d'afficher plus de détails.
            try:
                resp = getattr(e, "response", None)
                if resp is not None:
                    logger.error("OpenAI error response: %s", getattr(resp, "text", str(resp)))
            except Exception:
                logger.exception("Erreur pendant le logging de la réponse OpenAI")
            raise HTTPException(status_code=502, detail=f"Analyse IA indisponible: {e}") from e

        report = build_repro_report(specimens)
        report.setdefault("ok", True)
        report["warning"] = warning
        report["meta"] = {
            **report.get("meta", {}),
            **analyzer_meta,
            "frames_total": len(frames),
            "frames_kept": sum(1 for fr in frames if fr.get("is_candidate", True)),
        }
        report["frames"] = [_frame_preview(fr) for fr in frames[:60]]
        return JSONResponse(report)



def _image_bytes_to_jpeg_bytes(raw: bytes) -> bytes:
    """Convertit n'importe quelle image (png/jpg/...) en JPEG bytes pour l'analyzer (vision)."""
    if not raw:
        return raw
    if cv2 is None or np is None:
        return raw
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return raw
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return raw
        return bytes(buf)
    except Exception:
        return raw


def _sharpness_from_jpeg_bytes(jpeg: bytes) -> float:
    if not jpeg or cv2 is None or np is None:
        return 0.0
    try:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        return 0.0


@app.post("/analyze-stats-image")
async def analyze_stats_image(
    file: UploadFile = File(...),
    min_confidence_percent: float = Form(72),
):
    """Analyse une image unique (screen stats) avec le même analyzer Vision que la page Repro."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier manquant")

    ctype = (file.content_type or "").lower()
    if "image" not in ctype and not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="Fichier image attendu (PNG/JPG)")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Fichier vide")

    # On limite un peu pour éviter des uploads énormes côté front (GitHub pages -> Render)
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image trop volumineuse (> 25 Mo)")

    min_confidence = max(0.0, min(1.0, float(min_confidence_percent) / 100.0))

    jpeg = _image_bytes_to_jpeg_bytes(raw)
    frame = {
        "index": 0,
        "time_sec": 0.0,
        "jpeg_bytes": jpeg,
        "is_candidate": True,
        "confidence": 1.0,
        "sharpness": _sharpness_from_jpeg_bytes(jpeg),
    }

    try:
        specimens, analyzer_meta = analyze_candidate_frames([frame], min_confidence=min_confidence)
    except AnalyzerNotConfiguredError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        logger.exception("Erreur analyse_candidate_frames (image)")
        raise HTTPException(status_code=502, detail=f"Analyse IA indisponible: {e}") from e

    best = None
    if specimens:
        best = max(specimens, key=lambda s: float(s.get("confidence") or 0.0))

    return JSONResponse({
        "ok": True,
        "specimen": best,
        "specimens": specimens,
        "meta": analyzer_meta,
    })


def _frame_preview(frame: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: frame.get(k) for k in ["index", "time_sec", "sharpness", "confidence", "is_candidate"]}
    img_bytes = frame.get("jpeg_bytes")
    if img_bytes:
        out["preview_data_url"] = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode("ascii")
    return out
