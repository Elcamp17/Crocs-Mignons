from __future__ import annotations

import base64
import json
import os
import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import requests

STAT_KEYS = ["health", "stamina", "weight", "oxygen", "food", "melee"]
logger = logging.getLogger("uvicorn.error")


class AnalyzerNotConfiguredError(RuntimeError):
    pass




def _resolve_provider() -> str:
    """
    Résout le provider de vision.

    Priorité:
    1) ARK_VISION_PROVIDER si défini
    2) fallback auto sur OpenAI si OPENAI_API_KEY est présente
    """
    provider = (os.getenv('ARK_VISION_PROVIDER', '') or '').strip().lower()
    if provider:
        return provider
    if os.getenv('OPENAI_API_KEY', '').strip():
        return 'openai'
    return ''


def analyze_candidate_frames(frames: List[Dict[str, Any]], min_confidence: float = 0.72) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Analyse des frames candidates avec un provider vision.

    Providers supportés:
    - json_stub (tests UI / pipeline)
    - openai (Responses API + vision sur frames JPEG)
    """
    provider = _resolve_provider()
    logger.info(
        "[ANALYZER] analyze_candidate_frames provider=%s frames_total=%s min_confidence=%.3f",
        provider or "<none>",
        len(frames or []),
        float(min_confidence),
    )

    if not provider:
        raise AnalyzerNotConfiguredError(
            "Backend déployé ✅ mais analyse IA non configurée. Configure ARK_VISION_PROVIDER=openai + OPENAI_API_KEY pour la lecture vidéo auto."
        )

    if provider == 'json_stub':
        return _provider_json_stub()
    if provider == 'openai':
        specs, meta = _provider_openai(frames, min_confidence=min_confidence)
        return specs, meta

    raise AnalyzerNotConfiguredError(f"Provider '{provider}' non implémenté dans analyzer.py")


def _provider_json_stub() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = os.getenv('ARK_VISION_JSON_STUB', '').strip()
    if not path:
        raise AnalyzerNotConfiguredError('ARK_VISION_JSON_STUB manquant pour provider json_stub')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    specs = data.get('specimens', [])
    return specs, {'analyzer_mode': 'json_stub'}


def _provider_openai(frames: List[Dict[str, Any]], min_confidence: float = 0.72) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if not api_key:
        raise AnalyzerNotConfiguredError('OPENAI_API_KEY manquante (Render > Environment)')

    model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini').strip() or 'gpt-4.1-mini'
    api_base = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    batch_size = _int_env('ARK_VISION_BATCH_SIZE', 6, 1, 12)
    max_call_frames = _int_env('ARK_VISION_MAX_CALL_FRAMES', 18, 4, 40)
    image_detail = (os.getenv('ARK_VISION_IMAGE_DETAIL', 'low') or 'low').strip().lower()
    if image_detail not in {'low', 'high', 'auto'}:
        image_detail = 'low'
    timeout_sec = _int_env('ARK_VISION_TIMEOUT_SEC', 120, 20, 300)

    chosen = _choose_frames_for_llm(frames, max_count=max_call_frames)
    logger.info(
        "[OPENAI] config model=%s base=%s batch_size=%s max_call_frames=%s image_detail=%s timeout_sec=%s chosen=%s",
        model, api_base, batch_size, max_call_frames, image_detail, timeout_sec, len(chosen)
    )
    if not chosen:
        logger.warning("[OPENAI] aucune frame candidate exploitable")
        return [], {'analyzer_mode': 'openai', 'openai_model': model, 'reason': 'Aucune frame candidate exploitable'}

    all_items: List[Dict[str, Any]] = []
    batches = [chosen[i:i + batch_size] for i in range(0, len(chosen), batch_size)]
    for i, b in enumerate(batches, start=1):
        logger.info("[OPENAI] batch %s/%s frames=%s", i, len(batches), [int(x.get('index') or -1) for x in b])
        all_items.extend(_openai_extract_batch(
            b,
            api_key=api_key,
            api_base=api_base,
            model=model,
            image_detail=image_detail,
            timeout_sec=timeout_sec,
        ))

    specimens = _normalize_and_consensus(all_items, min_confidence=min_confidence)
    logger.info("[OPENAI] done raw_detections=%s specimens=%s", len(all_items), len(specimens))
    return specimens, {
        'analyzer_mode': 'openai',
        'openai_model': model,
        'frames_sent': len(chosen),
        'batches': len(batches),
        'raw_detections': len(all_items),
    }


def _int_env(key: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(key, str(default)))
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _choose_frames_for_llm(frames: List[Dict[str, Any]], max_count: int = 18) -> List[Dict[str, Any]]:
    """Select high-quality, diverse frames to reduce cost and duplicates."""
    candidates = [f for f in (frames or []) if (f.get('is_candidate') is True) and f.get('jpeg_bytes')]
    if not candidates:
        # fallback: some top frames even if heuristic said no candidate
        candidates = [f for f in (frames or []) if f.get('jpeg_bytes')][:max_count]

    for f in candidates:
        f['_ahash'] = _ahash_from_jpeg_bytes(f.get('jpeg_bytes'))

    # Prefer confidence, but spread in time and avoid near-duplicates
    candidates = sorted(candidates, key=lambda x: (float(x.get('confidence') or 0), float(x.get('sharpness') or 0)), reverse=True)
    picked: List[Dict[str, Any]] = []
    for fr in candidates:
        if len(picked) >= max_count:
            break
        too_close_time = any(abs(float(fr.get('time_sec') or 0) - float(p.get('time_sec') or 0)) < 0.35 for p in picked)
        same_hash = any(fr.get('_ahash') and fr.get('_ahash') == p.get('_ahash') for p in picked)
        # Accept if good and diverse enough
        if same_hash:
            continue
        if too_close_time and len(picked) < max_count // 2:
            continue
        picked.append(fr)

    # fill if too aggressive
    if len(picked) < min(6, max_count):
        for fr in candidates:
            if len(picked) >= max_count:
                break
            if fr in picked:
                continue
            if any(fr.get('_ahash') and fr.get('_ahash') == p.get('_ahash') for p in picked):
                continue
            picked.append(fr)

    return sorted(picked, key=lambda x: float(x.get('time_sec') or 0))


def _ahash_from_jpeg_bytes(b: bytes | None) -> str:
    if not b:
        return ''
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return ''
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = ''.join('1' if px > mean else '0' for px in small.flatten())
    return f"{int(bits, 2):016x}"


def _openai_extract_batch(batch: List[Dict[str, Any]], *, api_key: str, api_base: str, model: str, image_detail: str, timeout_sec: int) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    content.append({
        'type': 'input_text',
        'text': (
            "Analyse ces captures de vidéo ARK Survival Ascended (popup stats cryopod / implant). "
            "Pour chaque image, lis UNIQUEMENT si le panneau de stats d'un dino est clairement visible. "
            "Concentre-toi sur les stats JAUNES (points) et non sur les valeurs blanches. "
            "Retourne strictement un JSON valide avec la forme {\"items\":[...]} sans markdown. "
            "Pour chaque item: frame_index (int), species (string), sex ('m' ou 'f'), stage ('baby' ou 'adult' ou 'unknown'), "
            "level (int si lisible sinon null), wild {health, stamina, weight, oxygen, food, melee} (entiers 0-255), confidence (0..1). "
            "Si une image n'est pas lisible ou n'a pas de panneau complet, ignore-la. "
            "Utilise les noms de stats anglais dans l'objet wild: health, stamina, weight, oxygen, food, melee. "
            "Important: ne fabrique pas de chiffres; si une stat jaune n'est pas lisible, mets null ou ignore l'image."
        )
    })
    for fr in batch:
        idx = int(fr.get('index') or -1)
        t = float(fr.get('time_sec') or 0.0)
        content.append({'type': 'input_text', 'text': f'FRAME index={idx}, time_sec={t:.2f}'})
        b64 = base64.b64encode(fr['jpeg_bytes']).decode('ascii')
        content.append({'type': 'input_image', 'image_url': f'data:image/jpeg;base64,{b64}', 'detail': image_detail})

    payload: Dict[str, Any] = {
        'model': model,
        'input': [
            {
                'role': 'user',
                'content': content,
            }
        ],
        'max_output_tokens': 1800,
    }

    # Optional: ask for JSON if supported by the selected model; if API rejects this field, fallback without it.
    use_json_format = os.getenv('ARK_VISION_USE_JSON_SCHEMA', '0').strip() in {'1', 'true', 'yes'}
    if use_json_format:
        payload['text'] = {'format': {'type': 'json_object'}}

    url = f"{api_base}/responses"
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    logger.info(
        "[OPENAI] request start url=%s model=%s batch_frames=%s use_json_format=%s timeout=%s",
        url, model, len(batch), use_json_format, timeout_sec
    )
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    logger.info("[OPENAI] response status=%s", resp.status_code)
    if resp.status_code >= 400 and use_json_format:
        # Retry once without format if the target endpoint/model rejects the text.format field.
        payload.pop('text', None)
        logger.warning("[OPENAI] retry without text.format after status=%s", resp.status_code)
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        logger.info("[OPENAI] retry response status=%s", resp.status_code)
    try:
        resp.raise_for_status()
    except Exception:
        logger.error("[OPENAI] error body=%s", resp.text[:2000] if hasattr(resp, "text") else "<no body>")
        raise
    data = resp.json()
    text = data.get('output_text') or _extract_output_text(data)
    parsed = _parse_model_json(text)

    items = parsed.get('items') if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []

    frame_map = {int(fr.get('index') or -1): fr for fr in batch}
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        norm = _normalize_raw_item(item)
        if not norm:
            continue
        fr = frame_map.get(int(norm.get('frame_index') or -1), {})
        frame_conf = float(fr.get('confidence') or 0.0)
        # Blend vision confidence with frame quality
        model_conf = float(norm.get('confidence') or 0.0)
        coverage = sum(1 for k in STAT_KEYS if norm['wild'].get(k) is not None)
        conf = (model_conf * 0.70) + (frame_conf * 0.20) + (min(coverage, 6) / 6.0 * 0.10)
        norm['confidence'] = round(max(0.0, min(1.0, conf)), 3)
        norm['frame_time_sec'] = fr.get('time_sec')
        norm['frame_quality'] = frame_conf
        out.append(norm)
    return out


def _extract_output_text(data: Dict[str, Any]) -> str:
    # Responses API usually exposes output_text, but we keep a robust extractor.
    chunks: List[str] = []
    for out in (data.get('output') or []):
        if not isinstance(out, dict):
            continue
        for c in (out.get('content') or []):
            if not isinstance(c, dict):
                continue
            t = c.get('text')
            if isinstance(t, str) and t.strip():
                chunks.append(t)
    return '\n'.join(chunks).strip()


def _parse_model_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    txt = text.strip()
    # direct
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # first JSON object heuristic
    m = re.search(r"(\{[\s\S]*\})", txt)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _normalize_raw_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    # sex
    sx = str(item.get('sex', '')).strip().lower()
    if sx in {'m', 'male', '♂'}:
        sex = 'm'
    elif sx in {'f', 'female', '♀'}:
        sex = 'f'
    else:
        return None

    species = str(item.get('species') or '').strip() or None
    stage_raw = str(item.get('stage') or '').strip().lower()
    stage = stage_raw if stage_raw in {'baby', 'adult'} else 'unknown'

    # level
    level = item.get('level', None)
    try:
        level = int(level) if level is not None else None
        if level is not None and (level < 1 or level > 1000):
            level = None
    except Exception:
        level = None

    # stats
    src = item.get('wild') or item.get('stats') or {}
    if not isinstance(src, dict):
        src = {}
    wild: Dict[str, int | None] = {}
    present = 0
    for k in STAT_KEYS:
        v = src.get(k)
        try:
            if v is None:
                wild[k] = None
                continue
            iv = int(round(float(v)))
            if not (0 <= iv <= 255):
                wild[k] = None
                continue
            wild[k] = iv
            present += 1
        except Exception:
            wild[k] = None
    if present < 4:
        return None

    frame_index = item.get('frame_index')
    try:
        frame_index = int(frame_index) if frame_index is not None else -1
    except Exception:
        frame_index = -1

    try:
        conf = float(item.get('confidence') or 0.0)
    except Exception:
        conf = 0.0

    return {
        'frame_index': frame_index,
        'species': species,
        'sex': sex,
        'stage': stage,
        'level': level,
        'wild': wild,
        'confidence': max(0.0, min(1.0, conf)),
    }


def _normalize_and_consensus(items: List[Dict[str, Any]], min_confidence: float = 0.72) -> List[Dict[str, Any]]:
    if not items:
        return []

    # Filter obvious weak detections first
    items = [x for x in items if isinstance(x, dict) and float(x.get('confidence') or 0.0) >= max(0.35, min_confidence - 0.2)]
    if not items:
        return []

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for x in items:
        species = (x.get('species') or 'Unknown').strip()
        sex = x.get('sex')
        lvl = x.get('level')
        stage = x.get('stage') or 'unknown'
        # primary grouping by species/sex/level/stage; if level unreadable, bucket by time (rough fallback)
        if lvl is None:
            t = x.get('frame_time_sec')
            t_bucket = int(float(t or 0.0) / 1.5)
            key = ('nolvl', species, sex, stage, t_bucket)
        else:
            key = ('lvl', species, sex, stage, int(lvl))
        groups[key].append(x)

    specimens: List[Dict[str, Any]] = []
    for _, group in groups.items():
        merged = _merge_group(group)
        if not merged:
            continue
        if float(merged.get('confidence') or 0.0) < min_confidence:
            continue
        specimens.append(merged)

    # Exact stat/sex duplicates across different frame groups -> keep best confidence
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for s in specimens:
        sig = tuple(s['wild'].get(k) for k in STAT_KEYS)
        key = (s.get('species'), s.get('sex'), s.get('level'), sig)
        prev = dedup.get(key)
        if (prev is None) or (float(s.get('confidence') or 0) > float(prev.get('confidence') or 0)):
            dedup[key] = s

    out = list(dedup.values())
    out.sort(key=lambda x: (str(x.get('species') or ''), str(x.get('sex') or ''), int(x.get('level') or 0), float(x.get('confidence') or 0)), reverse=False)

    # Assign user-friendly labels M447 / F463 / M1...
    counters = {'m': 0, 'f': 0}
    for s in out:
        counters[s['sex']] += 1
        if s.get('level'):
            s['label'] = f"{'M' if s['sex']=='m' else 'F'}{s['level']}"
        else:
            s['label'] = f"{'M' if s['sex']=='m' else 'F'}{counters[s['sex']]}"

    return out


def _merge_group(group: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not group:
        return None
    # Representative metadata
    best = max(group, key=lambda x: float(x.get('confidence') or 0.0))
    species = best.get('species')
    sex = best.get('sex')
    stage = best.get('stage')
    level = best.get('level')

    # Weighted vote per stat
    wild: Dict[str, int] = {}
    stat_conf_parts: List[float] = []
    for k in STAT_KEYS:
        votes: Dict[int, float] = defaultdict(float)
        for g in group:
            v = (g.get('wild') or {}).get(k)
            if v is None:
                continue
            try:
                iv = int(v)
            except Exception:
                continue
            w = max(0.05, float(g.get('confidence') or 0.0))
            votes[iv] += w
        if not votes:
            continue
        value, weight = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0]
        total = sum(votes.values()) or 1.0
        wild[k] = int(value)
        stat_conf_parts.append(weight / total)

    if len(wild) < 4:
        return None

    base_conf = float(best.get('confidence') or 0.0)
    consensus = (sum(stat_conf_parts) / len(stat_conf_parts)) if stat_conf_parts else 0.0
    multi_bonus = min(0.12, max(0.0, (len(group) - 1) * 0.03))
    conf = max(0.0, min(1.0, base_conf * 0.55 + consensus * 0.35 + multi_bonus))

    return {
        'id': _specimen_id(species, sex, level, wild),
        'species': species,
        'sex': sex,
        'stage': stage if stage in {'baby', 'adult'} else None,
        'level': level,
        'wild': wild,
        'confidence': round(conf, 3),
    }


def _specimen_id(species: Any, sex: Any, level: Any, wild: Dict[str, int]) -> str:
    parts = [str(species or 'Unknown'), str(sex or '?'), str(level or '?')]
    for k in STAT_KEYS:
        parts.append(str(wild.get(k, '-')))
    raw = '|'.join(parts)
    return 'spec-' + re.sub(r'[^a-zA-Z0-9]+', '-', raw).strip('-').lower()[:120]
