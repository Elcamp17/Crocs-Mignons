from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

STAT_KEYS = ["health", "stamina", "weight", "oxygen", "food", "melee"]


def _clean_specimen(x: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(x, dict):
        return None
    sex = str(x.get("sex", "")).strip().lower()
    if sex in {"male", "m", "♂"}:
        sex = "m"
    elif sex in {"female", "f", "♀"}:
        sex = "f"
    else:
        return None
    wild = {}
    src = x.get("wild") or x.get("stats") or {}
    for k in STAT_KEYS:
        try:
            v = src.get(k)
            if v is None:
                continue
            iv = int(round(float(v)))
            if 0 <= iv <= 255:
                wild[k] = iv
        except Exception:
            pass
    if len(wild) < 3:
        return None
    label = x.get("label") or f"{('M' if sex == 'm' else 'F')}{idx + 1}"
    return {
        "id": x.get("id") or f"spec-{idx+1}",
        "label": str(label),
        "sex": sex,
        "species": x.get("species"),
        "stage": x.get("stage"),
        "level": x.get("level"),
        "wild": wild,
        "confidence": float(x.get("confidence") or 0.0),
        "validation": x.get("validation"),
        "warning": x.get("warning"),
    }


def _stats_sig(wild: Dict[str, int]) -> str:
    return "|".join(f"{k}:{wild.get(k, '-') }" for k in STAT_KEYS)


def _compute_target(specimens: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in specimens:
        for k in STAT_KEYS:
            v = s["wild"].get(k)
            if isinstance(v, int) and (k not in out or v > out[k]):
                out[k] = v
    return out


def _mask_for(wild: Dict[str, int], target: Dict[str, int]) -> int:
    m = 0
    for i, k in enumerate(STAT_KEYS):
        if k in target and wild.get(k) == target[k]:
            m |= (1 << i)
    return m


def _bitcount(n: int) -> int:
    return n.bit_count() if hasattr(int, 'bit_count') else bin(n).count('1')


def _mask_labels(mask: int) -> List[str]:
    return [STAT_KEYS[i] for i in range(len(STAT_KEYS)) if mask & (1 << i)]


def build_repro_report(specimens_in: List[Dict[str, Any]]) -> Dict[str, Any]:
    specimens = []
    for i, s in enumerate(specimens_in or []):
        cs = _clean_specimen(s, i)
        if cs:
            specimens.append(cs)

    species_counter = Counter([s.get("species") for s in specimens if s.get("species")])
    species = species_counter.most_common(1)[0][0] if species_counter else None

    if not specimens:
        return {
            "ok": True,
            "species": species,
            "detected_count": 0,
            "specimens": [],
            "target_best": {},
            "recommendations": {"best_pair_now": None, "mirror_pairs": [], "top_pairs": []},
            "meta": {"reason": "Aucun spécimen analysé (analyzer non configuré ou aucune détection fiable)."}
        }

    usable = [s for s in specimens if not (isinstance(s.get('validation'), dict) and s['validation'].get('status') == 'mismatch')]
    if not usable:
        usable = specimens

    target = _compute_target(usable)
    full_mask = (1 << len(STAT_KEYS)) - 1
    for s in specimens:
        s["mask"] = _mask_for(s["wild"], target)
        s["sig"] = _stats_sig(s["wild"])

    males = [s for s in usable if s["sex"] == "m"]
    females = [s for s in usable if s["sex"] == "f"]

    duplicates = []
    seen = {}
    for s in specimens:
        key = (s["sex"], s["sig"])
        if key in seen:
            duplicates.append({"sex": s["sex"], "a": seen[key]["label"], "b": s["label"]})
        else:
            seen[key] = s

    mirror_pairs = []
    for m in males:
        for f in females:
            if m["sig"] == f["sig"]:
                mirror_pairs.append({"male": m["id"], "male_label": m["label"], "female": f["id"], "female_label": f["label"]})

    pairs = []
    for m in males:
        for f in females:
            union = m["mask"] | f["mask"]
            inter = m["mask"] & f["mask"]
            union_count = _bitcount(union)
            inter_count = _bitcount(inter)
            missing_mask = full_mask & ~union
            pairs.append({
                "male": m["id"], "male_label": m["label"],
                "female": f["id"], "female_label": f["label"],
                "covered_top_stats": union_count,
                "coverage": f"{union_count}/{len(STAT_KEYS)}",
                "missing": _mask_labels(missing_mask),
                "male_top_stats": _mask_labels(m["mask"]),
                "female_top_stats": _mask_labels(f["mask"]),
                "reason": "Très bon couple (toutes les stats max du pool sont couvertes)" if union_count == len(STAT_KEYS) else ("Il manque: " + ", ".join(_mask_labels(missing_mask)) if _mask_labels(missing_mask) else ""),
                "score": union_count * 100 + (_bitcount(union) - inter_count) * 10 - inter_count * 2,
            })

    pairs.sort(key=lambda p: p["score"], reverse=True)
    best_pair = pairs[0] if pairs else None

    # Keep user-facing specimen list clean
    out_specs = []
    for s in specimens:
        out_specs.append({
            "id": s["id"], "label": s["label"], "sex": s["sex"], "species": s.get("species"), "stage": s.get("stage"), "level": s.get("level"),
            "wild": s["wild"], "confidence": s.get("confidence", 0.0), "validation": s.get("validation"), "warning": s.get("warning")
        })

    return {
        "ok": True,
        "species": species,
        "detected_count": len(specimens),
        "specimens": out_specs,
        "target_best": target,
        "duplicates": duplicates,
        "validation_summary": {
            "mismatch": sum(1 for s in specimens if isinstance(s.get("validation"), dict) and s["validation"].get("status") == "mismatch"),
            "inferred": sum(1 for s in specimens if isinstance(s.get("validation"), dict) and s["validation"].get("status") == "inferred"),
            "ok": sum(1 for s in specimens if isinstance(s.get("validation"), dict) and s["validation"].get("status") == "ok"),
        },
        "recommendations": {
            "best_pair_now": best_pair,
            "mirror_pairs": mirror_pairs,
            "top_pairs": pairs[:5],
        },
    }
