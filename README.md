# ARK Repro Tool — push GitHub → déploiement auto (1 service)

Tu voulais **"je pose les fichiers sur GitHub et ça marche"** ✅

Cette version est prête pour ça:
- **1 seul service Render** (backend + frontend embarqué)
- dépôt MP4 depuis `repro.html`
- extraction automatique des frames (OpenCV)
- analyse vidéo via provider IA (OpenAI configurable)
- calcul repro auto (meilleur couple, paires miroir, stats max du pool)

## Ce qui est déjà fait
- `frontend/repro.html` : UI upload MP4 + affichage résultat
- `backend/main.py` : API FastAPI + sert aussi les pages frontend (`/`, `/repro`, `/checklist.html`, `/scan.html`)
- `backend/video_extract.py` : scoring frames (netteté + jaune)
- `backend/analyzer.py` : provider `openai` + `json_stub`
- `backend/repro_logic.py` : calcul des couples repro
- `render.yaml` + `Dockerfile` : déploiement auto sur Render

## Déploiement (vraiment simple)
1. Crée un repo GitHub avec ce contenu
2. Va sur **Render** → **New Web Service** → connecte le repo
3. Render détecte `render.yaml` / Dockerfile
4. Ajoute la variable **secret** `OPENAI_API_KEY`
5. Déploie
6. Ouvre l'URL Render → tu arrives directement sur `repro.html`

👉 Tu n'as pas besoin d'un hébergement frontend séparé (Vercel/Netlify) si tu gardes cette version.

## Variables d'environnement (Render)
### Obligatoire
- `OPENAI_API_KEY` (secret)

### Déjà prévues dans `render.yaml`
- `ARK_VISION_PROVIDER=openai`
- `OPENAI_MODEL=gpt-4.1-mini` (change si tu préfères un autre modèle vision)
- `ARK_VISION_BATCH_SIZE=6`
- `ARK_VISION_MAX_CALL_FRAMES=18`
- `ARK_VISION_IMAGE_DETAIL=low`

### Optionnelles
- `OPENAI_BASE_URL` (si proxy / endpoint compatible)
- `ARK_VISION_TIMEOUT_SEC=120`
- `ARK_VISION_USE_JSON_SCHEMA=1` (tente un format JSON structuré si ton endpoint le supporte)
- `MAX_UPLOAD_MB=250`

## Mode de test sans IA (stub)
Pour tester le pipeline sans coût API:
- `ARK_VISION_PROVIDER=json_stub`
- `ARK_VISION_JSON_STUB=/app/sample_detection.json`

## URLs utiles après déploiement
- `/` ou `/repro` → page repro backend-ready
- `/health` → état backend / provider
- `/checklist.html` et `/scan.html` → copies de tes pages

## Notes pratiques
- L'analyse vidéo est **beaucoup plus fiable** que l'ancien OCR front, mais reste dépendante de la qualité du MP4 (frames nettes, panneau visible).
- Le backend filtre les frames et envoie seulement un sous-ensemble au modèle pour réduire coût/latence.
- La logique repro utilise les **stats jaunes** (points) et propose automatiquement le meilleur couple.

## Lancement local (optionnel)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```
Puis ouvre `http://localhost:8000/`

## Structure
```
frontend/   # pages HTML (repro + tes pages existantes)
backend/    # API + extraction vidéo + analyse + logique repro
render.yaml # auto-deploy Render
Dockerfile  # image backend+frontend
```
