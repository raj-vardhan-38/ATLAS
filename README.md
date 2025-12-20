# ATLAS – AI Taxonomic Learning & Analysis System

ATLAS is an environmental DNA (eDNA) analysis platform that combines a cinematic marketing site with an AI-powered FastAPI backend. Users can upload FASTA sequences (or paste raw data) to classify taxa, surface novel clusters, and download detailed biodiversity reports. Models are sourced from Hugging Face and the Explorer pipeline adds Doc2Vec + HDBSCAN discovery capabilities.

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Key Capabilities](#key-capabilities)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Getting Started](#getting-started)
6. [Backend Configuration](#backend-configuration)
7. [API Reference](#api-reference)
8. [Testing & Diagnostics](#testing--diagnostics)
9. [Troubleshooting](#troubleshooting)
10. [Roadmap Ideas](#roadmap-ideas)

## Repository Structure
```
ATLAS/
├── backend/                 # FastAPI service + ML pipeline
│   ├── app.py               # API surface and static hosting
│   ├── predict.py           # Classifiers + Explorer pipeline
│   ├── requirements.txt     # pip dependencies
│   ├── environment-cpu.yml  # Conda environment (CPU-only)
│   ├── static/              # Production-ready single-page frontend
│   └── templates/           # (Legacy) Jinja templates
├── frontend/                # Marketing site source (Tailwind, AOS, custom JS)
│   ├── index.html
│   ├── css/                 # styles.css overrides Tailwind defaults
│   ├── js/                  # script.js handles interactivity + analytics
│   └── assets/              # Videos, imagery, favicon
├── runtime.txt, .python-version
└── README.md                # You are here
```

## Key Capabilities
- **Multi-marker AI classification** – 16S / 18S / COI / ITS genus models trained on millions of curated sequences.
- **Explorer pipeline** – Doc2Vec vectorization + HDBSCAN clustering + BLAST interpretation to flag potential novel taxa.
- **Adaptive ingestion** – Accepts drag-and-drop uploads in the static UI or raw FASTA text via `/run_analysis`.
- **Report automation** – Generates rich visualizations and exports under `backend/reports/`.
- **Cloud-friendly deployment** – Designed for Render / Cloudflare Pages with CORS rules baked in.

## Technology Stack
| Layer        | Details |
|--------------|---------|
| Frontend     | Tailwind CSS, custom typography (Rejouice / NB-BOO), FontAwesome, AOS animations, vanilla JS.
| Backend API  | FastAPI + Uvicorn, served via `app.py`.
| ML Pipeline  | TensorFlow / Keras, scikit-learn, Biopython, Doc2Vec (Gensim), HDBSCAN, Hugging Face Hub downloads.
| Data Formats | FASTA inputs, JSON API responses, HTML report exports.

## System Architecture
1. **Acquisition** – Users drop a FASTA file or paste sequences.
2. **Classification** – `TaxonClassifier` downloads marker-specific artifacts from `aashutosh69/Atlas` on Hugging Face (cached in `backend/models/`).
3. **Explorer (optional)** – Unclassified reads flow through Doc2Vec embeddings and HDBSCAN clustering, with BLAST lookups for putative matches.
4. **Reporting** – Results are serialized into JSON for the frontend and persisted under `backend/reports/`.
5. **Delivery** – FastAPI exposes `/run_analysis`, `/healthz`, and serves the production-ready static assets.

## Getting Started
### Prerequisites
- Python 3.10 (matches `.python-version` and `runtime.txt`).
- (Optional) Conda if you prefer the curated `environment-cpu.yml`.
- Hugging Face account/token for faster, authenticated model downloads.

### Clone & environment
```powershell
# create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# install dependencies
pip install -r backend/requirements.txt
```

To reproduce the exact Conda environment instead:
```powershell
conda env create -f backend/environment-cpu.yml
conda activate atlas-cpu
```

### Run the backend API
```powershell
cd backend
set HUGGINGFACE_TOKEN=<your_token_here>   # optional but recommended
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Visit `http://localhost:8000` to load the static SPA served from `backend/static`.

### Develop the standalone frontend
The `frontend/` directory contains the raw marketing site. You can open `frontend/index.html` directly in a browser during design. When ready, copy the built assets into `backend/static/` (or point your CDN to `frontend/`).

## Backend Configuration
| Setting | Location | Notes |
|---------|----------|-------|
| Allowed origins | `backend/app.py` | Update `origins` to include deployed frontend domains.
| Hugging Face repo | `predict.py → MODEL_REPO` | Currently `aashutosh69/Atlas`.
| Model cache | `backend/models/` | Created automatically; pre-populate for offline use.
| Reports dir | `backend/reports/` | Auto-created; ensure write permissions in production.
| Explorer toggles | `predict.py` | Pipeline auto-disables if Doc2Vec/HDBSCAN/Gensim imports fail.

**Environment variables**
- `HUGGINGFACE_TOKEN` – optional token for authenticated downloads.
- Standard FastAPI/Uvicorn variables (`HOST`, `PORT`) can be set when deploying.

## API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/healthz` | Basic readiness probe.
| GET    | `/`        | Serves the static frontend (`backend/static/index.html`).
| POST   | `/run_analysis` | Accepts `multipart/form-data` (`file`) or `application/x-www-form-urlencoded` (`data`) FASTA payloads and returns JSON predictions.

Example request:
```bash
curl -X POST \
     -F "file=@sample.fasta" \
     http://localhost:8000/run_analysis
```

Successful responses include high-confidence genus predictions, explorer findings, runtime metadata, and error messages (if applicable).

## Testing & Diagnostics
- **Model smoke test**: `python backend/test_model_loading.py` ensures TensorFlow + artifact downloads work.
- **Health check**: `GET /healthz` verifies the service process.
- **GPU detection**: `predict.py → check_gpu_status()` logs whether TensorFlow can see CUDA devices.

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Model download hangs | Missing Hugging Face token or slow network | Set `HUGGINGFACE_TOKEN` and re-run; files cache under `backend/models/`.
| `ModuleNotFoundError: gensim` | Explorer pipeline dependencies absent | Install via `pip install gensim hdbscan` or use the Conda env.
| `413 Payload Too Large` | Reverse proxy upload limit | Increase NGINX/Render upload limits or reduce FASTA size (<50 MB default).
| CORS errors in browser | Frontend domain not whitelisted | Update `origins` array in `backend/app.py`.

## Roadmap Ideas
1. Containerize with a multi-stage Dockerfile (frontend build + FastAPI service).
2. Persist analysis history in a database (PostgreSQL / Supabase) for longitudinal studies.
3. Add authentication + role-based sharing for collaborative research teams.
4. Expand Explorer pipeline with explainability metrics and richer BLAST summaries.
