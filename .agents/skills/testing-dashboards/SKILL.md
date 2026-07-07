---
name: testing-dashboards
description: Test the TRIBE Scientific + Creator dashboards end-to-end locally. Use when verifying dashboard UI, backend wiring, or the prediction/analysis routes on PRs touching backend/app or frontend/.
---

# Testing the TRIBE dashboards

Two dashboards served by a FastAPI backend + React/Vite frontend:
- **Scientific** (`/scientific`): Dataset Builder, Scientific workflow, Pipeline, Monitoring, Latent analysis tabs.
- **Creator** (`/creator`): YouTube URL / local file prediction.

## Run locally
```bash
# backend (:8000)
.venv/bin/uvicorn app.main:app --app-dir backend --port 8000
# frontend (:5173, proxies /api -> :8000)
cd frontend && npm run dev
```
Open `http://localhost:5173/` — it redirects to `/scientific`.

Prereqs that have bitten us:
- `psutil` must be installed in `.venv` or backend startup fails (`monitoring_service.py`). `pip install psutil` if missing.
- The local SQLite DB (`data/ysp.db`) is typically **empty** in test mode and there is **no YouTube network access**. Design tests around clean empty/error states, not live data.

## What is safely testable with an empty DB / no network
- Boot + redirect + nav between the two dashboards.
- Dataset modes + channels are backend-driven (`GET /api/datasets/modes` from `configs/dataset_modes.yaml`); add/remove channel round-trips (`POST/DELETE /api/datasets/modes/{mode}/channels`) — verify the "N configured" badge changes.
- Latent analysis method list (`GET /api/analysis/methods`): `pca`, `supervised_pca`, `pls`, `cca`, `conditional_variance` + a `conditional_variance_justification` text entry (SIR / sufficient-dimension-reduction rationale).
- Empty states: "No cached tensors", analysis `no_cached_data` error card — should be polished, not crashes or fake charts.
- Monitoring tab: real host telemetry (CPU/RAM/Disk, Connection = WebSocket/Polling).
- Creator horizon badge reads "30 DAY HORIZON" via `GET /api/prediction/horizon`.

## What CANNOT be tested locally (needs videos + cached tensors + live YouTube)
Download / Validate / Run TRIBE / Train / Full-pipeline / Predict buttons submit real jobs. Do not expect them to complete in the isolated env.

## Known gotchas / things that might be broken
- **Dataset Builder "Preview" button** (`POST /api/datasets/preview`) does live yt-dlp channel expansion with **no timeout** — it hangs when YouTube is blocked. Do NOT click it during a recorded run; flag as a UX limitation. A fix would add a request timeout + fallback.
- **FastAPI route ordering:** literal routes must be declared before parameterized ones. `/api/prediction/horizon` was once shadowed by `/{job_id}` and returned `{}`. If a literal GET route mysteriously returns empty, check declaration order in `backend/app/api/routes/`.
- The Method dropdown may list `conditional_variance_justification` as if selectable; it is the justification text entry, not a runnable method.

## Regression / lint before committing
```bash
.venv/bin/pytest -q            # backend unit + integration tests
.venv/bin/ruff check .         # lint
cd frontend && npm run build && npm run lint
```

## Devin Secrets Needed
- None for the local empty-DB UI test.
- For a full end-to-end run (out of scope for UI testing): `HF_TOKEN` (Llama-3.2-3B access for TRIBE text extractor), `YSP_YTDLP_COOKIES` (YouTube cookies for downloads), and `VULTR_API_KEY` (GPU/CPU provisioning).
