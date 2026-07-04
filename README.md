<h1 align="center">YouTube Success Predictor</h1>

<p align="center">
  <b>Predict the future success of a YouTube video from the brain activity it is
  predicted to evoke</b>, using Meta's official <a href="https://github.com/facebookresearch/tribev2">TRIBE v2</a> Trimodal Brain Encoder.
</p>

---

## What it does

Give the platform a YouTube URL or a local video. It downloads the video,
extracts its metadata and media, runs the **official TRIBE v2** model to predict
the whole-brain fMRI response the video would evoke, compresses that
`(time × cortical-vertices)` signal into a 512-dimensional latent with a learned
**Brain Encoder**, and feeds that latent — plus video metadata — to a multitask
network that predicts:

- **Views after 7 days** and **after 30 days** (log-scale regression)
- **Engagement** (likes + comments per view)
- **Retention** (mean fractional watch time)
- **Virality** (growth ratio)
- **Calibrated confidence intervals** for every prediction

Everything is visualised in a professional React dashboard.

```
YouTube URL / file
   └─▶ download + metadata (yt-dlp, ffmpeg, opencv)
        └─▶ TRIBE v2 inference ──▶ brain activity  (T × ~20k vertices)
             └─▶ Brain Encoder ──▶ 512-d latent
                  └─▶ Multitask head ──▶ views / engagement / retention / virality (+ CI)
                       └─▶ FastAPI ──▶ React dashboard
```

## The pipeline maps directly onto the real TRIBE API

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
events = model.get_events_dataframe(video_path="video.mp4")
preds, segments = model.predict(events=events)   # preds: (n_timesteps, n_vertices)
```

TRIBE v2 is used **as-is** — never mocked or reimplemented. Its weights are
CC-BY-NC-4.0 (see [`NOTICE`](./NOTICE)); this project is a non-commercial
research tool.

## Quickstart

```bash
# 1. Backend (CPU dev environment)
python3.12 -m venv .venv && source .venv/bin/activate
make install                 # backend + dev/explain/train extras

# 2. GPU stack + official TRIBE v2 (on a CUDA host, e.g. Vultr)
make install-gpu
make install-tribe

# 3. Database
make db-upgrade

# 4. Run
make api                     # FastAPI at http://localhost:8000/docs
make frontend                # dashboard at http://localhost:5173

# …or the whole stack:
docker compose up --build
```

## Documentation

| Guide | |
|-------|--|
| [Architecture](docs/architecture.md) | System + model design and rationale |
| [Installation](docs/installation.md) | Local, GPU and TRIBE setup |
| [Training](docs/training.md) | Building datasets and training models |
| [Inference](docs/inference.md) | Running predictions |
| [Deployment (Vultr GPU)](docs/deployment.md) | Production deployment |
| [API reference](docs/api.md) | REST endpoints |
| [Developer guide](docs/developer.md) | Contributing, testing, layout |

## Status

Built incrementally; every commit leaves the repository in a runnable state.
See the top of each `docs/` page and the repository history for progress.

## License

Project source: Apache-2.0. Third-party model weights retain their own licenses
— notably TRIBE v2 (CC-BY-NC-4.0). See [`NOTICE`](./NOTICE).
