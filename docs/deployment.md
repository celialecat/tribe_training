# Remote training on Vultr GPU

The Mac is a **development machine only**. All dataset storage and GPU compute
(TRIBE v2 inference + model training) run on a **Vultr GPU instance** with an
attached **persistent block-storage volume**. Nothing large ever lands on the
laptop.

```
Mac  ──git push──▶  GitHub  ──git clone──▶  Vultr GPU instance
                                              │  mount block storage (persists)
                                              ├─ download + validate videos (yt-dlp)
                                              ├─ run TRIBE v2 → cache (T×V) tensors
                                              ├─ train Brain Encoder + Success Predictor
                                              └─ checkpoints + TensorBoard → block storage
```

## What lives where

| Artefact | Location | Survives instance destroy? |
|----------|----------|----------------------------|
| Videos, thumbnails, subtitles | `<mount>/data/` | ✅ (block storage) |
| Database (SQLite) | `<mount>/ysp.db` | ✅ |
| TRIBE brain tensors | `<mount>/cache/tribe/` | ✅ |
| Checkpoints, ONNX/TorchScript | `<mount>/outputs/<run>/` | ✅ |
| TensorBoard logs | `<mount>/outputs/<run>/tb/`, `<mount>/runs/` | ✅ |
| Python venv (incl. torch, TRIBE) | `<mount>/venv/` | ✅ (no reinstall on re-launch) |
| Repository code | `/opt/ysp` (root disk) | ♻️ re-cloned by bootstrap |

`<mount>` defaults to `/mnt/ysp-data`. Because checkpoints and the venv are on
the block volume, you can destroy the (expensive) GPU instance after a run and
recreate it later without losing data or reinstalling the stack.

## One-time prerequisites (on the Mac)

1. **Push the repo to GitHub** (this is how code reaches Vultr):
   ```bash
   git remote add origin https://github.com/<you>/youtube-success-predictor.git
   git push -u origin main
   ```
2. **An SSH key** (`~/.ssh/id_ed25519[.pub]`). Generate one if needed:
   `ssh-keygen -t ed25519`.
3. **A Vultr API key** — Vultr dashboard → *Account → API → Personal Access
   Token*. Export it (never commit it):
   ```bash
   export VULTR_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. Install the launcher deps locally: `make install` (only needs `httpx`,
   already a dependency — no GPU/torch required on the Mac).

## Configure the run

Edit [`deploy/vultr.yaml`](../deploy/vultr.yaml). The API key is **only** read
from `VULTR_API_KEY`, never from this file. Discover a GPU plan first:

```bash
ysp-vultr plans        # lists GPU plans: id, vCPU, RAM, GPU, VRAM, $/mo, regions
```

Pick a plan with **≥ 24 GB VRAM** (TRIBE v2 is ~1B parameters) and set in the YAML:

```yaml
region: ewr
plan:   "<gpu-plan-id-from-above>"
repo_url: "https://github.com/<you>/youtube-success-predictor.git"
dataset_source: "<video / playlist / channel URL>"
dataset_limit: 100
train_overrides: ["training.epochs=100"]
auto_halt: true          # stop compute billing when the job finishes
auto_destroy: false      # keep the instance (block storage is always retained)
```

## Launch (the single command)

```bash
ysp-vultr launch
```

This performs the entire remote workflow:

1. upload/reuse the SSH key; create/reuse the block-storage volume;
2. create (or start) the GPU instance and wait until SSH is reachable;
3. attach the block storage;
4. `bootstrap.sh` → mount storage, install ffmpeg/python/CUDA-torch, clone the
   repo, create the venv on the volume, install the project **and the official
   TRIBE v2 package**, run DB migrations;
5. `run_pipeline.sh` → `ysp-pipeline`:
   **download → validate → dataset report → TRIBE v2 → cache tensors → train
   Brain Encoder + Success Predictor → checkpoints**;
6. **halt** the instance (or destroy it if `auto_destroy: true`).

### Partial / manual control

```bash
ysp-vultr provision     # just create/start the instance + storage
ysp-vultr bootstrap     # (re)install the stack on an existing instance
ysp-vultr run           # run the pipeline on an existing instance
ysp-vultr status        # show instance state
ysp-vultr down          # halt (or destroy per config)
```

On the instance you can also run stages directly (after `source
<mount>/ysp.env && source $YSP_VENV/bin/activate`):

```bash
ysp-pipeline --source "<url>" --limit 100 -- training.epochs=200
ysp-pipeline --only brain            # just (re)run TRIBE inference
ysp-pipeline --skip download train   # only compute brain tensors + report
ysp-build-dataset --report           # print the dataset validation report
```

## Idempotency & resume

Every stage is safe to re-run:
- already-ingested videos are skipped; corrupt ones stay `rejected`;
- cached TRIBE tensors are **never recomputed**;
- training resumes from the latest checkpoint when
  `training.resume=<mount>/outputs/<run>/checkpoints/last.pt` is passed.

## Costs & safety

- GPU instances bill per hour while **running**. `auto_halt: true` stops that
  billing as soon as the job ends; the block volume bills separately (cheap).
- `auto_destroy: true` deletes the instance entirely on completion — the block
  storage (and thus all data/checkpoints) is **retained** and re-attached on the
  next `launch`.
- `ysp-vultr down` is always available to stop an instance manually.

## Troubleshooting

- **`nvidia-smi` missing in bootstrap** — use a Vultr image that ships NVIDIA
  drivers, or install the CUDA driver before re-running `ysp-vultr bootstrap`.
- **YouTube bot-checks** — if downloads start failing from the datacenter IP,
  supply a cookies file (`yt-dlp --cookies`) via the downloader options; see
  `app/dataset/download.py`.
- **TRIBE weights gated** — set `HUGGING_FACE_HUB_TOKEN` in `<mount>/ysp.env`
  before the `brain` stage.
