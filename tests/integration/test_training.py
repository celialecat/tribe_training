"""End-to-end training integration test on a small synthetic in-DB dataset.

Seeds videos + cached TRIBE brain activity, then runs the real Trainer through
dataloaders, checkpointing, resume and export. No component is mocked; only the
data is synthetic and small so it runs in seconds on CPU.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from app.core.constants import NUM_TARGETS
from app.db.models.enums import VideoStatus
from app.db.models.video import Video
from app.models import MaskedMultitaskLoss, build_model
from app.training import build_dataloaders, build_optimizer, build_scheduler
from app.training.export import export_torchscript
from app.training.trainer import TrainConfig, Trainer
from app.tribe.cache import BrainCache
from app.tribe.types import BrainActivity

V = 48  # small brain geometry


def _seed_dataset(session, n_channels: int = 4, per_channel: int = 4) -> int:
    cache = BrainCache(session, model_id="facebook/tribev2")
    rng = np.random.default_rng(0)
    count = 0
    for c in range(n_channels):
        for k in range(per_channel):
            views = int(10 ** rng.uniform(3, 6))
            video = Video(
                youtube_id=f"c{c}k{k}".ljust(11, "0")[:11],
                source="youtube", status=VideoStatus.ready, channel_id=f"chan{c}",
                title=f"vid {c}-{k}", subscriber_count=int(10 ** rng.uniform(2, 6)),
                duration_seconds=float(rng.uniform(30, 600)),
                view_count=views, like_count=views // 20, comment_count=views // 100,
                upload_date=datetime.now(UTC) - timedelta(days=int(rng.uniform(35, 400))),
                width=1280, height=720, fps=30.0, video_path="x/v.mp4",
            )
            session.add(video)
            session.flush()
            t = int(rng.integers(20, 45))
            f = Path(tempfile.mkstemp(suffix=".mp4")[1])
            f.write_bytes(f"video-{c}-{k}".encode())
            cache.put(
                video,
                BrainActivity(
                    array=rng.standard_normal((t, V)).astype(np.float32),
                    tr_seconds=1.49, hemodynamic_offset_seconds=5.0,
                    model_id="facebook/tribev2",
                ),
                source_path=f,
            )
            count += 1
    session.flush()
    return count


def _small_model_cfg(n_vertices: int) -> dict:
    return {
        "brain_encoder": {
            "input_vertices": n_vertices, "vertex_proj_dim": 24,
            "temporal_channels": [24, 32], "temporal_kernel_sizes": [5, 3],
            "d_model": 32, "n_heads": 4, "n_layers": 2, "ff_dim": 64,
            "dropout": 0.1, "pool": "attention", "latent_dim": 48,
        },
        "predictor": {
            "latent_dim": 48, "metadata_dim": 12, "hidden_dims": [48, 24],
            "dropout": 0.1, "num_targets": NUM_TARGETS, "heteroscedastic": True,
        },
    }


def _data_cfg() -> dict:
    return {
        "val_fraction": 0.25, "test_fraction": 0.25, "split_by": "channel",
        "max_time_steps": 48, "pad_value": 0.0, "approximate_view_curve": True,
        "batch_size": 4, "num_workers": 0, "pin_memory": False,
    }


def _train_cfg() -> dict:
    return {
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.01, "betas": [0.9, 0.999],
        "scheduler": "cosine", "warmup_steps": 2, "min_lr": 1e-6,
    }


def test_full_training_loop_checkpoints_resumes_and_exports(db_session, tmp_path) -> None:
    n = _seed_dataset(db_session)
    assert n == 16

    bundle = build_dataloaders(
        db_session, data_cfg=_data_cfg(), tribe_model_id="facebook/tribev2", seed=1
    )
    assert bundle.sizes["train"] > 0
    assert bundle.n_vertices == V

    model = build_model(_small_model_cfg(bundle.n_vertices))
    loss_fn = MaskedMultitaskLoss(heteroscedastic=True)
    optimizer = build_optimizer(model, cfg=_train_cfg())
    scheduler = build_scheduler(
        optimizer, kind="cosine", total_steps=20, warmup_steps=2, base_lr=1e-3
    )

    run_dir = tmp_path / "run"
    config = TrainConfig(epochs=3, amp=False, tensorboard=False, checkpoint_every=1)
    trainer = Trainer(
        model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
        normalizer=bundle.normalizer, config=config, run_dir=run_dir, device="cpu",
    )

    metrics = trainer.fit(bundle.train_loader, bundle.val_loader)
    assert "train/loss" in metrics
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()

    # ---- resume ----
    model2 = build_model(_small_model_cfg(bundle.n_vertices))
    opt2 = build_optimizer(model2, cfg=_train_cfg())
    trainer2 = Trainer(
        model2, loss_fn=loss_fn, optimizer=opt2, scheduler=None,
        normalizer=bundle.normalizer,
        config=TrainConfig(epochs=5, amp=False, tensorboard=False),
        run_dir=run_dir, device="cpu",
    )
    trainer2.resume(run_dir / "checkpoints" / "last.pt")
    assert trainer2.epoch >= 2
    assert trainer2.global_step > 0

    # ---- export ----
    ts_path = export_torchscript(model.cpu(), tmp_path / "m.torchscript")
    assert ts_path is not None and ts_path.exists()
    loaded = torch.jit.load(str(ts_path))
    brain = torch.randn(2, 32, V)
    meta = torch.randn(2, 12)
    pad = torch.zeros(2, 32, dtype=torch.bool)
    mean, _log_var = loaded(brain, meta, pad)
    assert mean.shape == (2, NUM_TARGETS)


def test_resolve_resume_auto_finds_latest_sibling_checkpoint(tmp_path) -> None:
    from app.training.cli import _resolve_resume

    base = tmp_path / "outputs" / "run"
    current = base / "2026-01-02_00-00-00"
    current.mkdir(parents=True)

    assert _resolve_resume(None, current) is None
    assert _resolve_resume("auto", current) is None  # nothing to resume yet
    assert _resolve_resume("/x/last.pt", current) == Path("/x/last.pt")

    older = base / "2026-01-01_00-00-00" / "checkpoints"
    older.mkdir(parents=True)
    (older / "last.pt").write_bytes(b"ckpt")
    assert _resolve_resume("auto", current) == older / "last.pt"

    # The current run's own (future) checkpoint dir is never a resume source.
    own = current / "checkpoints"
    own.mkdir()
    (own / "last.pt").write_bytes(b"ckpt")
    assert _resolve_resume("auto", current) == older / "last.pt"


def test_early_stopping_triggers() -> None:
    from app.training.early_stopping import EarlyStopping

    es = EarlyStopping(patience=2, mode="min")
    assert es.step(1.0) is True   # first is best
    assert es.step(0.9) is True   # improvement
    assert es.step(0.95) is False
    assert es.should_stop is False
    assert es.step(0.96) is False
    assert es.should_stop is True


def test_cosine_scheduler_warmup_and_decay() -> None:
    from torch import nn

    from app.training.scheduler import build_scheduler as bs

    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = bs(opt, kind="cosine", total_steps=100, warmup_steps=10, min_lr=0.0, base_lr=1.0)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]          # warming up
    assert lrs[10] == max(lrs)      # peak at end of warmup
    assert lrs[-1] < 0.05           # decayed toward min
