"""End-to-end pipeline orchestration (runs on the compute node).

Chains the independent stages — dataset ingestion + validation, TRIBE v2 brain
inference + caching, and model training — into a single resumable command.
Each stage is idempotent, so re-running the pipeline resumes rather than
repeats: already-ingested videos are skipped, cached brain tensors are not
recomputed, and training can resume from the latest checkpoint.
"""
