#!/usr/bin/env bash
# ==============================================================================
# Install the official Meta TRIBE v2 package from source.
# Clones facebookresearch/tribev2 and installs it (editable) into the ACTIVE
# Python environment. Weights (facebook/tribev2, CC-BY-NC-4.0) download from the
# HuggingFace Hub on first use into $TRIBE_CACHE_DIR. Never mocked or vendored.
# ==============================================================================
set -euo pipefail
TRIBE_SRC="${TRIBE_SRC:-/opt/tribev2}"
TRIBE_REPO="${TRIBE_REPO:-https://github.com/facebookresearch/tribev2}"

if [ ! -d "${TRIBE_SRC}/.git" ]; then
    echo "[tribe] cloning ${TRIBE_REPO} -> ${TRIBE_SRC}"
    git clone --depth 1 "${TRIBE_REPO}" "${TRIBE_SRC}"
else
    echo "[tribe] updating existing checkout at ${TRIBE_SRC}"
    git -C "${TRIBE_SRC}" pull --ff-only || true
fi

echo "[tribe] installing (editable, with training extras) ..."
# Fall back to the base install if the [training] extra is unavailable.
pip install -e "${TRIBE_SRC}[training]" || pip install -e "${TRIBE_SRC}"

python - <<'PY'
try:
    import tribev2  # noqa: F401
    print("[tribe] import OK:", getattr(tribev2, "__version__", "unknown"))
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"[tribe] import FAILED: {exc}")
PY
echo "[tribe] done."
