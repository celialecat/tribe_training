#!/usr/bin/env bash
# ==============================================================================
# Mount Vultr block storage at $MOUNT_POINT (idempotent).
# Formats the volume ext4 on first use only, adds an fstab entry so it survives
# reboots, and creates the persistent directory layout for the dataset, TRIBE
# cache, database, checkpoints and logs.
# ==============================================================================
set -euo pipefail
MOUNT_POINT="${MOUNT_POINT:-/mnt/ysp-data}"

echo "[mount] target: ${MOUNT_POINT}"

# Already mounted? Nothing to do.
if mountpoint -q "${MOUNT_POINT}"; then
    echo "[mount] ${MOUNT_POINT} already mounted."
else
    # Pick the first unmounted, unpartitioned block device that is not the root
    # disk (Vultr attaches block storage as an extra /dev/vdX device).
    root_disk="$(lsblk -no PKNAME "$(findmnt -no SOURCE /)" 2>/dev/null || true)"
    device=""
    for dev in $(lsblk -dn -o NAME,TYPE | awk '$2=="disk"{print $1}'); do
        [ "${dev}" = "${root_disk}" ] && continue
        if [ -z "$(lsblk -no MOUNTPOINT "/dev/${dev}" | tr -d ' \n')" ]; then
            device="/dev/${dev}"
            break
        fi
    done
    if [ -z "${device}" ]; then
        echo "[mount] ERROR: no unmounted block device found (is storage attached?)." >&2
        lsblk >&2
        exit 1
    fi
    echo "[mount] using device ${device}"

    # Format only if there is no filesystem yet (preserves data on re-runs).
    if ! blkid "${device}" >/dev/null 2>&1; then
        echo "[mount] no filesystem on ${device}; creating ext4."
        mkfs.ext4 -F "${device}"
    fi

    mkdir -p "${MOUNT_POINT}"
    uuid="$(blkid -s UUID -o value "${device}")"
    if ! grep -q "${uuid}" /etc/fstab; then
        echo "UUID=${uuid} ${MOUNT_POINT} ext4 defaults,nofail 0 2" >> /etc/fstab
    fi
    mount "${MOUNT_POINT}"
fi

# Persistent directory layout (survives instance destruction).
mkdir -p \
    "${MOUNT_POINT}/data/raw" \
    "${MOUNT_POINT}/data/processed" \
    "${MOUNT_POINT}/cache/tribe" \
    "${MOUNT_POINT}/models" \
    "${MOUNT_POINT}/outputs" \
    "${MOUNT_POINT}/runs" \
    "${MOUNT_POINT}/venv"

df -h "${MOUNT_POINT}"
echo "[mount] done."
