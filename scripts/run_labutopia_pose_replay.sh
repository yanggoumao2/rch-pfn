#!/usr/bin/env bash
set -euo pipefail

# This wrapper is intended to run on hp-workstation.  It keeps the formal
# output inside the RCH-PFN project while referring to the LabUtopia checkout
# only for its simulator runtime, prepared stage, and source motion.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABUTOPIA_PROJECT="${LABUTOPIA_PROJECT:-/home/yxy/data/projects/labutopia-20261021}"
LABUTOPIA_ROOT="${LABUTOPIA_ROOT:-${LABUTOPIA_PROJECT}/LabUtopia}"
RUNTIME_ROOT="${LABUTOPIA_PROJECT}/runtime"

MOTION="${MOTION:-${LABUTOPIA_PROJECT}/artifacts/nominal-contact-fix-20261023/nominal-candidate05-01/episode/motion.json}"
OUTCOME="${OUTCOME:-${LABUTOPIA_PROJECT}/artifacts/nominal-contact-fix-20261023/nominal-candidate05-01/episode/outcome.json}"
STAGE="${STAGE:-${LABUTOPIA_PROJECT}/artifacts/nominal-contact-fix-20261023/nominal-candidate05-01/episode/physics-scene.usdc}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/data/official-recorder-close-candidate05-45}"

if [[ ! -x "${RUNTIME_ROOT}/ubuntu22/bin/bash" ]]; then
    echo "LabUtopia runtime rootfs is missing: ${RUNTIME_ROOT}/ubuntu22" >&2
    exit 2
fi

# runtime_enter.sh intentionally exposes only LabUtopia as /work.  Bind the
# RCH-PFN project as /work/rch-pfn so the formal entry point remains in this
# repository while Isaac Sim still runs inside the known LabUtopia runtime.
case "${MOTION}" in "${LABUTOPIA_PROJECT}"/*) MOTION_INNER="/work/${MOTION#${LABUTOPIA_PROJECT}/}";; *) echo "MOTION must be under LABUTOPIA_PROJECT" >&2; exit 2;; esac
case "${OUTCOME}" in "${LABUTOPIA_PROJECT}"/*) OUTCOME_INNER="/work/${OUTCOME#${LABUTOPIA_PROJECT}/}";; *) echo "OUTCOME must be under LABUTOPIA_PROJECT" >&2; exit 2;; esac
case "${STAGE}" in "${LABUTOPIA_PROJECT}"/*) STAGE_INNER="/work/${STAGE#${LABUTOPIA_PROJECT}/}";; *) echo "STAGE must be under LABUTOPIA_PROJECT" >&2; exit 2;; esac
case "${OUTPUT}" in "${PROJECT_ROOT}"/*) OUTPUT_INNER="/work/rch-pfn/${OUTPUT#${PROJECT_ROOT}/}";; *) echo "OUTPUT must be under the RCH-PFN project" >&2; exit 2;; esac

exec bwrap --die-with-parent --unshare-user --uid 0 --gid 0 --unshare-pid --unshare-ipc --unshare-uts \
    --bind "${RUNTIME_ROOT}/ubuntu22" / \
    --dev-bind /dev /dev --proc /proc --ro-bind /sys /sys \
    --bind "${LABUTOPIA_PROJECT}" /work \
    --bind "${PROJECT_ROOT}" /work/rch-pfn \
    --ro-bind /usr/lib/x86_64-linux-gnu /host-driver \
    --ro-bind /usr/share/nvidia/nvoptix.bin /usr/share/nvidia/nvoptix.bin \
    --tmpfs /tmp --chdir /work/rch-pfn \
    --setenv HOME /work/runtime/home \
    --setenv XDG_CACHE_HOME /work/cache \
    --setenv UV_CACHE_DIR /work/cache/uv \
    --setenv LC_ALL C.UTF-8 --setenv NO_PROXY localhost,127.0.0.1 \
    --setenv LD_LIBRARY_PATH /work/runtime/nvidia \
    --setenv VK_ICD_FILENAMES /usr/share/vulkan/icd.d/nvidia_icd.json \
    --setenv OMNI_KIT_ACCEPT_EULA YES --setenv ACCEPT_EULA Y \
    /bin/bash -lc "exec /work/runtime/venv/bin/python /work/rch-pfn/scripts/record_labutopia_pose_replay.py \
      --labutopia-root /work/LabUtopia \
      --motion '${MOTION_INNER}' \
      --outcome '${OUTCOME_INNER}' \
      --stage '${STAGE_INNER}' \
      --output '${OUTPUT_INNER}' \
      --sample-every '${SAMPLE_EVERY:-6}' \
      --width '${WIDTH:-1280}' \
      --height '${HEIGHT:-720}' \"\$@\"" _ "$@"
