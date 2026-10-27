# Workstation deployment

The canonical development/training copy for this project is
`/home/yxy/data/projects/rch-pfn` on the existing HP workstation (`hp-workstation`).
The workstation has an independent `.venv`; no other project's Python environment was
modified. CPU engineering validation avoids the two GPUs already occupied by other
workloads. All model execution in this implementation session took place there.

```bash
cd /home/yxy/data/projects/rch-pfn
source .venv/bin/activate
rch-pfn environment
pytest -q
```

Portable new-machine installation:

```bash
# Select an existing Python 3.11+ interpreter; install only into this project.
bash scripts/workstation_bootstrap.sh /path/to/python3.12
```

`TORCH_INDEX_URL` can select a compatible CUDA wheel index. The recorded
`requirements-workstation.txt` is the exact CPU validation environment, not a promise
that those wheels fit every GPU/driver. Check GPU occupancy and driver compatibility
before selecting an explicit training `--device cuda:N`. Automatic GPU selection is
rejected so it cannot accidentally consume another task's device.

The existing LabUtopia directory is separate and remains unchanged. To introduce real
simulation data, implement its recorder against `data.export_recorder_npz`, retain
raw scene provenance and physical completion checks, then import into a separate
dataset directory. Do not retrofit the old scripted 10° outcome as learned-model data
without verified images/timestamps and privileged-target separation.

## Official LabUtopia dual-camera capture

The formal pose-replay entry point is
`scripts/record_labutopia_pose_replay.py`. It is owned by this repository and
receives the LabUtopia checkout explicitly with `--labutopia-root`; it does not
create a second project copy. Isaac Sim must be launched with LabUtopia's runtime
Python on `hp-workstation`.

The approved close third-person camera and panda-hand wrist camera are configured
inside the recorder. The default capture is 1280x720 PathTracing, samples every
sixth source frame while retaining phase transitions, and writes the official
calibrated-RGB HDF5 plus three preview frames under `data/`.

Run the complete capture from the canonical workstation checkout:

```bash
cd /home/yxy/data/projects/rch-pfn
git status --short
bash scripts/run_labutopia_pose_replay.sh
```

Run a short timestamp/debug capture before a long run:

```bash
cd /home/yxy/data/projects/rch-pfn
OUTPUT=/home/yxy/data/projects/rch-pfn/data/debug-timestamp \
SAMPLE_EVERY=1020 WIDTH=224 HEIGHT=126 \
bash scripts/run_labutopia_pose_replay.sh
```

Inspect `capture_report.json` and require a small
`camera_to_source_timestamp_max_abs_error_s` plus zero
`camera_cross_view_timestamp_max_abs_error_s` before starting the full capture.
The recorder fails closed when the camera queue drifts beyond the requested
source timestamp; an HDF5 file from a run that fails this gate must not be
used for PFN training.

For a different output directory or source episode, override the wrapper variables
without editing the script:

```bash
OUTPUT=/home/yxy/data/projects/rch-pfn/data/official-recorder-run-001 \
MOTION=/path/to/motion.json \
OUTCOME=/path/to/outcome.json \
STAGE=/path/to/physics-scene.usdc \
bash scripts/run_labutopia_pose_replay.sh
```

The recorder deliberately stores the measured Isaac Sim camera timestamp as the
canonical sample time and preserves source motion timestamps and frame indices as
provenance. After capture, import the generated episode using both policies:

```bash
cd /home/yxy/data/projects/rch-pfn
source .venv/bin/activate
H5=data/official-recorder-close-candidate05-45/dataset/episode_0000.h5
rch-pfn import-labutopia \
  --motion /home/yxy/data/projects/labutopia-20261021/artifacts/nominal-contact-fix-20261023/nominal-candidate05-01/episode/motion.json \
  --calibrated-rgb "$H5" \
  --config configs/labutopia_deployable.yaml \
  --policy deployable \
  --output data/verify-official-recorder-deployable
rch-pfn validate-data --data data/verify-official-recorder-deployable
```

Repeat the last two commands with `configs/labutopia_oracle.yaml`, policy
`recorder_oracle`, and output `data/verify-official-recorder-oracle` for the
simulator-state ablation. Do not call a single successful import a model or task
performance result.

`runs/` and `data/` live on the workstation data disk and are excluded from Git. A
Windows copy may be used for editing or source delivery, but is not the runtime owner.
Do not copy environments, asset caches, credentials or live sessions between systems.
