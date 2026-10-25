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

`runs/` and `data/` live on the workstation data disk and are excluded from Git. A
Windows copy may be used for editing or source delivery, but is not the runtime owner.
Do not copy environments, asset caches, credentials or live sessions between systems.
