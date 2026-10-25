# Dataset format and generation

The data package offers a deterministic procedural RGB kinematic generator, a
manifest-validated PyTorch dataset, variable-length collation, a strict NPZ recorder
interchange contract, and a file-based LabUtopia JSON importer. The synthetic backend
is a renderer with commanded object motion and simple hidden kinematic relations. It
does not provide live contact physics, simulator acquisition, or physical recorder
connection.

```python
from pathlib import Path

from rch_pfn.config import load_config
from rch_pfn.data import EpisodeDataset, collate_episodes, generate_dataset

config = load_config("configs/default.yaml")
manifest = generate_dataset(config, Path("data/procedural"))
train = EpisodeDataset("data/procedural", "train", config)
evidence, targets = collate_episodes([train[0], train[1]])
```

`generate_dataset(config, output)` derives independent seeds from the configured
training seed, system ID, and episode ID. Repeating generation with the same config
produces the same NPZ bytes and SHA-256 hashes. Each system has shared mechanism
parameters across its episodes. System IDs are shuffled deterministically, then
assigned wholly to train, val, or test; no system can occur in more than one split.
The manifest stores the serialized config, backend, system groups, and relative
episode paths with hashes. It contains no source-machine paths.

The generator simulates the full commanded trajectory before selecting an observation
cutoff. It then retains only frames that have arrived by the query time. Captures are
ordered by exposure time and can have delayed arrivals. Each command response is its
own segment while within capacity; longer records use contiguous grouped segments.
The query anchor is the latest exposure; its delay is zero in this fixture so the
next target frame is strictly future. Evidence ends at the latest retained capture. Its actions and
observed results summarize commands and measured motion since the preceding retained
capture. `future_actions`, `future_response`, `future_image`, and success labels are
stored only under `target__*` keys; they are never fields of `EvidenceBatch`.

## NPZ namespaces

Each episode is an NPZ archive written with fixed ZIP metadata and loaded with
`numpy.load(..., allow_pickle=False)`. It uses these namespaces:

- `sensor__*`: `images`, `pixel_valid`, `frame_valid`, `camera_valid`, `timestamps`,
  `arrival_times`, `query_time`, `actions`, `robots`, `calibration`, `task`,
  `observed_results`, `result_valid`, `segment_ids`, `anchor_poses`, `metric_valid`,
  `source_ids`, and `reference_version`.
- `target__*`: `poses`, `pose_valid`, `instance_masks`, `entity_valid`, `visibility`,
  `change_times`, `change_valid`, `future_actions`, `future_response`,
  `future_response_valid`, `future_image`, `future_image_valid`, `future_success`, and
  `future_success_valid`.
- Optional target pairs: `equivalent_poses` with `equivalent_valid`, `image_points`
  with `image_points_valid`, `relation_relevance` with `relation_relevance_valid`,
  and `contact_states` with `contact_valid`. Relevance is `[N,N]` in [0,1] and needs
  a measured/task-supervised definition; contact uses integer unknown/no/yes = 0/1/2.
  The renderer does not manufacture contact or causal relevance labels. Batches may
  mix labeled and unlabeled examples; missing optional groups receive false masks.
- `meta__*`: format version, system ID, episode ID, split, and backend. No credentials
  or imported source path is stored.

Array dimensions omit the leading batch dimension used by `EvidenceBatch` and
`TrainingTargets`. Images are stored as uint8 `[T,C,3,H,W]`; the loader converts them
to float `[0,1]`. Future images use `[C,3,H,W]`. Floating image arrays supplied by a
recorder must already be in `[0,1]`. Masks and validity arrays are boolean; identifiers
and categorical time labels are integer arrays. Times are seconds, poses are meters
and rotation vectors are radians. `metric_valid=False` is supported for a recording
whose physical scale is unknown; consumers must not interpret its pose scale as
calibrated metric distance.

`calibration[..., :9]` is row-major camera intrinsics `K`, followed by the row-major
camera-to-base 4x4 transform in `calibration[..., 9:]`. The synthetic backend has one
fixed overhead camera. Its `image_points[N,2]` are direct camera projections of each
valid target pose origin at the query frame. They are not estimated from instance-mask
centroids. Other recorders may omit this optional pair when object-frame origins are
unknown; if present, `image_points_valid` must mark only true projected origins.

For the default action width of eight, action dimensions 0:3 are commanded translation,
3:6 are commanded rotation-vector increments, dimension 6 is gripper aperture, and
dimension 7 is the command family. The robot vector stores gripper pose (six values),
aperture, and linear/angular twist (six values), then zero padding if configured wider.
`future_response` has `robot_dim` values and represents the next robot-state increment;
`observed_results` has `outcome_dim` values and represents measured historical action
results. `future_success` is a float32 0/1 target, with its own validity flag.

Objects have persistent random color and shape, a variable active object count padded
to `prior.max_objects`, and opaque rendered geometry. The configured occlusion rate
creates overlapping objects; instance masks describe visible pixels only. Hidden
pairwise relation changes can couple one object's motion to commanded actor motion.
The event itself is not rendered as a label, text, filename, or event ID. A change
label is 0 for no event inside the observed window, 1 for a relation active before the
window, and 2 through `time_bins + 1` for a change in an observed time bin. Relations
outside the observed window do not receive a past-window event label.

`EpisodeDataset` validates every manifest entry and file hash before yielding the
requested split, then checks each record against the supplied config. It returns
`(EvidenceBatch, TrainingTargets)` with batch size one. `collate_episodes` pads time
and object axes; padded evidence frames and targets are marked invalid. Training code
should construct the dataset through `EpisodeDataset` so manifest validation runs.

## LabUtopia JSON import

`import_labutopia_episode` converts one native LabUtopia insertion recording into the
`labutopia_recorder_v2` backend and validates the resulting manifest before returning.
The input path must be `motion.json` or `partial-motion.json`; an adjacent
`outcome.json` is used when present. The CLI uses the same API:

```bash
rch-pfn import-labutopia \
  --motion recordings/run-001/motion.json \
  --config configs/labutopia_deployable.yaml \
  --output data/labutopia \
  --system-id system-001 --episode-id episode-001 --split train
```

The Python entry point is:

```python
from rch_pfn.data import import_labutopia_episode

summary = import_labutopia_episode(
    "recordings/run-001/motion.json",
    "data/labutopia",
    config=config,
    system_id="system-001",
    episode_id="episode-001",
    split="train",
    policy="deployable",
)
```

The importer provides complete raw archive retention for the native JSON episode. The
output stores the byte-for-byte `motion.json`, optional `outcome.json`, and every other
top-level `*.json` file beside the motion file under `archive__*` NPZ fields. The
`archive__source_manifest_json` field records each source name, array key, byte count,
and SHA-256; `archive__field_manifest_json` records the projection and policy. Raw
source bytes remain available even when a field is excluded from model evidence. The
archive scan covers JSON sidecars in the episode directory; non-JSON files and nested
directories require separate preservation by the acquisition pipeline.

The model projection is bounded by `model.max_frames` and
`model.recorder_trace_frames`. Frame selection always keeps the first and last frame
and every phase boundary, then fills the remaining capacity deterministically. A
480 Hz actual-pose trace is sampled down to the trace budget. The selected recorder
arrays are structured, shape-checked, and stored under `recorder__*`; the raw JSON
archive is not truncated by either budget. The field manifest records raw and projected
frame/trace counts and the selected frame indices.

The two policies define which simulator-only channels are valid model evidence:

| Policy | Recorder evidence | Purpose |
|---|---|---|
| `deployable` | Joint state, joint targets/applied targets, and sample timing remain valid. Body transforms, contacts, scripted phase, and the high-rate trace are masked invalid. | Train or inspect the path that excludes privileged simulator state. |
| `recorder_oracle` | The normalized body, contact, and trace channels are also valid. | Explicit oracle or ablation experiments using simulator recorder state. |

Both policies retain the raw archive. The final tube pose is a training target and is
kept separate from `EvidenceBatch`; it is not supplied as current-state evidence. The
oracle policy must not be interpreted as a deployable sensor configuration.

Current importer limitations are explicit. It writes zero or invalid placeholders for
RGB images, camera validity/calibration, observed outcomes, robot responses, instance
masks, and future images because the native recorder input does not contain those
streams. Applied joint targets populate the nine-dimensional action field, and measured
joint positions plus velocities populate the 18-dimensional robot field. It expects the current 13-body, 9-joint, two-contact-pair
layout and all body paths. A boolean success value is imported when available;
non-boolean outcome values remain invalid. No live LabUtopia or robot connection and
no physical completion or camera acquisition check is performed.

## Real recorder NPZ contract

No physical acquisition adapter is implemented here. A recorder can use
`export_recorder_npz(path, sensor, targets, config=..., system_id=...,
episode_id=..., split=...)` to validate and write one contract archive. `sensor` and
`targets` may be unbatched mappings keyed by the field names above, or batch-one
`EvidenceBatch` and `TrainingTargets` instances. A recorder archive must include all
required sensor and training-target fields, the `meta__*` scalars, the exact units and
shapes in `IMPLEMENTATION_CONTRACT.md`, and any valid camera calibration and anchor
poses. Optional symmetry alternatives and true pose-origin image projections may be
omitted as pairs.

Import the archive with:

```python
from rch_pfn.data import import_recorder_npz

manifest = import_recorder_npz(
    "recording.npz", "data/recorder", config=config,
)
```

The system, episode, and split supplied to export are preserved and checked on import.
Repeated imports append episodes only when the system remains in its original split;
duplicate identities and attempts to mix recorder and synthetic backends fail. The
importer revalidates shapes, dtypes, finite values, causal timestamps, source IDs,
validity masks, and the system-level split before writing a manifest. An external
recording with unknown scale should set its scalar `sensor__metric_valid` to false and
provide no fabricated metric calibration.
