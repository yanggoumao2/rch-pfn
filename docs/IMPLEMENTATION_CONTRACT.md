# Implementation contract v1

Root implements the model, probabilistic losses, runner and public packaging. Data and
controller contributors own only their assigned paths. No hidden target is accepted
by `model.forward`; it receives an explicit `EvidenceBatch`. All times are seconds,
positions meters and rotations radians. The package is `rch_pfn` under `src/`.

## EvidenceBatch (root defines in schema.py)

Fields (all tensors): images [B,T,C,3,H,W] float [0,1]; pixel_valid [B,T,C,H,W]
bool; frame_valid [B,T] bool; camera_valid [B,T,C] bool; timestamps and arrival_times
[B,T]; query_time [B]; actions [B,T,A]; robots [B,T,R]; calibration [B,T,C,25]
(K row-major then camera-to-base 4x4); task [B,G]; observed_results [B,T,O];
result_valid [B,T,O] bool; segment_ids [B,T] long nonnegative, chronological;
anchor_poses [B,T,2,4,4] base and gripper known FK; metric_valid [B] bool;
source_ids [B,T] unique long; reference_version [B] long. Latest frame is last valid.
No GT or candidate future action is in this dataclass. `.to(device)`, `.validate(cfg)`.

TrainingTargets contains poses [B,N,6] in base-frame local chart (xyz + rotvec,
not global SE3 Haar density), pose_valid [B,N,6], instance_masks [B,T,N,H,W],
entity_valid [B,N], visibility [B,T,N], change_times [B,N,N] long (0=no_event,
1=before_window, 2..bins+1=window bins), change_valid [B,N,N], future_actions
[B,A], future_response [B,R], future_response_valid [B,R], future_image
[B,C,3,H,W], future_image_valid [B,C], future_success [B], future_success_valid [B].
N may equal slots; pad targets. Optional equivalent_poses [B,L,N,6] and
equivalent_valid [B,L] are explicit symmetry alternatives including canonical;
never duplicate identical alternatives. `.to(device)`.

## Data module public API

`generate_dataset(config: Config, output: Path) -> dict` writes manifest.json with
schema_version, generator/backend, config, episodes [{path, split, system_id,
episode_id, sha256}], system-level disjoint splits, sensor/target namespaces in NPZ.
Use `np.load(..., allow_pickle=False)`; no credentials or local absolute paths.
`EpisodeDataset(root, split, config)` yields `(EvidenceBatch, TrainingTargets)`
with leading batch dimension 1, variable causal prefix (fixed per record is fine).
`collate_episodes(samples)` concatenates padded equal-config shapes, variable T/N
support desired. Manifest validator checks file hashes, source identity, split
group disjointness. Training loader must invoke validation.

The synthetic backend is a procedural RGB kinematic renderer. Objects have persistent
random appearance, command-driven motion and hidden relation changes; RGB only encodes
observable geometry, never labels, filenames or event IDs. Targets remain separate.
Future action must not use hidden state. Include no-event, delayed arrival, occlusion
and variable objects. The first version may use one camera. Sampling is deterministic
per system. Generate trajectories before causal prefix construction; no held-out future
in E. The LabUtopia importer is a separate file adapter with recorder-only limitations:
it currently supplies no live RGB/camera acquisition, physical completion checks, or
robot connection, and it expects the fixed current body/joint layout. A real recorder
must provide those fields and timing through the typed sensor/target boundary.

## LabUtopia importer API

`import_labutopia_episode(motion_path, output_root, *, config, system_id=None,
episode_id=None, split="train", policy="deployable") -> dict` accepts a native
`motion.json` or `partial-motion.json` file and an optional adjacent `outcome.json`.
It writes a `labutopia_recorder_v2` manifest entry after validating the sensor,
target, recorder, and source identity contracts. The CLI surface is
`rch-pfn import-labutopia --motion ... --config ... --output ...`; the config's
`model.recorder_policy` supplies the default policy and `--policy` may override it.

The importer provides complete raw archive retention for the native JSON episode:
`archive__motion_json`, optional `archive__outcome_json`,
`archive__field_manifest_json`, `archive__source_manifest_json`, and hashed
`archive__source_file_<sha256>` fields preserve the source bytes and their provenance.
The archive includes every top-level JSON sidecar in the motion directory. The
retention boundary is the native JSON input; non-JSON files and nested directories
are outside this importer contract.

The bounded model projection uses `model.max_frames` for phase-aware motion-frame
selection and `model.recorder_trace_frames` for high-rate trace sampling. The raw
archive is never reduced by those budgets. The structured `recorder__*` fields are
optional structured evidence and are consumed only when `model.recorder_enabled=true`.

The recorder policy is part of the serialized configuration:

| Policy | Valid recorder evidence |
|---|---|
| `deployable` | Joint positions/velocities/targets/applied targets and sample timing; simulator body, contact, scripted phase, and trace channels are masked invalid. |
| `recorder_oracle` | The deployable channels plus normalized body transforms, contact geometry, and trace channels. |

The policy is enforced by `EvidenceBatch.validate()` as well as the importer and
manifest loader. Direct model calls and streaming appends under `deployable` reject
any simulator-only recorder validity mask that is enabled.

`recorder_oracle` is an explicit simulator-state oracle for research or ablation. It
does not represent a deployable sensor set. Targets remain separate from
`EvidenceBatch`; the importer may provide the final tube pose and a validated boolean
success target without exposing them as current evidence.

The normalized body/contact/trace coordinates use the initial robot-base frame; the
original world-coordinate values remain byte-for-byte in the raw JSON archive. The
LabUtopia configs use nine action values (applied joint targets) and 18 robot values
(measured joint positions followed by velocities). The existing 480 Hz tube-liner
contact and strict-geometry outputs are whole-episode summaries, not timestamped
per-step channels, so they remain archive-only and are never broadcast across frames.

## Controller module public API

Pure geometry helpers accept torch tensors (...,4,4): invert_transform,
relative_transform, compose_gripper_target(target_pose, desired_object_in_target,
object_in_gripper). SE3 functions rotvec_to_matrix / matrix_to_rotvec /
pose_to_matrix / matrix_to_pose. Test zero/small/near-pi and frame composition.

RepairSpec, PlanReply, gate and adapter use explicit IDs, units, time/frame versions,
identity confidence, calibrated flag, metric validity, tolerance and posterior
support. Never execute on one mixture mean. Finite deterministic injected IK/collision
protocol; no robot network connection. DryRunBackend is clearly marked; external
robot adapters use abstract backend contract. Strict proposed->accepted->executing
->completed/aborted/rejected/expired, revalidate before execution, one bounded step.
Unknown scale, uncalibrated posterior, unresolved target identity -> observe/hold.
Write tests and docs for control; no claim of actual hardware deployment.
