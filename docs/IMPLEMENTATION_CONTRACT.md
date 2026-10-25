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

Synthetic backend explicitly a procedural RGB kinematic renderer, not a contact
physics simulator nor a LabUtopia adapter. Objects with persistent random appearance,
command-driven motion and hidden relation changes; RGB only encodes observable
geometry, never labels, filenames or event IDs. Targets separate. Future action
must not use hidden state. Include no-event, delayed arrival, occlusion and variable
objects. First version may use one camera. Sampling is deterministic per system.
Generate trajectories before causal prefix construction; no held-out future in E.
Document a real recorder import contract and provide validated import/export API.

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
