"""Safe, versioned NPZ records shared by generators and recorder imports."""

from __future__ import annotations

import re
import zipfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import Config

FORMAT_VERSION = 2
LEGACY_FORMAT_VERSIONS = (1,)
SENSOR_FIELDS = (
    "images",
    "pixel_valid",
    "frame_valid",
    "camera_valid",
    "timestamps",
    "arrival_times",
    "query_time",
    "actions",
    "robots",
    "calibration",
    "task",
    "observed_results",
    "result_valid",
    "segment_ids",
    "anchor_poses",
    "metric_valid",
    "source_ids",
    "reference_version",
)
TARGET_FIELDS = (
    "poses",
    "pose_valid",
    "instance_masks",
    "entity_valid",
    "visibility",
    "change_times",
    "change_valid",
    "future_actions",
    "future_response",
    "future_response_valid",
    "future_image",
    "future_image_valid",
    "future_success",
    "future_success_valid",
)
RECORDER_FIELDS = (
    "body_positions",
    "body_orientations_wxyz",
    "body_valid",
    "joint_positions",
    "joint_velocities",
    "joint_targets",
    "applied_targets",
    "joint_valid",
    "contact_force_base",
    "contact_normal_force",
    "contact_count",
    "contact_valid",
    "contact_points_base",
    "contact_normals_base",
    "contact_separations",
    "contact_point_valid",
    "phase_ids",
    "phase_valid",
    "sample_time",
    "sample_valid",
    "trace_times",
    "trace_body_positions",
    "trace_body_orientations_wxyz",
    "trace_valid",
)
ARCHIVE_FIELDS = ("motion_json", "outcome_json", "field_manifest_json", "source_manifest_json")
OPTIONAL_TARGET_FIELDS = (
    "equivalent_poses",
    "equivalent_valid",
    "image_points",
    "image_points_valid",
    "relation_relevance",
    "relation_relevance_valid",
    "contact_states",
    "contact_valid",
)
OPTIONAL_TARGET_PAIRS = (
    ("equivalent_poses", "equivalent_valid"),
    ("image_points", "image_points_valid"),
    ("relation_relevance", "relation_relevance_valid"),
    ("contact_states", "contact_valid"),
)
SPLITS = ("train", "val", "test")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or ".." in value:
        raise ValueError(f"{field} must be a short path-safe identifier")
    return value


def stable_seed(*parts: object) -> int:
    raw = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def stable_source_id(system_id: str, episode_id: str, frame_index: int) -> int:
    return stable_seed("source", system_id, episode_id, frame_index)


def canonical_npz_bytes(arrays: Mapping[str, Any]) -> bytes:
    """Serialize arrays without pickle and with fixed ZIP metadata for stable hashes."""
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            if not isinstance(name, str) or not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"object arrays are forbidden: {name}")
            # np.ascontiguousarray promotes a scalar from shape () to (1,),
            # which changes the record contract for scalar metadata and labels.
            array = (
                np.ascontiguousarray(array) if array.ndim else np.array(array, copy=True, order="C")
            )
            payload = BytesIO()
            np.lib.format.write_array(payload, array, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6
            )
    return stream.getvalue()


def write_npz(path: str | Path, arrays: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = canonical_npz_bytes(arrays)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(contents)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(contents).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_record_arrays(
    sensor: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    system_id: str,
    episode_id: str,
    split: str,
    backend: str,
    recorder: Mapping[str, Any] | None = None,
    archives: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    safe_identifier(system_id, "system_id")
    safe_identifier(episode_id, "episode_id")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if set(sensor) != set(SENSOR_FIELDS):
        missing, extra = set(SENSOR_FIELDS) - set(sensor), set(sensor) - set(SENSOR_FIELDS)
        raise ValueError(
            f"sensor fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    required_targets = set(TARGET_FIELDS)
    allowed_targets = required_targets | set(OPTIONAL_TARGET_FIELDS)
    if not required_targets <= set(targets) or set(targets) - allowed_targets:
        missing = required_targets - set(targets)
        extra = set(targets) - allowed_targets
        raise ValueError(
            f"target fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for left, right in OPTIONAL_TARGET_PAIRS:
        if (left in targets) != (right in targets):
            raise ValueError(f"{left} and {right} must be supplied together")
    arrays = {f"sensor__{name}": np.asarray(sensor[name]) for name in SENSOR_FIELDS}
    arrays.update({f"target__{name}": np.asarray(targets[name]) for name in targets})
    if recorder is not None:
        unknown = set(recorder) - set(RECORDER_FIELDS)
        if unknown:
            raise ValueError(f"unknown recorder fields: {sorted(unknown)}")
        arrays.update({f"recorder__{name}": np.asarray(recorder[name]) for name in recorder})
    if archives is not None:
        unknown = {
            name
            for name in archives
            if name not in ARCHIVE_FIELDS
            and not re.fullmatch(r"source_file_[0-9a-f]{64}", name)
        }
        if unknown:
            raise ValueError(f"unknown archive fields: {sorted(unknown)}")
        arrays.update({f"archive__{name}": np.asarray(archives[name]) for name in archives})
    arrays.update(
        {
            "meta__format_version": np.asarray(FORMAT_VERSION, dtype=np.int64),
            "meta__system_id": np.asarray(system_id),
            "meta__episode_id": np.asarray(episode_id),
            "meta__split": np.asarray(split),
            "meta__backend": np.asarray(backend),
        }
    )
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {name}")
    return arrays


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot safely read NPZ record {path}: {exc}") from exc


def split_record_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    expected_meta = {
        "meta__format_version",
        "meta__system_id",
        "meta__episode_id",
        "meta__split",
        "meta__backend",
    }
    actual_meta = {key for key in arrays if key.startswith("meta__")}
    if actual_meta != expected_meta:
        raise ValueError(f"invalid record metadata fields: {sorted(actual_meta)}")
    meta: dict[str, str] = {}
    version = np.asarray(arrays["meta__format_version"])
    if version.shape != () or int(version) not in (*LEGACY_FORMAT_VERSIONS, FORMAT_VERSION):
        raise ValueError(f"unsupported record format version: {version}")
    if int(version) == 1 and any(
        key.startswith(("recorder__", "archive__")) for key in arrays
    ):
        raise ValueError("record format version 1 cannot contain recorder or archive namespaces")
    for name in ("system_id", "episode_id", "split", "backend"):
        item = np.asarray(arrays[f"meta__{name}"])
        if item.shape != () or item.dtype.kind not in "US":
            raise ValueError(f"meta__{name} must be a scalar string")
        meta[name] = str(item.item())
    safe_identifier(meta["system_id"], "system_id")
    safe_identifier(meta["episode_id"], "episode_id")
    if meta["split"] not in SPLITS:
        raise ValueError("invalid record split")
    sensor: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key.startswith("sensor__"):
            sensor[key.removeprefix("sensor__")] = value
        elif key.startswith("target__"):
            targets[key.removeprefix("target__")] = value
        elif key.startswith("recorder__") or key.startswith("archive__"):
            continue
        elif not key.startswith("meta__"):
            raise ValueError(f"unrecognized NPZ namespace: {key}")
    if set(sensor) != set(SENSOR_FIELDS):
        raise ValueError("record sensor namespace is incomplete or contains unknown fields")
    if not set(TARGET_FIELDS) <= set(targets) or set(targets) - (
        set(TARGET_FIELDS) | set(OPTIONAL_TARGET_FIELDS)
    ):
        raise ValueError("record target namespace is incomplete or contains unknown fields")
    for left, right in OPTIONAL_TARGET_PAIRS:
        if (left in targets) != (right in targets):
            raise ValueError(f"{left} and {right} must appear together")
    return sensor, targets, meta


def split_recorder_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return normalized recorder arrays and archived source JSON from a record."""
    recorder: dict[str, np.ndarray] = {}
    archives: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key.startswith("recorder__"):
            recorder[key.removeprefix("recorder__")] = value
        elif key.startswith("archive__"):
            archives[key.removeprefix("archive__")] = value
    unknown = set(recorder) - set(RECORDER_FIELDS)
    if unknown:
        raise ValueError(f"unrecognized recorder fields: {sorted(unknown)}")
    unknown = {
        name
        for name in archives
        if name not in ARCHIVE_FIELDS
        and not re.fullmatch(r"source_file_[0-9a-f]{64}", name)
    }
    if unknown:
        raise ValueError(f"unrecognized archive fields: {sorted(unknown)}")
    return recorder, archives


def validate_recorder_arrays(recorder: Mapping[str, np.ndarray], config: Config) -> None:
    """Validate the normalized LabUtopia recorder view without inspecting raw archives."""
    if not recorder:
        return
    m = config.model
    required = set(RECORDER_FIELDS[:20])
    if not required <= set(recorder):
        raise ValueError(f"recorder namespace is incomplete: missing {sorted(required - set(recorder))}")
    first = np.asarray(recorder["sample_time"])
    if first.ndim != 1:
        raise ValueError("recorder.sample_time must have shape [T]")
    t = first.shape[0]
    shapes = {
        "body_positions": (t, m.recorder_body_count, 3),
        "body_orientations_wxyz": (t, m.recorder_body_count, 4),
        "body_valid": (t, m.recorder_body_count),
        "joint_positions": (t, m.recorder_joint_count),
        "joint_velocities": (t, m.recorder_joint_count),
        "joint_targets": (t, m.recorder_joint_count),
        "applied_targets": (t, m.recorder_joint_count),
        "joint_valid": (t, m.recorder_joint_count),
        "contact_force_base": (t, m.recorder_contact_pairs, 3),
        "contact_normal_force": (t, m.recorder_contact_pairs),
        "contact_count": (t, m.recorder_contact_pairs),
        "contact_valid": (t, m.recorder_contact_pairs),
        "contact_points_base": (t, m.recorder_contact_pairs, m.recorder_contact_points, 3),
        "contact_normals_base": (t, m.recorder_contact_pairs, m.recorder_contact_points, 3),
        "contact_separations": (t, m.recorder_contact_pairs, m.recorder_contact_points),
        "contact_point_valid": (t, m.recorder_contact_pairs, m.recorder_contact_points),
        "phase_ids": (t,),
        "phase_valid": (t,),
        "sample_time": (t,),
        "sample_valid": (t,),
    }
    for name, expected in shapes.items():
        _shape(f"recorder.{name}", np.asarray(recorder[name]), expected)
    for name in ("body_valid", "joint_valid", "contact_valid", "contact_point_valid", "phase_valid", "sample_valid"):
        if np.asarray(recorder[name]).dtype.kind != "b":
            raise ValueError(f"recorder.{name} must be boolean")
    for name in ("phase_ids", "contact_count"):
        if np.asarray(recorder[name]).dtype.kind not in "iu":
            raise ValueError(f"recorder.{name} must be integer-valued")
    for name, value in recorder.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"recorder.{name} contains an object array")
        if name not in {
            "body_valid",
            "joint_valid",
            "contact_valid",
            "contact_point_valid",
            "phase_valid",
            "sample_valid",
            "trace_valid",
            "phase_ids",
            "contact_count",
        } and array.dtype.kind not in "fiu":
            raise ValueError(f"recorder.{name} must contain numeric values")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"recorder.{name} contains nonfinite values")
    contact_count = np.asarray(recorder["contact_count"], dtype=np.int64)
    contact_valid = np.asarray(recorder["contact_valid"], dtype=bool)
    point_valid = np.asarray(recorder["contact_point_valid"], dtype=bool)
    if np.any(contact_count < 0) or np.any(contact_count > m.recorder_contact_points):
        raise ValueError("recorder contact count exceeds the configured point capacity")
    if np.any(point_valid & ~contact_valid[..., None]):
        raise ValueError("contact points cannot be valid when their contact pair is missing")
    measured_point_count = point_valid.sum(-1)
    if np.any(contact_valid & (measured_point_count != contact_count)):
        raise ValueError("valid contact point count disagrees with contact_count")
    phase_ids = np.asarray(recorder["phase_ids"], dtype=np.int64)
    if np.any((phase_ids < 0) | (phase_ids >= 64)):
        raise ValueError("phase IDs must be in [0, 64)")
    sample_time = np.asarray(recorder["sample_time"], dtype=np.float64)
    sample_valid = np.asarray(recorder["sample_valid"], dtype=bool)
    if np.any(np.diff(sample_time[sample_valid]) < 0):
        raise ValueError("recorder samples must be chronological")
    trace = {name for name in RECORDER_FIELDS[20:] if name in recorder}
    if trace and trace != set(RECORDER_FIELDS[20:]):
        raise ValueError("recorder trace fields must be supplied as a complete group")
    if trace:
        h = np.asarray(recorder["trace_times"]).shape[0]
        if m.recorder_trace_frames and h > m.recorder_trace_frames:
            raise ValueError("recorder trace exceeds configured capacity")
        _shape("recorder.trace_times", np.asarray(recorder["trace_times"]), (h,))
        _shape("recorder.trace_body_positions", np.asarray(recorder["trace_body_positions"]), (h, 2, 3))
        _shape("recorder.trace_body_orientations_wxyz", np.asarray(recorder["trace_body_orientations_wxyz"]), (h, 2, 4))
        _shape("recorder.trace_valid", np.asarray(recorder["trace_valid"]), (h,))
        if np.asarray(recorder["trace_valid"]).dtype.kind != "b":
            raise ValueError("recorder.trace_valid must be boolean")
        trace_times = np.asarray(recorder["trace_times"], dtype=np.float64)
        trace_valid = np.asarray(recorder["trace_valid"], dtype=bool)
        if np.any(np.diff(trace_times[trace_valid]) < 0):
            raise ValueError("recorder trace must be chronological")


def _shape(name: str, array: np.ndarray, expected: tuple[int | None, ...]) -> None:
    if array.ndim != len(expected) or any(
        size is not None and actual != size for actual, size in zip(array.shape, expected)
    ):
        raise ValueError(f"{name}: expected shape {expected}, got {array.shape}")


def _kind(name: str, array: np.ndarray, kinds: str) -> None:
    if array.dtype.kind not in kinds or array.dtype.hasobject:
        raise ValueError(f"{name} must have a non-object dtype in {kinds!r}, got {array.dtype}")


def validate_episode_arrays(
    sensor: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    config: Config,
    recorder: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Validate shapes, units' numeric safety, causal timestamps, and supervision masks."""
    config.validate()
    m = config.model
    if set(sensor) != set(SENSOR_FIELDS):
        raise ValueError("sensor namespace does not match EvidenceBatch contract")
    if not set(TARGET_FIELDS) <= set(targets) or set(targets) - (
        set(TARGET_FIELDS) | set(OPTIONAL_TARGET_FIELDS)
    ):
        raise ValueError("target namespace does not match TrainingTargets contract")
    for left, right in OPTIONAL_TARGET_PAIRS:
        if (left in targets) != (right in targets):
            raise ValueError(f"{left} and {right} must be supplied together")

    images = np.asarray(sensor["images"])
    _shape("images", images, (None, None, 3, m.image_size, m.image_size))
    t, c = images.shape[:2]
    if not (1 <= t <= m.max_frames and 1 <= c <= m.max_cameras):
        raise ValueError("episode exceeds configured frame or camera capacity")
    n = np.asarray(targets["poses"]).shape[0] if np.asarray(targets["poses"]).ndim else 0
    if not 1 <= n <= m.slots:
        raise ValueError("target object count must be in [1, model.slots]")
    h = w = m.image_size
    sensor_shapes = {
        "pixel_valid": (t, c, h, w),
        "frame_valid": (t,),
        "camera_valid": (t, c),
        "timestamps": (t,),
        "arrival_times": (t,),
        "query_time": (),
        "actions": (t, m.action_dim),
        "robots": (t, m.robot_dim),
        "calibration": (t, c, 25),
        "task": (m.task_dim,),
        "observed_results": (t, m.outcome_dim),
        "result_valid": (t, m.outcome_dim),
        "segment_ids": (t,),
        "anchor_poses": (t, 2, 4, 4),
        "metric_valid": (),
        "source_ids": (t,),
        "reference_version": (),
    }
    target_shapes = {
        "poses": (n, 6),
        "pose_valid": (n, 6),
        "instance_masks": (t, n, h, w),
        "entity_valid": (n,),
        "visibility": (t, n),
        "change_times": (n, n),
        "change_valid": (n, n),
        "future_actions": (m.action_dim,),
        "future_response": (m.robot_dim,),
        "future_response_valid": (m.robot_dim,),
        "future_image": (c, 3, h, w),
        "future_image_valid": (c,),
        "future_success": (),
        "future_success_valid": (),
    }
    for name, expected in sensor_shapes.items():
        _shape(f"sensor.{name}", np.asarray(sensor[name]), expected)
    for name, expected in target_shapes.items():
        _shape(f"target.{name}", np.asarray(targets[name]), expected)
    if "equivalent_poses" in targets:
        equivalents = np.asarray(targets["equivalent_poses"])
        if equivalents.ndim != 3 or equivalents.shape[1:] != (n, 6) or equivalents.shape[0] < 1:
            raise ValueError("equivalent_poses must have shape [L,N,6]")
        _shape("equivalent_valid", np.asarray(targets["equivalent_valid"]), (equivalents.shape[0],))
    if "image_points" in targets:
        _shape("image_points", np.asarray(targets["image_points"]), (n, 2))
        _shape("image_points_valid", np.asarray(targets["image_points_valid"]), (n,))
    for name in (
        "relation_relevance",
        "relation_relevance_valid",
        "contact_states",
        "contact_valid",
    ):
        if name in targets:
            _shape(name, np.asarray(targets[name]), (n, n))

    bool_sensor = ("pixel_valid", "frame_valid", "camera_valid", "result_valid", "metric_valid")
    int_sensor = ("segment_ids", "source_ids", "reference_version")
    bool_targets = (
        "pose_valid",
        "instance_masks",
        "entity_valid",
        "change_valid",
        "future_response_valid",
        "future_image_valid",
        "future_success_valid",
    )
    int_targets = ("change_times",)
    for name in bool_sensor:
        if np.asarray(sensor[name]).dtype.kind != "b":
            raise ValueError(f"sensor.{name} must be boolean")
    for name in int_sensor:
        if np.asarray(sensor[name]).dtype.kind not in "iu":
            raise ValueError(f"sensor.{name} must be an integer array")
    for name in bool_targets:
        if np.asarray(targets[name]).dtype.kind != "b":
            raise ValueError(f"target.{name} must be boolean")
    for name in int_targets:
        if np.asarray(targets[name]).dtype.kind not in "iu":
            raise ValueError(f"target.{name} must be an integer array")
    for name in ("equivalent_valid", "image_points_valid"):
        if name in targets and np.asarray(targets[name]).dtype.kind != "b":
            raise ValueError(f"{name} must be boolean")
    if "contact_valid" in targets and np.asarray(targets["contact_valid"]).dtype.kind != "b":
        raise ValueError("contact_valid must be boolean")
    if (
        "relation_relevance_valid" in targets
        and np.asarray(targets["relation_relevance_valid"]).dtype.kind != "b"
    ):
        raise ValueError("relation_relevance_valid must be boolean")
    if images.dtype.kind not in "uif" or images.dtype.hasobject:
        raise ValueError("images must be integer RGB or normalized floating point")
    if images.dtype.kind in "ui" and images.size and (images.min() < 0 or images.max() > 255):
        raise ValueError("integer RGB images must be in [0,255]")
    for name, value in [*sensor.items(), *targets.items()]:
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are forbidden: {name}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains nonfinite values")
    for name in ("images",):
        array = np.asarray(sensor[name])
        if array.dtype.kind == "f" and (array.size and (array.min() < 0 or array.max() > 1)):
            raise ValueError("floating RGB images must be in [0,1]")
    future_image = np.asarray(targets["future_image"])
    if future_image.dtype.kind not in "uif" or (
        future_image.dtype.kind == "f"
        and future_image.size
        and (future_image.min() < 0 or future_image.max() > 1)
    ):
        raise ValueError("future_image must be integer RGB or normalized floating point")
    if (
        future_image.dtype.kind in "ui"
        and future_image.size
        and (future_image.min() < 0 or future_image.max() > 255)
    ):
        raise ValueError("integer future_image must be in [0,255]")
    for name in (
        "timestamps",
        "arrival_times",
        "query_time",
        "actions",
        "robots",
        "calibration",
        "task",
        "observed_results",
        "anchor_poses",
    ):
        if np.asarray(sensor[name]).dtype.kind not in "fiu":
            raise ValueError(f"sensor.{name} must be numeric")
    for name in ("poses", "visibility", "future_actions", "future_response"):
        if np.asarray(targets[name]).dtype.kind not in "fiu":
            raise ValueError(f"target.{name} must be numeric")
    if (
        "equivalent_poses" in targets
        and np.asarray(targets["equivalent_poses"]).dtype.kind not in "fiu"
    ):
        raise ValueError("equivalent_poses must be numeric")
    if "image_points" in targets and np.asarray(targets["image_points"]).dtype.kind not in "fiu":
        raise ValueError("image_points must be numeric")
    if (
        "relation_relevance" in targets
        and np.asarray(targets["relation_relevance"]).dtype.kind != "f"
    ):
        raise ValueError("relation_relevance must be floating point")
    if "contact_states" in targets and np.asarray(targets["contact_states"]).dtype.kind not in "iu":
        raise ValueError("contact_states must be integer-valued")
    future_success = np.asarray(targets["future_success"])
    if future_success.dtype != np.dtype(np.float32) or not 0.0 <= float(future_success) <= 1.0:
        raise ValueError("future_success must be a float32 value in [0,1]")

    frame_valid = np.asarray(sensor["frame_valid"], dtype=bool)
    camera_valid = np.asarray(sensor["camera_valid"], dtype=bool)
    pixel_valid = np.asarray(sensor["pixel_valid"], dtype=bool)
    if recorder:
        validate_recorder_arrays(recorder, config)
    if not frame_valid.any():
        raise ValueError("each episode requires at least one valid frame")
    recorder_sample_valid = (
        np.asarray(recorder["sample_valid"], dtype=bool).any() if recorder else False
    )
    if not (camera_valid & pixel_valid.reshape(t, c, -1).any(-1))[frame_valid].any() and not recorder_sample_valid:
        raise ValueError("each episode requires a valid visual observation")
    if recorder and not recorder_sample_valid:
        raise ValueError("recorder contains no valid sample")
    timestamps = np.asarray(sensor["timestamps"], dtype=np.float64)
    arrivals = np.asarray(sensor["arrival_times"], dtype=np.float64)
    query = float(np.asarray(sensor["query_time"]))
    if recorder:
        sample_time = np.asarray(recorder["sample_time"], dtype=np.float64)
        sample_valid = np.asarray(recorder["sample_valid"], dtype=bool)
        if sample_time.shape != (t,):
            raise ValueError("recorder sample axis must match the evidence frame axis")
        if not np.allclose(sample_time, timestamps, rtol=0, atol=1e-5):
            raise ValueError("recorder samples must be aligned to evidence timestamps")
        if np.any(sample_time[sample_valid] > query):
            raise ValueError("future recorder sample in current-state evidence")
        if "trace_times" in recorder:
            trace_valid = np.asarray(recorder["trace_valid"], dtype=bool)
            trace_times = np.asarray(recorder["trace_times"], dtype=np.float64)
            if np.any(trace_times[trace_valid] > query):
                raise ValueError("future recorder trace sample in current-state evidence")
    ids = np.asarray(sensor["source_ids"], dtype=np.int64)
    segments = np.asarray(sensor["segment_ids"], dtype=np.int64)
    valid_times, valid_arrivals = timestamps[frame_valid], arrivals[frame_valid]
    valid_ids, valid_segments = ids[frame_valid], segments[frame_valid]
    if np.any(np.diff(valid_times) < 0) or np.any(np.diff(valid_segments) < 0):
        raise ValueError("valid observations and segment IDs must be chronological")
    if np.unique(valid_ids).size != valid_ids.size:
        raise ValueError("duplicate source identity among valid observations")
    if np.any(valid_times > query) or np.any(valid_arrivals > query):
        raise ValueError("unobserved future exposure or delayed arrival in evidence")
    if np.any(valid_arrivals < valid_times):
        raise ValueError("sensor arrival cannot precede exposure")
    if np.any(valid_segments < 0) or np.any(valid_segments >= m.max_segments):
        raise ValueError("segment ID outside configured range")
    if np.any(np.asarray(targets["change_times"]) > m.time_bins + 1):
        raise ValueError("change_times exceeds configured time bins")
    if np.any(np.asarray(targets["change_times"]) < 0):
        raise ValueError("change_times cannot be negative")
    entity_valid = np.asarray(targets["entity_valid"], dtype=bool)
    change_valid = np.asarray(targets["change_valid"], dtype=bool)
    if np.any(np.asarray(targets["pose_valid"], dtype=bool)[~entity_valid]):
        raise ValueError("padded entities cannot have valid pose coordinates")
    if np.any(change_valid & (~entity_valid[:, None] | ~entity_valid[None, :])):
        raise ValueError("change_valid refers to a padded entity")
    pair_entity_valid = entity_valid[:, None] & entity_valid[None, :]
    for name in ("relation_relevance_valid", "contact_valid"):
        if name in targets and np.any(np.asarray(targets[name], dtype=bool) & ~pair_entity_valid):
            raise ValueError(f"{name} refers to a padded entity")
    if "image_points_valid" in targets and np.any(
        np.asarray(targets["image_points_valid"], dtype=bool) & ~entity_valid
    ):
        raise ValueError("image_points_valid refers to a padded entity")
    if np.any(np.diag(change_valid)):
        raise ValueError("self-relations cannot have change labels")
    visibility = np.asarray(targets["visibility"])
    if visibility.size and (visibility.min() < 0 or visibility.max() > 1):
        raise ValueError("visibility must be a fraction in [0,1]")
    if "relation_relevance" in targets:
        relevance = np.asarray(targets["relation_relevance"])
        if relevance.size and (relevance.min() < 0 or relevance.max() > 1):
            raise ValueError("relation_relevance must be in [0,1]")
    if "contact_states" in targets:
        contacts = np.asarray(targets["contact_states"])
        if contacts.size and (contacts.min() < 0 or contacts.max() > 2):
            raise ValueError("contact_states must use 0=unknown, 1=no, 2=yes")
        contact_valid = np.asarray(targets["contact_valid"], dtype=bool)
        if np.any(contact_valid & (contacts == 0)):
            raise ValueError("valid contact annotations cannot use the unknown state")
