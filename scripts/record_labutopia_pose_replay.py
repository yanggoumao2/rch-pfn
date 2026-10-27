"""Render a verified insertion motion through LabUtopia's official collector.

This script is kept in the RCH-PFN repository because it is part of the
project's data-producing pipeline.  Isaac Sim and the LabUtopia runtime still
run on the HP workstation; the repository only owns the entry point and the
capture contract.  The source motion and prepared USD stage remain explicit
inputs so a capture can be reproduced and audited later.
The default capture mode accepts only a verified success.  An explicit
``--capture-mode failure-safe`` preserves a completed non-success outcome for
research while keeping all camera, timestamp, and RGB gates unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PATH_TRACING_SPP = 128
PATH_TRACING_TOTAL_SPP = 128
PATH_TRACING_MAX_BOUNCES = 12
PATH_TRACING_MAX_SPECULAR_TRANSMISSION_BOUNCES = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_frames(frames: list[dict], every: int) -> list[int]:
    """Choose a bounded replay sequence while retaining phase transitions."""
    selected = set(range(0, len(frames), every))
    selected.update((0, len(frames) - 1))
    selected.update(
        index
        for index in range(1, len(frames))
        if frames[index]["phase"] != frames[index - 1]["phase"]
    )
    return sorted(selected)


def _pose_matrix(position: list[float], quaternion_wxyz: list[float]) -> np.ndarray:
    """Build a USD-compatible world transform from a motion pose."""
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("source motion contains an invalid body quaternion")
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]] / norm).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _set_world_pose(stage, prim_path: str, position: list[float], quaternion_wxyz: list[float]) -> None:
    """Apply one recorded world pose as a session-layer transform override."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid() or not prim.IsActive():
        raise ValueError(f"pose replay prim is missing or inactive: {prim_path}")
    parent_world = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    ).T
    local = np.linalg.inv(parent_world) @ _pose_matrix(position, quaternion_wxyz)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(opSuffix="pfnReplay").Set(Gf.Matrix4d(local.T.tolist()))


def _look_at_usd(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    forward = target - position
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    camera_to_world = np.column_stack((right, up, -forward))
    return Rotation.from_matrix(camera_to_world).as_quat()[[3, 0, 1, 2]]


def _wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)[:3, :3]).as_quat()[[3, 0, 1, 2]]


def _to_uint8_rgb(image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Validate and normalize one camera RGB frame before it reaches the collector."""
    image = np.asarray(image)
    width, height = resolution
    if image.ndim != 3 or image.shape[:2] != (height, width) or image.shape[2] < 3:
        raise ValueError(f"camera returned unexpected RGB shape: {image.shape}")
    image = image[..., :3]
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
            raise ValueError("camera returned nonnumeric or non-finite RGB values")
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not image.any() or image.std() < 1.0:
        raise ValueError("camera returned a blank RGB frame")
    return image.copy()


def _camera_to_base(camera, initial_base_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, orientation_wxyz = camera.get_world_pose(camera_axes="usd")
    camera_world = _pose_matrix(np.asarray(position).tolist(), np.asarray(orientation_wxyz).tolist())
    return np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64), np.linalg.inv(initial_base_world) @ camera_world


def _render_frame_reference(frame: dict) -> tuple[int, int]:
    if not isinstance(frame, dict):
        raise ValueError("camera is missing Isaac Sim rendering-frame metadata")
    numerator = frame.get("referenceTimeNumerator")
    denominator = frame.get("referenceTimeDenominator")
    if not isinstance(numerator, (int, np.integer)) or not isinstance(
        denominator, (int, np.integer)
    ):
        raise ValueError("camera has incomplete Isaac Sim rendering-frame metadata")
    if numerator < 0 or denominator <= 0:
        raise ValueError("camera has invalid Isaac Sim rendering-frame metadata")
    return int(numerator), int(denominator)


def _render_frame_number(frame: dict) -> int:
    """Return Isaac Sim's shared rational render-frame numerator."""
    if not isinstance(frame, dict):
        raise ValueError(f"camera rendering frame metadata is not a mapping: {frame!r}")
    numerator = frame.get("referenceTimeNumerator")
    denominator = frame.get("referenceTimeDenominator")
    if not isinstance(numerator, (int, np.integer)) or not isinstance(
        denominator, (int, np.integer)
    ):
        raise ValueError(f"camera rendering frame metadata is incomplete: {frame!r}")
    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"camera rendering frame metadata is invalid: {frame!r}")
    return int(numerator)


def validate_source_outcome(outcome: dict, capture_mode: str) -> dict:
    """Validate outcome provenance without dropping failed episodes.

    The default success mode remains strict so an accidental invocation cannot
    turn a failed replay into a training record.  Failure-safe mode only
    relaxes the success predicate; it still requires independently produced,
    explicit boolean outcome fields and returns the original mapping for
    collector provenance.
    """
    if capture_mode not in {"success", "failure-safe"}:
        raise ValueError(f"unsupported capture mode: {capture_mode!r}")
    required = ("success", "task_success", "strict_geometry_pass")
    if not isinstance(outcome, dict):
        raise ValueError("source outcome must be a JSON object")
    for key in required:
        if type(outcome.get(key)) is not bool:
            raise ValueError(f"source outcome field {key!r} must be boolean")
    if capture_mode == "success" and not all(outcome[key] for key in required):
        raise ValueError("source episode is not a verified strict-geometry success")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--outcome", type=Path)
    parser.add_argument(
        "--capture-mode",
        choices=("success", "failure-safe"),
        default="success",
        help="success requires all outcome gates; failure-safe preserves non-success outcomes",
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labutopia-root",
        type=Path,
        required=True,
        help="LabUtopia checkout that provides factories and the official collector",
    )
    parser.add_argument("--sample-every", type=int, default=6)
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="square compatibility resolution used for any unspecified dimension",
    )
    parser.add_argument("--width", type=int, help="RGB width; defaults to --resolution")
    parser.add_argument("--height", type=int, help="RGB height; defaults to --resolution")
    args = parser.parse_args()
    args.motion = args.motion.resolve()
    args.outcome = (args.outcome or args.motion.with_name("outcome.json")).resolve()
    args.stage = args.stage.resolve()
    args.output = args.output.resolve()
    args.labutopia_root = args.labutopia_root.resolve()
    width = args.width if args.width is not None else args.resolution
    height = args.height if args.height is not None else args.resolution
    resolution = (width, height)
    # The recorder entry point belongs to RCH-PFN, while the simulator-side
    # collector implementation belongs to the LabUtopia checkout.  Make that
    # dependency explicit instead of relying on the script's old location.
    if not args.labutopia_root.is_dir():
        raise FileNotFoundError(f"LabUtopia root does not exist: {args.labutopia_root}")
    sys.path.insert(0, str(args.labutopia_root))
    if args.sample_every < 1 or width < 32 or height < 32:
        raise ValueError("sample-every must be positive and width/height must each be at least 32")
    if args.output.exists():
        raise FileExistsError(f"refusing to reuse existing output directory: {args.output}")
    if not args.motion.is_file() or not args.outcome.is_file() or not args.stage.is_file():
        raise FileNotFoundError("motion, outcome, and prepared stage files must exist")

    motion_bytes = args.motion.read_bytes()
    outcome_bytes = args.outcome.read_bytes()
    motion = json.loads(motion_bytes)
    outcome = json.loads(outcome_bytes)
    frames = motion.get("frames")
    body_paths = motion.get("body_paths")
    if not isinstance(frames, list) or not frames or not isinstance(body_paths, list):
        raise ValueError("motion JSON must contain body_paths and nonempty frames")
    validate_source_outcome(outcome, args.capture_mode)
    selected = _selected_frames(frames, args.sample_every)
    source_times = np.asarray([frames[index]["time"] for index in selected], dtype=np.float64)
    if not np.isfinite(source_times).all() or np.any(np.diff(source_times) <= 0):
        raise ValueError("selected source timestamps must be finite and strictly increasing")

    # SimulationApp must be imported and constructed before most Isaac Sim
    # modules.  The launch command therefore uses LabUtopia's Isaac runtime
    # Python, rather than the RCH-PFN virtualenv used for model-only tests.
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "multi_gpu": False,
            "extra_args": ["--allow-root"],
        }
    )
    collector = None
    try:
        import carb
        import omni.timeline
        import omni.usd
        from factories.collector_factory import create_collector
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", "PathTracing")
        settings.set("/rtx/sceneDb/ambientLightIntensity", 0.0)
        settings.set("/rtx/post/histogram/enabled", False)
        settings.set("/rtx/pathtracing/spp", PATH_TRACING_SPP)
        settings.set("/rtx/pathtracing/totalSpp", PATH_TRACING_TOTAL_SPP)
        settings.set("/rtx/pathtracing/maxBounces", PATH_TRACING_MAX_BOUNCES)
        settings.set(
            "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
            PATH_TRACING_MAX_SPECULAR_TRANSMISSION_BOUNCES,
        )
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)
        settings.set("/rtx/pathtracing/optixDenoiser/temporalMode/enabled", False)
        settings.set("/rtx/pathtracing/cached/enabled", False)
        # Each replayed pose is a new image sample. Reset accumulation when
        # the pose/time changes so previous wrist or third-person frames do
        # not ghost into the current frame.
        settings.set("/rtx/resetPtAccumOnAnimTimeChange", True)
        settings.set("/rtx/post/dof/enabled", False)
        # Open the prepared scene once and apply replay poses in the session
        # layer, so the source USD asset is never modified on disk.
        context = omni.usd.get_context()
        if not context.open_stage(str(args.stage)):
            raise RuntimeError(f"failed to open prepared stage: {args.stage}")
        for _ in range(4):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        # Match the accepted insertion_movie_v6 photographic lighting. The
        # previous replay added a 1000-intensity white dome, which washed out
        # the laboratory and produced the current harsh look.
        lighting_specs = {
            "WindowKey": ((-2.0, 1.9, 2.8), (0.0, -0.4, 0.7), (1.8, 2.0), 7600.0, (0.93, 0.97, 1.0)),
            "CeilingFill": ((1.5, -0.7, 2.9), (0.0, -0.3, 0.7), (2.0, 2.0), 240.0, (1.0, 0.94, 0.86)),
            "FrontBounce": ((-0.6, -2.0, 2.2), (0.0, -0.3, 0.8), (2.8, 2.0), 280.0, (1.0, 0.98, 0.95)),
        }
        shared_light_exposure = 0.0
        for name, (eye, target, size, power, color) in lighting_specs.items():
            rect = UsdLux.RectLight.Define(stage, f"/World/PFNReplayLighting/{name}")
            rect.CreateWidthAttr(size[0]).Set(size[0])
            rect.CreateHeightAttr(size[1]).Set(size[1])
            api = UsdLux.LightAPI(rect.GetPrim())
            api.CreateIntensityAttr(power).Set(power)
            api.CreateColorAttr(Gf.Vec3f(*color)).Set(Gf.Vec3f(*color))
            api.CreateExposureAttr(shared_light_exposure).Set(shared_light_exposure)
            UsdGeom.Xformable(rect).ClearXformOpOrder()
            UsdGeom.Xformable(rect).AddTransformOp().Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            )
        environment = stage.GetPrimAtPath("/World/Photo/EnvironmentFill")
        if environment and environment.IsValid():
            environment_api = UsdLux.LightAPI(environment)
            environment_api.GetIntensityAttr().Set(120.0)
            environment_api.CreateExposureAttr(shared_light_exposure).Set(shared_light_exposure)
        fill_dome = UsdLux.DomeLight.Define(stage, "/World/PFNReplayLighting/EnvironmentFill")
        fill_dome.CreateIntensityAttr(120.0).Set(120.0)
        fill_dome_api = UsdLux.LightAPI(fill_dome.GetPrim())
        fill_dome_api.CreateExposureAttr(shared_light_exposure).Set(shared_light_exposure)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        stage.SetTimeCodesPerSecond(480.0)
        stage.SetFramesPerSecond(480.0)
        timeline.set_time_codes_per_second(120.0)
        timeline.set_current_time(0.0)
        if len(body_paths) != 13:
            raise ValueError(f"expected 13 recorded body paths, received {len(body_paths)}")
        if any(not stage.GetPrimAtPath(path).IsValid() for path in body_paths):
            raise ValueError("prepared stage does not contain every source motion body path")

        first = frames[0]
        initial_base_world = _pose_matrix(first["positions"][0], first["orientations_wxyz"][0])
        camera_root = "/World/PFNReplayCameras"
        UsdGeom.Xform.Define(stage, camera_root)
        # Reuse the previously accepted workbench view from insertion_movie_v6.
        # This is the full LabUtopia laboratory composition, rather than the
        # stripped physics replay stage used only for collision verification.
        # Exact close view from the accepted insertion_movie_v6 photographic
        # render: gripper, tube and insertion fixture are the subject.
        third_position = np.array([0.16, 0.16, 1.86], dtype=np.float64)
        scene_target = np.array([0.54, -0.44, 1.18], dtype=np.float64)
        third = Camera(
            prim_path=f"{camera_root}/third_person",
            translation=third_position,
            name="third_person",
            frequency=120,
            resolution=resolution,
        )
        third.set_local_pose(
            translation=third_position,
            orientation=_look_at_usd(third_position, scene_target),
            camera_axes="usd",
        )
        # USD old-camera ratio was 32 mm / 36 mm aperture. Isaac Sim's
        # Camera API retains its 2.0955 cm default aperture after initialize,
        # so use the equivalent ~1.86 mm focal length for the same FOV.
        third.set_focal_length(32.0)
        third.set_clipping_range(near_distance=0.05, far_distance=8.0)

        # Attach the wrist camera below panda_hand so its local pose follows
        # the actual wrist link through the replay, as in LabUtopia's camera
        # probe. A fixed hand-local pose keeps the gripper and tube in the
        # optical neighborhood without placing the camera inside the seat.
        wrist_path = f"{body_paths[9]}/pfn_wrist_camera"
        # Libero-style wrist view: the camera is mounted just behind the
        # hand, looking along the gripper toward the work surface so the two
        # finger pads form the lower foreground and the held tube is centered.
        wrist_local_translation = np.array([-0.20, 0.0, -0.02], dtype=np.float64)
        # Libero-style wrist view selected from the validated candidate 05-26.
        wrist_local_orientation = np.array(
            [0.20083, 0.67799, -0.67799, -0.20083], dtype=np.float64
        )
        wrist = Camera(
            prim_path=wrist_path,
            translation=wrist_local_translation,
            name="wrist",
            frequency=120,
            resolution=resolution,
        )
        wrist.set_local_pose(
            translation=wrist_local_translation,
            orientation=wrist_local_orientation,
            camera_axes="usd",
        )
        wrist.set_focal_length(1.0)
        wrist.set_clipping_range(near_distance=0.05, far_distance=4.0)
        cameras = {"third_person": third, "wrist": wrist}
        camera_configs = [
            {
                "name": name,
                "image_type": "rgb",
                "prim_path": f"{camera_root}/third_person" if name == "third_person" else wrist_path,
                "resolution": list(resolution),
                "focal_length": float(camera.get_focal_length()),
                "camera_axes": "usd",
            }
            for name, camera in cameras.items()
        ]
        for camera in cameras.values():
            camera.initialize()
        # Camera.initialize() restores sensor defaults. Apply the old
        # insertion_movie_v6 workbench aperture after initialization so the
        # 32 mm/36 mm composition is actually used by the recorder.
        third_prim = stage.GetPrimAtPath(f"{camera_root}/third_person")
        UsdGeom.Camera(third_prim).CreateHorizontalApertureAttr(36.0).Set(36.0)
        UsdGeom.Camera(third_prim).CreateVerticalApertureAttr(36.0).Set(36.0)
        for camera_path in (f"{camera_root}/third_person", wrist_path):
            usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
            usd_camera.CreateExposureAttr(0.0).Set(0.0)
        for camera in cameras.values():
            camera.resume()
        for _ in range(12):
            app.update()
        for camera in cameras.values():
            camera.pause()

        args.output.mkdir(parents=True, exist_ok=False)
        collector = create_collector(
            "calibrated_rgb",
            camera_configs=camera_configs,
            save_dir=str(args.output),
            max_episodes=1,
            max_workers=1,
            compression="gzip",
        )
        motion_sha256 = hashlib.sha256(motion_bytes).hexdigest()
        outcome_sha256 = hashlib.sha256(outcome_bytes).hexdigest()
        collector.set_outcome(outcome, motion_sha256, outcome_sha256)
        collector.set_task_properties(
            {
                "task": "test_tube_insertion",
                "capture_mode": "isaacsim_pose_replay",
                "outcome_policy": args.capture_mode,
                "sample_every_source_frames": args.sample_every,
                "source_motion_sha256": motion_sha256,
                "source_outcome_sha256": outcome_sha256,
            }
        )

        intrinsic = {name: np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64) for name, camera in cameras.items()}
        preview_indices = {0, len(selected) // 2, len(selected) - 1}
        preview_stats = {}
        preview_images: dict[tuple[int, str], np.ndarray] = {}
        camera_render_times: list[dict[str, float]] = []
        camera_render_references: list[dict[str, list[int]]] = []
        previous_render_time = -float("inf")
        previous_render_reference: tuple[int, int] | None = None
        timeline_dt = 1.0 / float(timeline.get_time_codes_per_second())
        capture_origin_time = 0.1
        # Each selected source pose is rendered only after both cameras expose
        # the same measured rendering time and rational render-frame token.
        # This preserves cross-view synchronization without pretending that
        # the camera clock equals the source motion clock.
        for capture_index, source_index in enumerate(selected):
            frame = frames[source_index]
            positions = frame["positions"]
            orientations = frame["orientations_wxyz"]
            if len(positions) != len(body_paths) or len(orientations) != len(body_paths):
                raise ValueError(f"body pose count changed at source frame {source_index}")
            for body_index, prim_path in enumerate(body_paths):
                _set_world_pose(stage, prim_path, positions[body_index], orientations[body_index])
            source_time = float(frame["time"])
            render_target_time = capture_origin_time + source_time - float(source_times[0])
            synchronized_frames = None
            last_camera_diagnostics: dict[str, object] = {}
            # The initialization warm-up pauses the sensors so it cannot
            # leak a stale frame into the first capture. Resume both sensors
            # for this pose; pausing the timeline below still lets the camera
            # renderer accumulate samples at one fixed simulation time.
            for camera in cameras.values():
                camera.resume()
            # Large phase-transition gaps can require many Kit updates before
            # the camera queue reaches the requested source timestamp.
            for wait_iteration in range(512):
                # Render the full accepted PathTracing sample budget in one
                # update. Multiple updates at the same pose advance Kit's
                # camera queue and are the source of the historical timestamp
                # drift for adjacent source frames.
                timeline.pause()
                timeline.set_current_time(render_target_time - timeline_dt)
                timeline.play()
                app.update()
                current_frames = {
                    name: camera.get_current_frame(clone=True) for name, camera in cameras.items()
                }
                try:
                    current_times = {
                        name: float(current_frames[name]["rendering_time"])
                        for name in cameras
                    }
                    current_references = {
                        name: _render_frame_reference(current_frames[name]["rendering_frame"])
                        for name in cameras
                    }
                    if any(current_frames[name].get("rgb") is None for name in cameras):
                        last_camera_diagnostics = {
                            name: {
                                "keys": sorted(str(key) for key in current_frames[name]),
                                "rgb_shape": None
                                if current_frames[name].get("rgb") is None
                                else list(np.asarray(current_frames[name]["rgb"]).shape),
                            }
                            for name in cameras
                        }
                        continue
                    for name in cameras:
                        try:
                            _to_uint8_rgb(current_frames[name]["rgb"], resolution)
                        except ValueError as exc:
                            image = np.asarray(current_frames[name]["rgb"])
                            last_camera_diagnostics["invalid_rgb"] = {
                                name: {
                                    "error": repr(exc),
                                    "min": float(image[..., :3].min()) if image.ndim == 3 else None,
                                    "max": float(image[..., :3].max()) if image.ndim == 3 else None,
                                    "std": float(image[..., :3].std()) if image.ndim == 3 else None,
                                }
                                for name in cameras
                            }
                            raise
                    last_camera_diagnostics = {
                        "source_time": source_time,
                        "render_target_time": render_target_time,
                        "previous_render_time": previous_render_time,
                        "rendering_times": current_times,
                        "render_frame_references": {
                            name: list(reference) for name, reference in current_references.items()
                        },
                        "rgb_shapes": {
                            name: list(np.asarray(current_frames[name]["rgb"]).shape) for name in cameras
                        },
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    last_camera_diagnostics = {
                        name: {
                            "keys": sorted(str(key) for key in current_frames[name]),
                            "rendering_time": repr(current_frames[name].get("rendering_time")),
                            "rendering_frame": repr(current_frames[name].get("rendering_frame")),
                            "rgb_shape": None
                            if current_frames[name].get("rgb") is None
                            else list(np.asarray(current_frames[name]["rgb"]).shape),
                        }
                        for name in cameras
                    }
                    last_camera_diagnostics["error"] = repr(exc)
                    continue
                current_reference = next(iter(current_references.values()))
                reference_is_new = previous_render_reference is None or (
                    current_reference[0] * previous_render_reference[1]
                    > previous_render_reference[0] * current_reference[1]
                )
                timestamp_is_target = max(
                    abs(current_times[name] - render_target_time) for name in cameras
                ) <= max(2.0 * timeline_dt, 0.02)
                # Timestamp correctness is the hard gate. The complete RTX
                # sample budget was rendered in the single update above.
                minimum_render_updates = 1
                if (
                    len(set(current_references.values())) == 1
                    and min(current_times.values()) > previous_render_time
                    and max(current_times.values()) - min(current_times.values()) <= 1e-6
                    and reference_is_new
                    and timestamp_is_target
                    and wait_iteration + 1 >= minimum_render_updates
                ):
                    synchronized_frames = (current_frames, current_times, current_references)
                    break
            for camera in cameras.values():
                camera.pause()
            if synchronized_frames is None:
                timeline.pause()
                raise RuntimeError(
                    f"cameras did not render one synchronized frame at source index {source_index}; "
                    f"last diagnostics: {last_camera_diagnostics}"
                )
            current_frames, actual_times, actual_references = synchronized_frames
            timeline.pause()
            previous_render_time = float(np.mean(list(actual_times.values())))
            previous_render_reference = next(iter(actual_references.values()))

            images, extrinsics, render_numbers = {}, {}, {}
            for camera_name, camera in cameras.items():
                current_frame = current_frames[camera_name]
                rgb = _to_uint8_rgb(current_frame["rgb"], resolution)
                images[f"{camera_name}_rgb"] = rgb
                camera_intrinsic, extrinsic = _camera_to_base(camera, initial_base_world)
                if not np.allclose(camera_intrinsic, intrinsic[camera_name], rtol=0, atol=1e-6):
                    raise ValueError(f"{camera_name} intrinsics changed during capture")
                extrinsics[camera_name] = extrinsic
                render_numbers[camera_name] = capture_index + 1
                if capture_index in preview_indices:
                    preview_images[(capture_index, camera_name)] = rgb.copy()
                    preview_stats[f"{camera_name}:{capture_index}"] = {
                        "source_frame_index": source_index,
                        "timestamp_s": source_time,
                        "camera_rendering_time_s": actual_times[camera_name],
                        "render_sequence_number": render_numbers[camera_name],
                        "render_frame_reference": list(actual_references[camera_name]),
                        "rgb_min": int(rgb.min()),
                        "rgb_max": int(rgb.max()),
                        "rgb_std": float(rgb.std()),
                        "render_frame_keys": sorted(str(key) for key in current_frame),
                    }
            camera_render_times.append(actual_times)
            camera_render_references.append(
                {name: list(reference) for name, reference in actual_references.items()}
            )

            joint_positions = np.asarray(frame["joint_positions"], dtype=np.float32)
            action = np.asarray(frame["applied_targets"], dtype=np.float32)
            # Store measured camera time as the canonical recorder timestamp;
            # source time and source frame index remain separate provenance.
            collector.cache_step(
                images,
                joint_positions,
                action,
                language_instruction="Insert the test tube into the centrifuge fixture",
                timestamp_s=previous_render_time,
                source_timestamp_s=source_time,
                camera_timestamps=actual_times,
                camera_intrinsics=intrinsic,
                camera_extrinsics=extrinsics,
                source_frame_index=source_index,
                render_sequence_numbers=render_numbers,
                render_frame_references=actual_references,
            )
        preview_dir = args.output / "previews"
        preview_dir.mkdir()
        from PIL import Image

        for (capture_index, camera_name), image in preview_images.items():
            preview = Image.fromarray(image)
            if preview.size != resolution:
                raise ValueError(
                    f"preview image has unexpected size for {camera_name}: "
                    f"{preview.size}, expected {resolution}"
                )
            preview.save(preview_dir / f"{capture_index:04d}_{camera_name}.png")
        collector.write_cached_data()
        collector.close()
        collector = None

        report = {
            "schema": "labutopia-pose-replay-capture-v1",
            "capture_mode": "isaacsim_pose_replay",
            "outcome_policy": args.capture_mode,
            "official_collector": "calibrated_rgb",
            "source_motion_sha256": motion_sha256,
            "source_outcome_sha256": outcome_sha256,
            "success": outcome["success"],
            "task_success": outcome["task_success"],
            "strict_geometry_pass": outcome["strict_geometry_pass"],
            "source_frame_count": len(frames),
            "captured_frame_count": len(selected),
            "source_frame_indices": selected,
            "source_timestamps_s": source_times.tolist(),
            "camera_rendering_times_s": camera_render_times,
            "camera_render_frame_references": camera_render_references,
            "camera_names": list(cameras),
            "camera_to_source_timestamp_max_abs_error_s": float(
                max(
                    abs(camera_render_times[row][name] - source_times[row])
                    for row in range(len(source_times))
                    for name in cameras
                )
            ),
            "camera_to_source_timestamp_mean_signed_offset_s": float(
                np.mean(
                    [
                        camera_render_times[row][list(cameras)[0]] - source_times[row]
                        for row in range(len(source_times))
                    ]
                )
            ),
            "camera_to_source_timestamp_drift_range_s": float(
                np.ptp(
                    [
                        camera_render_times[row][list(cameras)[0]] - source_times[row]
                        for row in range(len(source_times))
                    ]
                )
            ),
            "camera_cross_view_timestamp_max_abs_error_s": float(
                max(
                    max(camera_render_times[row].values()) - min(camera_render_times[row].values())
                    for row in range(len(source_times))
                )
            ),
            "timestamp_semantics": "timestamps are measured Isaac Sim rendering_time_s; source_timestamps_s are motion.frames[].time",
            "capture_origin_time_s": capture_origin_time,
            "camera_configs": camera_configs,
            "preview_stats": preview_stats,
        }
        (args.output / "capture_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    except BaseException as exc:
        print(f"pose replay capture failed: {exc!r}", file=sys.stderr, flush=True)
        raise
    finally:
        if collector is not None:
            collector.close()
        app.close()


if __name__ == "__main__":
    main()
