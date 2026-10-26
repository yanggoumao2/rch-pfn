"""Command-line project entrypoints. Paths and experiment sizes are configuration inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import config_from_dict, load_config
from .data import EpisodeDataset, generate_dataset, import_labutopia_episode, validate_manifest
from .inference import belief_to_dict
from .model import RCHPFN
from .runner import device_for, environment_report, evaluate, load_checkpoint, train, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="RCH-PFN research project")
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="procedural kinematic RGB integration dataset")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    labutopia = commands.add_parser("import-labutopia", help="import one LabUtopia recorder episode")
    labutopia.add_argument("--motion", "--motion-path", dest="motion", type=Path, required=True)
    labutopia.add_argument("--config", type=Path, required=True)
    labutopia.add_argument("--output", type=Path, required=True)
    labutopia.add_argument("--system-id")
    labutopia.add_argument("--episode-id")
    labutopia.add_argument("--split", choices=["train", "val", "test"], default="train")
    labutopia.add_argument("--policy", choices=["deployable", "recorder_oracle"])
    labutopia.add_argument("--calibrated-rgb", type=Path, help="official calibrated RGB episode HDF5")
    validate = commands.add_parser("validate-data")
    validate.add_argument("--data", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--init-from", type=Path)
    training.add_argument("--steps", type=int)
    training.add_argument("--device", default=None)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--output", type=Path, required=True)
    inference = commands.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--data", type=Path, required=True)
    inference.add_argument("--split", choices=["train", "val", "test"], default="test")
    inference.add_argument("--index", type=int, default=0)
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--output", type=Path, required=True)
    commands.add_parser("environment")
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.command == "environment":
        payload = environment_report()
    elif args.command == "generate":
        manifest = generate_dataset(load_config(args.config), args.output)
        payload = {
            "output": str(args.output),
            "episodes": len(manifest["episodes"]),
            "backend": manifest["backend"],
        }
    elif args.command == "import-labutopia":
        cfg = load_config(args.config)
        payload = import_labutopia_episode(
            args.motion,
            args.output,
            config=cfg,
            system_id=args.system_id,
            episode_id=args.episode_id,
            split=args.split,
            policy=args.policy or cfg.model.recorder_policy,
            calibrated_rgb_path=args.calibrated_rgb,
        )
    elif args.command == "validate-data":
        manifest = validate_manifest(args.data)
        payload = {"valid": True, "episodes": len(manifest["episodes"])}
    elif args.command == "train":
        cfg = load_config(args.config)
        if args.steps is not None:
            cfg.train.steps = args.steps
        if args.device is not None:
            cfg.train.device = args.device
        payload = train(cfg, args.data, args.run, args.resume, args.init_from)
    elif args.command == "evaluate":
        payload = evaluate(args.checkpoint, args.data, args.split, args.device)
        write_json(args.output, payload)
    else:
        torch.set_num_threads(4)
        checkpoint = load_checkpoint(args.checkpoint)
        cfg = config_from_dict(checkpoint["config"])
        device = device_for(args.device)
        model = RCHPFN(cfg.model).to(device).eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        dataset = EpisodeDataset(args.data, args.split, cfg)
        evidence, _ = dataset[args.index]
        evidence = evidence.to(device)
        with torch.no_grad():
            output = model(evidence)
        payload = belief_to_dict(output, evidence, cfg.model)
        write_json(args.output, payload)
        payload = {
            "output": str(args.output),
            "control_ready": payload["control_ready"],
            "mode_weights": payload["mode_weights"],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
