"""Command line interface. Use --help without installing simulator dependencies."""

import argparse
import json
from pathlib import Path

import numpy as np

from .episode import load_episode, resolve_bddl
from .rollout import VideoRecorder, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Replay stored controls or run a new AI policy")
    commands = parser.add_subparsers(dest="mode", required=True)
    for name in ("replay", "infer"):
        command = commands.add_parser(name)
        command.add_argument("--episode", required=True, type=Path)
        command.add_argument("--bddl-root", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path, help="New output directory")
        command.add_argument("--no-video", action="store_true")
        if name == "replay":
            command.add_argument("--joint-tolerance", type=float, default=1e-5)
        else:
            command.add_argument("--checkpoint", required=True, type=Path,
                                 help="Local merged OFT model; weights are not bundled")
            command.add_argument("--unnorm-key", required=True)
            command.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(argv)
    if args.mode == "infer" and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.mode == "replay" and args.joint_tolerance < 0:
        parser.error("--joint-tolerance must be nonnegative")
    metadata, trajectory = load_episode(args.episode)
    bddl = resolve_bddl(args.bddl_root, metadata)
    from libero.libero.envs import OffScreenRenderEnv
    policy = None
    if args.mode == "infer":
        from .policy import OpenVLAPolicy
        policy = OpenVLAPolicy(args.checkpoint, args.unnorm_key)
    args.output.mkdir(parents=True, exist_ok=False)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128,
                            camera_widths=128, control_freq=metadata["control_hz"])
    recorder = None if args.no_video else VideoRecorder(args.output, metadata["control_hz"])
    try:
        report, trace = run(env, metadata, trajectory, policy=policy, record=recorder,
                            max_steps=getattr(args, "max_steps", 300),
                            joint_tolerance=getattr(args, "joint_tolerance", 1e-5))
        np.savez_compressed(args.output / "trajectory.npz", **trace)
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if report.get("replay_matches", True) else 2
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
