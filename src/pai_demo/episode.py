"""Load the portable episode export without importing a simulator or model."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_episode(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    trajectory_path = directory / "trajectory.npz"
    if file_sha256(trajectory_path) != metadata["trajectory_sha256"]:
        raise ValueError("Trajectory file does not match its recorded SHA-256")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    steps = metadata["steps"]
    for key in ("actions", "pre_joint", "post_joint"):
        value = trajectory[key]
        if value.shape != (steps, 7) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}: expected {steps} finite 7D vectors")
    state = trajectory["initial_state"]
    if state.ndim != 1 or not np.isfinite(state).all():
        raise ValueError("Invalid flattened initial state")
    if array_sha256(state) != metadata["initial_state_sha256"]:
        raise ValueError("Initial simulator state hash differs from metadata")
    return metadata, trajectory


def resolve_bddl(root, metadata):
    root = Path(root).resolve()
    target = (root / metadata["bddl_relative_path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("BDDL path must remain inside the selected BDDL root")
    if file_sha256(target) != metadata["bddl_sha256"]:
        raise ValueError("BDDL differs from the recorded task; use the pinned upstream assets")
    return target
