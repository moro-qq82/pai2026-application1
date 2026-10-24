"""CPU checks for exported data and rollout boundaries; no GPU/simulator needed."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pai_demo.episode import array_sha256, load_episode, resolve_bddl
from pai_demo.rollout import run


class FakeEnvironment:
    """A deterministic joint integrator with reset and goal behavior."""

    def __init__(self, scale=1.0, goal_after=2):
        self.scale, self.goal_after = scale, goal_after
        self.calls, self.q, self.moves = [], np.zeros(7), 0
        self.robots = [SimpleNamespace(action_dim=7,
                       controller=SimpleNamespace(initial_joint=np.zeros(7)))]
        self.sim = SimpleNamespace(
            get_state=lambda: self.q.copy(),
            set_state_from_flattened=self.set_state,
            forward=lambda: self.calls.append("forward"),
        )

    def seed(self, value):
        self.calls.append(("seed", value))

    def reset(self):
        self.calls.append("reset")
        self.q, self.moves = np.zeros(7), 0

    def set_state(self, value):
        self.calls.append("set_state")
        self.q = value.copy()

    def step(self, action):
        self.calls.append(("step", action.copy()))
        if np.any(action):
            self.moves += 1
        self.q += np.asarray(action) * self.scale
        return {"robot0_joint_pos": self.q.copy()}, 0, self.moves >= self.goal_after, {}


def sample(initialization="load_initial_state"):
    actions = np.array([[.1] * 7, [.2] * 7], dtype=np.float32)
    post = np.cumsum(actions.astype(np.float64), axis=0)
    zero = np.zeros(7)
    metadata = dict(case_id="fixture", env_seed=3, policy_seed=4, instruction="move",
                    initialization=initialization, settle_steps=10, control_hz=20,
                    initial_state_sha256=array_sha256(zero),
                    post_settle_state_sha256=array_sha256(zero),
                    controller_initial_joint=zero.tolist(),
                    expected={"goal_reached": True})
    trajectory = dict(initial_state=zero, actions=actions, post_joint=post,
                      pre_joint=np.vstack([zero, post[:-1]]))
    return metadata, trajectory


class RolloutChecks(unittest.TestCase):
    def test_replay_settles_before_recording_and_matches_reference(self):
        metadata, trajectory = sample()
        env, frames = FakeEnvironment(), []
        report, _ = run(env, metadata, trajectory, record=frames.append)
        self.assertTrue(report["replay_matches"])
        self.assertEqual(env.calls[:4], [("seed", 3), "reset", "set_state", "forward"])
        self.assertEqual(len([call for call in env.calls if isinstance(call, tuple)
                              and call[0] == "step"]), 12)
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0]["robot0_joint_pos"], np.zeros(7))

    def test_extra_reset_is_preserved(self):
        metadata, trajectory = sample("capture_reset_before_rollout")
        env = FakeEnvironment()
        report, _ = run(env, metadata, trajectory)
        self.assertTrue(report["replay_matches"])
        self.assertEqual(env.calls[:5], [("seed", 3), "reset", "reset", "set_state", "forward"])

    def test_joint_drift_is_not_reported_as_reproduced(self):
        metadata, trajectory = sample()
        report, _ = run(FakeEnvironment(scale=1.1), metadata, trajectory)
        self.assertFalse(report["replay_matches"])
        self.assertGreater(report["joint_max_abs_error"]["post_joint"], .02)

    def test_replay_does_not_truncate_on_early_goal(self):
        metadata, trajectory = sample()
        report, trace = run(FakeEnvironment(goal_after=1), metadata, trajectory)
        self.assertEqual(report["first_goal_step"], 1)
        self.assertEqual(len(trace["actions"]), 2)

    def test_new_inference_stops_at_goal_and_uses_observation(self):
        metadata, trajectory = sample()
        resets, observations = [], []
        policy = SimpleNamespace(reset=lambda *args: resets.append(args),
                                 get_action=lambda obs: observations.append(obs) or np.ones(7))
        report, _ = run(FakeEnvironment(goal_after=1), metadata, trajectory, policy=policy)
        self.assertEqual(report["mode"], "new_model_inference")
        self.assertEqual(report["steps"], 1)
        self.assertNotIn("replay_matches", report)
        self.assertEqual(resets, [("move", 4)])
        self.assertEqual(len(observations), 1)

    def test_invalid_model_action_fails(self):
        metadata, trajectory = sample()
        policy = SimpleNamespace(reset=lambda *args: None,
                                 get_action=lambda obs: np.full(7, np.nan))
        with self.assertRaisesRegex(ValueError, "invalid 7D"):
            run(FakeEnvironment(), metadata, trajectory, policy=policy)

    def test_bddl_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "inside"):
                resolve_bddl(folder, {"bddl_relative_path": "../outside.bddl"})

    def test_exported_eight_episodes_are_intact(self):
        manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
        entries = {entry["case_id"]: entry for entry in manifest["episodes"]}
        folders = sorted((ROOT / "data" / "episodes").iterdir())
        self.assertEqual(len(folders), 8)
        self.assertEqual(set(entries), {folder.name for folder in folders})
        for folder in folders:
            with self.subTest(case=folder.name):
                digest = hashlib.sha256((folder / "metadata.json").read_bytes()).hexdigest()
                self.assertEqual(digest, entries[folder.name]["metadata_sha256"])
                metadata, trajectory = load_episode(folder)
                self.assertEqual(metadata["case_id"], folder.name)
                self.assertEqual(len(trajectory["actions"]), metadata["steps"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
