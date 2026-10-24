"""Standalone LIBERO rollout, without competition pipeline or scoring code.

Replay executes stored controls; it does not query an AI model. New inference
uses a policy object. The default frame convention matches the old recordings:
one pre-action observation per control step, display rotated 180 degrees.
"""

import numpy as np

from .episode import array_sha256


def initialize(env, metadata, initial_state):
    env.seed(metadata["env_seed"])
    if metadata["initialization"] == "capture_reset_before_rollout":
        env.reset()
        captured = np.asarray(env.sim.get_state().flatten())
        if array_sha256(captured) != metadata["initial_state_sha256"]:
            raise ValueError("Extra-reset state differs; simulator/assets are not aligned")
    elif metadata["initialization"] != "load_initial_state":
        raise ValueError("Unsupported episode initialization")
    env.reset()
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    for _ in range(metadata["settle_steps"]):
        observation, _, _, _ = env.step(np.zeros(env.robots[0].action_dim))
    return observation


def run(env, metadata, trajectory, *, policy=None, record=None, max_steps=300,
        joint_tolerance=1e-5):
    """Return a trace and report; caller owns the environment and video writer.

    In replay mode all stored actions are executed, even if the simulator reports
    an earlier goal. This permits a complete trace comparison. A difference is
    reported as a failed match, never silently described as reproduced.
    """
    observation = initialize(env, metadata, trajectory["initial_state"])
    state_hash_matches = (
        array_sha256(env.sim.get_state().flatten()) == metadata["post_settle_state_sha256"]
    )
    controller_matches = bool(np.allclose(
        env.robots[0].controller.initial_joint, metadata["controller_initial_joint"],
        atol=joint_tolerance, rtol=0,
    ))
    replay = policy is None
    if policy is not None:
        policy.reset(metadata["instruction"], metadata["policy_seed"])
    count = len(trajectory["actions"]) if replay else max_steps
    actions, pre_joints, post_joints = [], [], []
    first_goal_step = None
    done = False
    for step in range(count):
        action = trajectory["actions"][step] if replay else policy.get_action(observation)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Policy returned an invalid 7D control")
        if record is not None:
            record(observation)
        pre_joints.append(np.asarray(observation["robot0_joint_pos"]).copy())
        observation, _, done, _ = env.step(action)
        actions.append(action.copy())
        post_joints.append(np.asarray(observation["robot0_joint_pos"]).copy())
        if done and first_goal_step is None:
            first_goal_step = step + 1
        if done and not replay:
            break
    trace = {"actions": np.asarray(actions), "pre_joint": np.asarray(pre_joints),
             "post_joint": np.asarray(post_joints)}
    result = {
        "mode": "recorded_action_replay" if replay else "new_model_inference",
        "case_id": metadata["case_id"], "steps": len(actions),
        "control_hz": metadata["control_hz"], "goal_reached": bool(done),
        "first_goal_step": first_goal_step,
        "post_settle_state_hash_matches": state_hash_matches,
        "controller_initial_joint_matches": controller_matches,
        "original_recorded_outcome": metadata["expected"],
        "collision_evaluation": "not recomputed; original outcome is provenance only",
    }
    if replay:
        errors = {key: float(np.max(np.abs(trace[key] - trajectory[key])))
                  for key in ("pre_joint", "post_joint")}
        joint_matches = all(error <= joint_tolerance for error in errors.values())
        goal_matches = bool(done) == metadata["expected"]["goal_reached"]
        result.update(joint_max_abs_error=errors, joint_tolerance=joint_tolerance,
                      joint_trace_matches=joint_matches, goal_matches=goal_matches,
                      replay_matches=(state_hash_matches and controller_matches
                                      and joint_matches and goal_matches))
    return result, trace


class VideoRecorder:
    """Write both native camera views; import OpenCV only when recording."""

    def __init__(self, directory, fps=20):
        self.directory, self.fps, self.writers = directory, fps, {}

    def __call__(self, observation):
        import cv2
        for key, filename in (("agentview_image", "agentview.mp4"),
                              ("robot0_eye_in_hand_image", "wrist.mp4")):
            image = np.ascontiguousarray(observation[key][::-1, ::-1])
            if key not in self.writers:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(str(self.directory / filename),
                                         cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                         (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Could not open video encoder: {filename}")
                self.writers[key] = writer
            self.writers[key].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self):
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
