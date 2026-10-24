"""Optional new-inference adapter for an installed public OpenVLA-OFT runtime.

Uses the public project's model classes and preprocessing functions. Checkpoint
files are read only; unlike upstream get_vla(), this loader does not replace
config.json or Python files. This is not the unavailable original mixed500
deployment, and identical actions from that deployment are not guaranteed.
"""

from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def component_path(checkpoint, prefix):
    candidates = sorted(Path(checkpoint).glob(prefix + "*checkpoint.pt"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {prefix} checkpoint, found {len(candidates)}")
    return candidates[0]


class OpenVLAPolicy:
    def __init__(self, checkpoint, unnorm_key):
        import torch
        from transformers import AutoTokenizer
        from experiments.robot.libero.run_libero_eval import prepare_observation, process_action
        from experiments.robot.openvla_utils import get_vla_action
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import (
            ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK, ACTION_PROPRIO_NORMALIZATION_TYPE,
        )

        checkpoint = Path(checkpoint).resolve(strict=True)
        if not torch.cuda.is_available():
            raise RuntimeError("New OpenVLA inference requires a compatible CUDA environment")
        if (ACTION_DIM, PROPRIO_DIM, NUM_ACTIONS_CHUNK) != (7, 8, 8):
            raise ValueError("The installed OFT runtime must use LIBERO constants (7, 8, 8)")
        if ACTION_PROPRIO_NORMALIZATION_TYPE.value != "bounds_q99":
            raise ValueError("This adapter supports the public LIBERO BOUNDS_Q99 runtime")
        self.torch = torch
        self.prepare_observation, self.process_action = prepare_observation, process_action
        self.predict = get_vla_action
        stats = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
        if unnorm_key not in stats:
            raise ValueError(f"Unknown normalization key. Available: {list(stats)}")
        for component in ("action", "proprio"):
            for key in ("q01", "q99"):
                if key not in stats[unnorm_key][component]:
                    raise ValueError(f"Missing {component}.{key} normalization statistics")
        config = OpenVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        self.model = OpenVLAForActionPrediction.from_pretrained(
            checkpoint, config=config, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to("cuda").eval()
        self.model.vision_backbone.set_num_images_in_input(2)
        self.model.norm_stats = stats
        # Instantiate the known public classes directly. AutoProcessor's custom
        # auto_map may otherwise request remote code or registration in config.
        self.processor = PrismaticProcessor(
            image_processor=PrismaticImageProcessor.from_pretrained(checkpoint, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                                                    trust_remote_code=False),
        )
        width = self.model.llm_dim
        self.head = L1RegressionActionHead(input_dim=width, hidden_dim=width, action_dim=7)
        self.proprio = ProprioProjector(llm_dim=width, proprio_dim=8)
        for module, prefix in ((self.head, "action_head"), (self.proprio, "proprio_projector")):
            weights = torch.load(component_path(checkpoint, prefix),
                                 map_location="cpu", weights_only=True)
            weights = {key.removeprefix("module."): value for key, value in weights.items()}
            module.load_state_dict(weights, strict=True)
            module.to(device="cuda", dtype=torch.bfloat16).eval()
        self.config = SimpleNamespace(num_images_in_input=2, use_proprio=True,
                                      center_crop=True, unnorm_key=unnorm_key,
                                      use_film=False, use_l1_regression=True, use_diffusion=False)
        self.queue = deque()

    def reset(self, instruction, seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        self.instruction = instruction
        self.queue.clear()

    def get_action(self, raw_observation):
        if not self.queue:
            observation, _ = self.prepare_observation(raw_observation, 224)
            with self.torch.inference_mode():
                chunk = np.asarray(self.predict(
                    self.config, self.model, self.processor, observation, self.instruction,
                    action_head=self.head, proprio_projector=self.proprio, use_film=False,
                ))
            if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid predicted action chunk: {chunk.shape}")
            self.queue.extend(chunk)
        return np.asarray(self.process_action(self.queue.popleft().copy(), "openvla"),
                          dtype=np.float32)
