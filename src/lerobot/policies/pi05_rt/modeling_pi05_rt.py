"""Pi05 Triton-accelerated inference policy.

Wraps the ``Pi05Inference`` engine from ``realtime_vla/pi05_infer.py``.
Pi05 extends Pi0 with discrete state tokenization (Ada-RMSNorm conditioning).

Checkpoint setup
----------------
1. Convert a JAX Pi05 checkpoint::

    python convert_from_jax_pi05.py \\
        --jax_path /path/to/jax_ckpt \\
        --output checkpoint.pkl \\
        --prompt "your task" \\
        --tokenizer_path /path/to/paligemma-3b-pt-224

2. Create a pretrained directory and save the config, then run::

    lerobot-eval --policy.type=pi05_rt --policy.path=my_pi05_rt/ ...
"""

import logging
import os
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05_rt.configuration_pi05_rt import Pi05RTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


class Pi05RTPolicy(PreTrainedPolicy):
    """Pi05 Triton-accelerated inference policy.

    Supports 1–3 observation cameras.  Requires CUDA.
    When ``discrete_state_input=True`` (default), proprioceptive state values
    are first quantile-binned into discrete tokens by the Pi05Inference engine.
    """

    config_class = Pi05RTConfig
    name = "pi05_rt"

    def __init__(self, config: Pi05RTConfig, **kwargs):
        super().__init__(config)
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._engine = None
        self._action_queue: deque[Tensor] = deque()
        self._task_prompt: str = ""

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> "Pi05RTPolicy":
        model_id = str(pretrained_name_or_path)

        if config is None:
            config = Pi05RTConfig.from_pretrained(pretrained_name_or_path, **kwargs)

        if config.tokenizer_path is None:
            raise ValueError(
                "Pi05RT requires a PaliGemma tokenizer.  Set 'tokenizer_path' in the config to the "
                "local path of 'paligemma-3b-pt-224'."
            )

        policy = cls(config, **kwargs)

        ckpt = config.checkpoint_path or "checkpoint.pkl"
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(model_id, ckpt)

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Pi05RT Triton checkpoint not found at '{ckpt}'. "
                "Convert a JAX checkpoint with convert_from_jax_pi05.py first."
            )

        logger.info("Loading Pi05RT Triton checkpoint from %s", ckpt)
        with open(ckpt, "rb") as f:
            checkpoint = pickle.load(f)

        policy._init_engine(checkpoint)
        policy.eval()
        return policy

    def _init_engine(self, checkpoint: dict) -> None:
        from lerobot.policies.realtime_vla.pi05_infer import Pi05Inference

        image_keys = sorted(k for k in self.config.input_features if k.startswith(OBS_IMAGES))
        num_views = max(1, len(image_keys))

        # Determine state_dim for max_prompt_text length computation
        state_feature = self.config.input_features.get(OBS_STATE)
        state_dim = state_feature.shape[0] if state_feature is not None else self.config.max_state_dim

        self._engine = Pi05Inference(
            checkpoint=checkpoint,
            num_views=num_views,
            chunk_size=self.config.chunk_size,
            tokenizer_path=self.config.tokenizer_path,
            max_tokenize_len=self.config.tokenizer_max_length,
            discrete_state_input=self.config.discrete_state_input,
            max_prompt_text="",
            state_dim_for_max_prompt=state_dim,
        )
        logger.info(
            "Pi05Inference Triton engine ready (%d views, chunk_size=%d, discrete_state=%s)",
            num_views,
            self.config.chunk_size,
            self.config.discrete_state_input,
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return ``(1, n_action_steps, action_dim)``."""
        self.eval()
        if self._engine is None:
            raise RuntimeError("Pi05RT engine not initialised. Call from_pretrained() first.")

        device = self._placeholder.device
        if str(device) == "cpu":
            raise RuntimeError("Pi05RT requires a CUDA device.")

        # ---- images ----------------------------------------------------------
        image_keys = sorted(k for k in batch if k.startswith(OBS_IMAGES))
        B = batch[image_keys[0]].shape[0]
        if B != 1:
            raise ValueError(f"Pi05RT supports batch_size=1 only, got {B}.")

        imgs = []
        for k in image_keys:
            img = batch[k][0]
            res = self.config.image_resolution
            if img.shape[-2:] != res:
                img = F.interpolate(img.unsqueeze(0), size=res, mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img)
        images = torch.stack(imgs).permute(0, 2, 3, 1).to(torch.bfloat16).to(device)  # (V, H, W, C)

        # ---- noise -----------------------------------------------------------
        noise = torch.randn(self.config.chunk_size, 32, dtype=torch.bfloat16, device=device)

        # ---- task prompt -----------------------------------------------------
        task_prompt = ""
        if "task" in batch:
            raw = batch["task"]
            task_prompt = raw[0] if isinstance(raw, (list, tuple)) else str(raw)

        # ---- state tokens (discrete) or continuous state ---------------------
        state_tokens = None
        if self.config.discrete_state_input and OBS_STATE in batch:
            # Pi05Inference quantile-bins the raw state values → discrete int tokens.
            # It expects a 1-D numpy array of raw (unnormalised) state values.
            state_np = batch[OBS_STATE][0].float().cpu().numpy()  # (state_dim,)
            # Pad to max_state_dim if needed
            pad = self.config.max_state_dim - state_np.shape[0]
            if pad > 0:
                state_np = np.pad(state_np, (0, pad))
            state_tokens = state_np.astype(np.float32)

        # ---- forward ---------------------------------------------------------
        actions = self._engine.forward(
            images,
            noise,
            task_prompt,
            state_tokens,
        )  # (chunk_size, 32)

        action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[: self.config.n_action_steps, :action_dim]

        return actions.unsqueeze(0).float()  # (1, n_action_steps, action_dim)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "Pi05RT is inference-only. Use PI05Policy (--policy.type=pi05) for training."
        )
