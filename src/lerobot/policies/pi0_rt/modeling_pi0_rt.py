"""Pi0 Triton-accelerated inference policy.

Wraps the ``Pi0Inference`` engine from ``realtime_vla/pi0_infer.py`` as a
lerobot ``PreTrainedPolicy``.  This policy is **inference-only** — training is
not supported.  Use the full ``PI0Policy`` (``--policy.type=pi0``) for training.

Checkpoint setup
----------------
1. Convert a JAX Pi0 checkpoint::

    python convert_from_jax.py \\
        --jax_path /path/to/jax_ckpt \\
        --output checkpoint.pkl \\
        --prompt "your task" \\
        --tokenizer_path /path/to/paligemma-3b-pt-224

2. Create a pretrained directory::

    mkdir my_pi0_rt/
    cp checkpoint.pkl my_pi0_rt/
    # save Pi0RTConfig as policy_config.json:
    python -c "
    from lerobot.policies.pi0_rt.configuration_pi0_rt import Pi0RTConfig
    cfg = Pi0RTConfig(checkpoint_path='checkpoint.pkl', ...)
    cfg.save_pretrained('my_pi0_rt/')
    "

3. Run with lerobot::

    lerobot-eval --policy.type=pi0_rt --policy.path=my_pi0_rt/ ...
"""

import logging
import os
import pickle
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi0_rt.configuration_pi0_rt import Pi0RTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


class Pi0RTPolicy(PreTrainedPolicy):
    """Pi0 Triton-accelerated inference policy.

    Supports 1–3 observation cameras.  Requires a CUDA device.
    The ``Pi0Inference`` engine uses CUDA graphs; first call compiles the graph
    (takes ~5 s), subsequent calls are fast.
    """

    config_class = Pi0RTConfig
    name = "pi0_rt"

    def __init__(self, config: Pi0RTConfig, **kwargs):
        super().__init__(config)
        # Placeholder parameter so PyTorch treats this as a proper nn.Module with a device.
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._engine = None
        self._action_queue: deque[Tensor] = deque()

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
    ) -> "Pi0RTPolicy":
        """Load a Pi0RT policy from a directory containing ``policy_config.json``
        and the ``.pkl`` Triton checkpoint."""
        model_id = str(pretrained_name_or_path)

        if config is None:
            config = Pi0RTConfig.from_pretrained(pretrained_name_or_path, **kwargs)

        policy = cls(config, **kwargs)

        # Resolve checkpoint path
        ckpt = config.checkpoint_path or "checkpoint.pkl"
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(model_id, ckpt)

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Pi0RT Triton checkpoint not found at '{ckpt}'. "
                "Convert a JAX checkpoint with convert_from_jax.py first."
            )

        logger.info("Loading Pi0RT Triton checkpoint from %s", ckpt)
        with open(ckpt, "rb") as f:
            checkpoint = pickle.load(f)

        policy._init_engine(checkpoint)
        policy.eval()
        return policy

    def _init_engine(self, checkpoint: dict) -> None:
        """Initialise the Pi0Inference Triton engine (allocates CUDA buffers + compiles graph)."""
        from lerobot.policies.realtime_vla.pi0_infer import Pi0Inference

        image_keys = sorted(k for k in self.config.input_features if k.startswith(OBS_IMAGES))
        num_views = max(1, len(image_keys))
        self._engine = Pi0Inference(checkpoint, num_views, self.config.chunk_size)
        logger.info("Pi0Inference Triton engine ready (%d views, chunk_size=%d)", num_views, self.config.chunk_size)

    def _save_pretrained(self, save_directory: Path) -> None:
        # Only the config is saved; the Triton .pkl is managed externally.
        self.config._save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a full action chunk of shape ``(B, n_action_steps, action_dim)``.

        Only ``B=1`` is supported at inference time (Triton engines process a
        single observation).
        """
        self.eval()
        if self._engine is None:
            raise RuntimeError("Pi0RT engine not initialised. Call from_pretrained() first.")

        device = self._placeholder.device
        if str(device) == "cpu":
            raise RuntimeError("Pi0RT requires a CUDA device. Run with --policy.device=cuda.")

        # ---- gather images ---------------------------------------------------
        image_keys = sorted(k for k in batch if k.startswith(OBS_IMAGES))
        B = batch[image_keys[0]].shape[0]
        if B != 1:
            raise ValueError(f"Pi0RT supports batch_size=1 only, got {B}.")

        imgs = []
        for k in image_keys:
            img = batch[k][0]  # (C, H, W) float32
            H, W = img.shape[-2:]
            res = self.config.image_resolution
            if (H, W) != res:
                img = F.interpolate(img.unsqueeze(0), size=res, mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img)

        # Pi0Inference expects (num_views, H, W, C) bfloat16  [NHWC]
        images = torch.stack(imgs).permute(0, 2, 3, 1).to(torch.bfloat16).to(device)  # (V, H, W, C)

        # ---- state -----------------------------------------------------------
        state = batch[OBS_STATE][0].to(torch.bfloat16).to(device)  # (state_dim,)
        pad = self.config.max_state_dim - state.shape[0]
        if pad > 0:
            state = F.pad(state, (0, pad))

        # ---- noise -----------------------------------------------------------
        noise = torch.randn(self.config.chunk_size, 32, dtype=torch.bfloat16, device=device)

        # ---- forward ---------------------------------------------------------
        actions = self._engine.forward(images, state, noise)  # (chunk_size, 32)

        # Crop to valid action dim and n_action_steps
        action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[: self.config.n_action_steps, :action_dim]  # (n_action_steps, action_dim)

        return actions.unsqueeze(0).float()  # (1, n_action_steps, action_dim)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return a single action ``(B, action_dim)`` using a cached action queue."""
        self.eval()
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch, **kwargs)  # (1, n_action_steps, action_dim)
            self._action_queue.extend(chunk.transpose(0, 1))    # enqueue (n_action_steps,) tensors of shape (1, action_dim)
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError(
            "Pi0RT is inference-only. Use PI0Policy (--policy.type=pi0) for training."
        )
