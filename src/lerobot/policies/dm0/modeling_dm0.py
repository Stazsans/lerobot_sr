"""DM0 Triton-accelerated inference policy.

Wraps the ``DM0Inference`` engine from ``realtime_vla/dm0_infer.py``.

DM0 is a high-speed VLA model that uses:
- A CLIP-style ViT vision encoder (728×728, 23 transformer layers)
- A 28-layer LLM (hidden dim 2048) with a shared vocabulary tokeniser
- A separate 28-layer action decoder (hidden dim 1024)
- Diffusion-based action generation (10 steps, chunk_size=50)

Key difference from Pi0/Pi05: **no proprioceptive state input** — actions are
conditioned only on images and language.

Checkpoint setup
----------------
1. Obtain the DM0 HuggingFace model and ``modeling_dm0_init.py`` (required by
   the conversion script)::

    python convert_dm0_weight.py \\
        --model_path /path/to/dm0_hf_model \\
        --output checkpoint.pt

2. Create a pretrained directory::

    mkdir my_dm0/
    cp checkpoint.pt my_dm0/
    python -c "
    from lerobot.policies.dm0.configuration_dm0 import DM0Config
    cfg = DM0Config(tokenizer_path='/path/to/dm0_tokenizer')
    cfg.save_pretrained('my_dm0/')
    "

3. Run::

    lerobot-eval --policy.type=dm0 --policy.path=my_dm0/ ...
"""

import logging
import os
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.dm0.configuration_dm0 import DM0Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.pt"


class DM0Policy(PreTrainedPolicy):
    """DM0 Triton-accelerated inference policy.

    Supports 1–3 observation cameras at 728×728 resolution.
    Requires a CUDA device (CUDA graphs are used for fast repeated inference).
    First forward call compiles the CUDA graph (~5 s); subsequent calls are fast.

    This policy is **inference-only** — training is not supported.
    """

    config_class = DM0Config
    name = "dm0"

    def __init__(self, config: DM0Config, **kwargs):
        super().__init__(config)
        self._placeholder = nn.Parameter(torch.zeros(1), requires_grad=False)
        self._engine = None
        self._tokenizer = None
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
    ) -> "DM0Policy":
        model_id = str(pretrained_name_or_path)

        if config is None:
            config = DM0Config.from_pretrained(pretrained_name_or_path, **kwargs)

        policy = cls(config, **kwargs)

        # Resolve checkpoint path
        ckpt = config.checkpoint_path or _CHECKPOINT_FILENAME
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(model_id, ckpt)

        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"DM0 Triton checkpoint not found at '{ckpt}'. "
                "Convert a HuggingFace DM0 checkpoint with convert_dm0_weight.py first."
            )

        logger.info("Loading DM0 Triton checkpoint from %s", ckpt)
        device = config.device or "cuda"
        checkpoint = torch.load(ckpt, map_location=device, weights_only=True)

        policy._init_engine(checkpoint, device=device)
        policy._init_tokenizer()
        policy.eval()
        return policy

    def _init_engine(self, checkpoint: dict, device: str = "cuda") -> None:
        from lerobot.policies.realtime_vla.dm0_infer import DM0Inference

        self._engine = DM0Inference(
            checkpoint=checkpoint,
            num_images=self.config.num_images,
            max_lang_len=self.config.max_lang_len,
            device=device,
        )
        logger.info(
            "DM0Inference Triton engine ready (%d images, max_lang_len=%d, chunk_size=%d)",
            self.config.num_images,
            self.config.max_lang_len,
            self.config.chunk_size,
        )

    def _init_tokenizer(self) -> None:
        if self.config.tokenizer_path is None:
            logger.warning(
                "DM0: tokenizer_path not set.  Language conditioning will use empty input_ids. "
                "Set 'tokenizer_path' in the config to enable language conditioning."
            )
            return
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            logger.info("DM0 tokenizer loaded from %s", self.config.tokenizer_path)
        except Exception as exc:
            logger.warning("DM0: failed to load tokenizer (%s). Language conditioning disabled.", exc)

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def _tokenize_task(self, task_text: str, device: torch.device | str) -> Tensor:
        """Return ``input_ids`` tensor of shape ``(max_lang_len,)`` on *device*."""
        max_len = self.config.max_lang_len
        if self._tokenizer is None:
            return torch.zeros(max_len, dtype=torch.long, device=device)

        encoding = self._tokenizer(
            task_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoding["input_ids"][0].to(device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return ``(1, n_action_steps, action_dim)``."""
        self.eval()
        if self._engine is None:
            raise RuntimeError("DM0 engine not initialised. Call from_pretrained() first.")

        device = self._placeholder.device
        if str(device) == "cpu":
            raise RuntimeError("DM0 requires a CUDA device.")

        # ---- images ----------------------------------------------------------
        image_keys = sorted(k for k in batch if k.startswith(OBS_IMAGES))
        B = batch[image_keys[0]].shape[0]
        if B != 1:
            raise ValueError(f"DM0 supports batch_size=1 only, got {B}.")

        imgs = []
        res = self.config.image_resolution  # (728, 728)
        for k in image_keys:
            img = batch[k][0]  # (C, H, W)
            if img.shape[-2:] != res:
                img = F.interpolate(img.unsqueeze(0), size=res, mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img)

        # Pad to num_images with zero frames if fewer cameras are present
        while len(imgs) < self.config.num_images:
            imgs.append(torch.zeros_like(imgs[0]))

        # DM0Inference expects (num_images, 3, H, W) bfloat16  [NCHW]
        images = torch.stack(imgs[: self.config.num_images]).to(torch.bfloat16).to(device)

        # ---- language --------------------------------------------------------
        task_text = ""
        if "task" in batch:
            raw = batch["task"]
            task_text = raw[0] if isinstance(raw, (list, tuple)) else str(raw)

        input_ids = self._tokenize_task(task_text, device=device)  # (max_lang_len,)

        # ---- noise -----------------------------------------------------------
        noise = torch.randn(self.config.chunk_size, 32, dtype=torch.bfloat16, device=device)

        # ---- forward ---------------------------------------------------------
        actions = self._engine.forward(images, input_ids, noise)  # (chunk_size, 32)

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
            "DM0 is inference-only. Training is not supported for this policy."
        )
