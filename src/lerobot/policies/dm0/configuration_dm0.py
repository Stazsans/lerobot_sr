from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DM0_IMAGE_SIZE = 728


@PreTrainedConfig.register_subclass("dm0")
@dataclass
class DM0Config(PreTrainedConfig):
    """Configuration for the DM0 Triton-accelerated inference policy.

    DM0 is a high-speed VLA with a separate action expert decoder architecture.
    Uses a CLIP-style ViT (728×728) + custom LLM + diffusion action decoder.
    Requires a checkpoint converted via ``convert_dm0_weight.py`` (produces a .pt file).

    Checkpoint format: flat dict saved with ``torch.save``, keys include
    ``decoder_*``, ``llm_*``, ``vision_*``.
    """

    # Path to the .pt Triton checkpoint (relative or absolute).
    # When loading via ``from_pretrained(directory)``, set to "checkpoint.pt" (the filename
    # inside that directory) or provide an absolute path here.
    checkpoint_path: str | None = None

    # Tokenizer for the LLM (same vocab used by the DM0 model, typically a Qwen-compatible tokenizer).
    # Required at inference time.
    tokenizer_path: str | None = None

    num_images: int = 3
    max_lang_len: int = 100
    chunk_size: int = 50
    n_action_steps: int = 50
    max_action_dim: int = 32

    image_resolution: tuple[int, int] = (DM0_IMAGE_SIZE, DM0_IMAGE_SIZE)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        for i in range(self.num_images):
            key = f"{OBS_IMAGES}.camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_resolution),
                )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self):
        raise NotImplementedError("DM0 is inference-only; training is not supported.")

    def get_scheduler_preset(self):
        raise NotImplementedError("DM0 is inference-only; training is not supported.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
