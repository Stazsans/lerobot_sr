from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

PI05_RT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_rt")
@dataclass
class Pi05RTConfig(PreTrainedConfig):
    """Configuration for Pi05 Triton-accelerated inference policy.

    Wraps the ``Pi05Inference`` Triton engine from realtime-vla.
    Pi05 adds discrete state tokenization (Ada-RMSNorm) on top of Pi0.
    Achieves ~22–39ms latency per step (1–3 views) on RTX 4090.

    Checkpoint format: flat dict stored as ``.pkl`` (converted from JAX via
    ``convert_from_jax_pi05.py``).  Requires ``embedding_weight`` key when
    ``discrete_state_input=True``.
    """

    # Path to the .pkl Triton checkpoint.  When using from_pretrained(directory),
    # defaults to "checkpoint.pkl" inside that directory.
    checkpoint_path: str | None = None

    # Path to PaliGemma-3b-pt-224 tokenizer directory (required at inference).
    tokenizer_path: str | None = None

    # Use discrete (quantile-binned) state tokens (recommended; matches test.py).
    discrete_state_input: bool = True

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    tokenizer_max_length: int = 200

    image_resolution: tuple[int, int] = (PI05_RT_IMAGE_SIZE, PI05_RT_IMAGE_SIZE)
    empty_cameras: int = 0

    # Pi05 uses quantile normalization (same as full PI05Config).
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_resolution),
                )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self):
        raise NotImplementedError("Pi05RT is inference-only; training is not supported.")

    def get_scheduler_preset(self):
        raise NotImplementedError("Pi05RT is inference-only; training is not supported.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
