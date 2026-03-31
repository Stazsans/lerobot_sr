from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

PI0_RT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0_rt")
@dataclass
class Pi0RTConfig(PreTrainedConfig):
    """Configuration for Pi0 Triton-accelerated inference policy.

    Wraps the ``Pi0Inference`` Triton engine from realtime-vla.
    Achieves ~20–37ms latency per step (1–3 views) on RTX 4090.

    Checkpoint format: flat dict stored as ``.pkl`` (converted from JAX via
    ``convert_from_jax.py``).  Keys follow the pattern
    ``{vision|encoder|decoder}_{sublayer}_{tensor_type}``.
    """

    # Path to the .pkl Triton checkpoint.  When using from_pretrained(directory),
    # defaults to "checkpoint.pkl" inside that directory.
    checkpoint_path: str | None = None

    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32

    image_resolution: tuple[int, int] = (PI0_RT_IMAGE_SIZE, PI0_RT_IMAGE_SIZE)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
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
        raise NotImplementedError("Pi0RT is inference-only; training is not supported.")

    def get_scheduler_preset(self):
        raise NotImplementedError("Pi0RT is inference-only; training is not supported.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
