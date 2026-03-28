#!/usr/bin/env python

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE

IMAGE_KEY = "observation.images.cam0"


def make_env_only_act_policy() -> ACTPolicy:
    config = ACTConfig(
        device="cpu",
        chunk_size=3,
        n_action_steps=1,
        input_features={
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )
    return ACTPolicy(config)


def make_image_only_act_policy() -> ACTPolicy:
    config = ACTConfig(
        device="cpu",
        chunk_size=2,
        n_action_steps=1,
        pretrained_backbone_weights=None,
        input_features={
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        },
    )
    return ACTPolicy(config)


def test_act_forward_without_observation_state():
    policy = make_env_only_act_policy()

    batch = {
        OBS_ENV_STATE: torch.randn(2, 4),
        ACTION: torch.randn(2, policy.config.chunk_size, 2),
        "action_is_pad": torch.zeros(2, policy.config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)

    assert loss.ndim == 0
    assert "l1_loss" in loss_dict
    assert "kld_loss" in loss_dict


def test_act_predict_action_chunk_without_observation_state():
    policy = make_env_only_act_policy()

    batch = {
        OBS_ENV_STATE: torch.randn(2, 4),
    }

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (2, policy.config.chunk_size, 2)


def test_act_forward_without_observation_state_with_images():
    policy = make_image_only_act_policy()

    batch = {
        IMAGE_KEY: torch.randn(1, 3, 32, 32),
        ACTION: torch.randn(1, policy.config.chunk_size, 2),
        "action_is_pad": torch.zeros(1, policy.config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)

    assert loss.ndim == 0
    assert "l1_loss" in loss_dict
    assert "kld_loss" in loss_dict


def test_act_predict_action_chunk_without_observation_state_with_images():
    policy = make_image_only_act_policy()

    batch = {
        IMAGE_KEY: torch.randn(1, 3, 32, 32),
    }

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (1, policy.config.chunk_size, 2)
