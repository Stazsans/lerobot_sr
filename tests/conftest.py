#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import traceback
from importlib.util import find_spec
import os
from pathlib import Path

import pytest
from serial import SerialException

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from tests.utils import DEVICE

_OPTIONAL_PYTEST_PLUGINS = [
    "tests.fixtures.dataset_factories",
    "tests.fixtures.files",
    "tests.fixtures.hub",
    "tests.fixtures.optimizers",
]


def _plugin_exists(plugin: str) -> bool:
    try:
        return find_spec(plugin) is not None
    except ModuleNotFoundError:
        return False


# Some downstream/cut-down checkouts do not include the full optional test fixture tree.
pytest_plugins = [plugin for plugin in _OPTIONAL_PYTEST_PLUGINS if _plugin_exists(plugin)]

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "tests"
HF_TEST_HOME = TEST_TMP_ROOT / "hf_home"
HF_DATASETS_CACHE = HF_TEST_HOME / "datasets"

os.environ.setdefault("HF_HOME", str(HF_TEST_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))


def pytest_configure(config):
    HF_TEST_HOME.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)

    try:
        import datasets.config as datasets_config

        datasets_config.HF_CACHE_HOME = str(HF_TEST_HOME)
        datasets_config.HF_DATASETS_CACHE = str(HF_DATASETS_CACHE)
    except Exception:
        pass


def pytest_collection_finish():
    print(f"\nTesting with {DEVICE=}")


def _check_component_availability(component_type, available_components, make_component):
    """Generic helper to check if a hardware component is available"""
    if component_type not in available_components:
        raise ValueError(
            f"The {component_type} type is not valid. Expected one of these '{available_components}'"
        )

    try:
        component = make_component(component_type)
        component.connect()
        del component
        return True

    except Exception as e:
        print(f"\nA {component_type} is not available.")

        if isinstance(e, ModuleNotFoundError):
            print(f"\nInstall module '{e.name}'")
        elif isinstance(e, SerialException):
            print("\nNo physical device detected.")
        elif isinstance(e, ValueError) and "camera_index" in str(e):
            print("\nNo physical camera detected.")
        else:
            traceback.print_exc()

        return False


@pytest.fixture
def patch_builtins_input(monkeypatch):
    def print_text(text=None):
        if text is not None:
            print(text)

    monkeypatch.setattr("builtins.input", print_text)


@pytest.fixture
def policy_feature_factory():
    """PolicyFeature factory"""

    def _pf(ft: FeatureType, shape: tuple[int, ...]) -> PolicyFeature:
        return PolicyFeature(type=ft, shape=shape)

    return _pf


def assert_contract_is_typed(features: dict[PipelineFeatureType, dict[str, PolicyFeature]]) -> None:
    assert isinstance(features, dict)
    assert all(isinstance(k, PipelineFeatureType) for k in features)
    assert all(isinstance(v, dict) for v in features.values())
    assert all(all(isinstance(nk, str) for nk in v) for v in features.values())
    assert all(all(isinstance(nv, PolicyFeature) for nv in v.values()) for v in features.values())
