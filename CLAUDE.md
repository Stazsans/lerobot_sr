# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a fork of [HuggingFace LeRobot v0.4.4](https://github.com/huggingface/lerobot) — a PyTorch-based robotics ML library. Package name is `lerobot`, installed in `src/` layout. The fork adds several custom extensions on top of upstream.

## Commands

**Package manager:** `uv` is preferred.

```bash
# Install (development)
pip install -e ".[dev,test]"
# Or with specific policy/hardware extras:
pip install -e ".[dev,test,smolvla,dynamixel]"

# Lint
ruff check src/
pre-commit run --all-files

# Type check (selective modules only)
mypy src/lerobot/configs/ src/lerobot/envs/ src/lerobot/transport/

# Run all unit tests
uv run pytest tests -vv

# Run a single test
uv run pytest tests/test_available.py::test_some_function -vv

# End-to-end training tests
make test-act-ete-train
make test-end-to-end          # All policies (act, diffusion, tdmpc, smolvla)

# Training
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Resume training
lerobot-train --config_path=outputs/train/.../pretrained_model/train_config.json --resume=true

# Evaluation
lerobot-eval --policy.path=outputs/train/.../pretrained_model --env.type=aloha

# Docker
make build-user      # docker/Dockerfile.user
make build-internal  # docker/Dockerfile.internal
```

Git LFS is required for test artifacts: `git lfs install && git lfs pull`.

## Architecture

### Configuration System (`src/lerobot/configs/`)

All configs use `draccus` (dataclass-based CLI parsing). The key pattern:

- `TrainPipelineConfig` (`configs/train.py`) — top-level training config; dispatches `--policy.type=act` to the right policy config via `draccus.ChoiceRegistry`.
- `PreTrainedConfig` (`configs/policies.py`) — base for all policy configs; each policy registers itself with `@draccus.ChoiceRegistry`.
- CLI overrides work directly on nested fields: `--policy.dim_model=64`, `--dataset.episodes="[0]"`.
- Configs serialize to `train_config.json` and can be loaded from HF Hub.

### Policies (`src/lerobot/policies/`)

Each policy is a directory with three files:
- `configuration_<name>.py` — `@dataclass` config registered in the ChoiceRegistry
- `modeling_<name>.py` — `PreTrainedPolicy(nn.Module, HubMixin)` with `config_class` and `name` attrs
- `processor_<name>.py` — optional custom data processor

`factory.py` lazily loads policy classes by name. When adding a new policy, register it in `factory.py` and `__init__.py`.

**Available policies:** `act`, `diffusion`, `tdmpc`, `vqbet`, `smolvla`, `groot`, `pi0`, `pi05`, `pi0_fast`, `sarm`, `wall_x`, `xvla`, `sac`, `rtc`

### Dataset System (`src/lerobot/datasets/`)

LeRobotDataset v3.0 format: Parquet files for metadata/actions + MP4 videos (or raw images). `lerobot_dataset.py` is the main class. Datasets are hosted on HF Hub and referenced by `repo_id`.

### Processor Pipeline (`src/lerobot/processor/`)

`PolicyProcessorPipeline` (`pipeline.py`, ~73KB — the core data flow component) chains:
normalization → observation processing → batch conversion → delta action computation

### Robots & Hardware (`src/lerobot/robots/`, `cameras/`, `motors/`)

Abstract base in `robot.py`. 12+ robot implementations. Motor backends: `dynamixel`, `feetech`, `damiao`. Camera backends: `opencv`, `realsense`, `zmq`, `reachy2_camera`.

---

## Fork-Specific Extensions

### realtime-vla Integration (`src/lerobot/policies/realtime_vla/`, `pi0_rt/`, `pi05_rt/`, `dm0/`)

Three inference-only policies wrapping Triton GPU kernels from [realtime-vla](https://github.com/yunchaoyang/realtime-vla).
All three require CUDA (CUDA graphs are used; first call compiles ~5 s).

| Policy type | Model | Latency (RTX 4090, 2 views) | Checkpoint format |
|---|---|---|---|
| `pi0_rt` | Pi0 Triton | ~27 ms | `.pkl` (convert with `convert_from_jax.py`) |
| `pi05_rt` | Pi05 Triton | ~29 ms | `.pkl` (convert with `convert_from_jax_pi05.py`) |
| `dm0` | DM0 Triton | ~56 ms (3 views) | `.pt` (convert with `convert_dm0_weight.py`) |

These are **inference-only** — use `pi0`/`pi05` for training.
Checkpoint conversion scripts are in each policy directory alongside `configuration_*.py`.
Shared Triton kernels live in `src/lerobot/policies/realtime_vla/`.

**DM0** is a new model not in upstream lerobot: CLIP-style ViT (728×728, 23 layers) + 28-layer LLM + separate 28-layer action decoder. No proprioceptive state input; language-conditioned only.

### RTC — Real-Time Chunking (`src/lerobot/policies/rtc/`)

Adds streaming/chunked action execution for flow-matching policies (pi0, pi05, smolvla). Files: `modeling_rtc.py`, `configuration_rtc.py`, `action_queue.py`, `latency_tracker.py`, `training_time.py`.

### SARM — Stage-Aware Reward Modeling (`src/lerobot/policies/sarm/`)

A policy that models task progress as a reward signal. Used to generate `sarm_progress.parquet` files that feed into RA-BC training.

### RA-BC — Reward-Aligned Behavior Cloning (`src/lerobot/configs/train.py`)

`TrainPipelineConfig` fields:
- `use_rabc: bool` — enables reward-weighted training
- `rabc_progress_path: str` — path to SARM-generated progress parquet (auto-detected if not set)
- `rabc_kappa`, `rabc_epsilon`, `rabc_head_mode` — tuning parameters

### Async Inference (`src/lerobot/async_inference/`)

Client/server architecture for decoupled policy inference. `policy_server.py` runs the model; `robot_client.py` sends observations and receives compressed/streamed actions.

### SO101 End-Effector Control (`so101_to_so101_EE/`)

Workflow for end-effector pose control on SO-101 arms, separate from the main package.

---

## Dependency Groups (pyproject.toml)

Several extras conflict — do not install together:
- `wallx` conflicts with: `smolvla`, `groot`, `xvla`, `sarm`, `hilserl`, `libero`, `peft` (pins `transformers==4.49.0`)
- `pi` conflicts with same set (uses a custom transformers branch)

Hardware extras: `dynamixel`, `feetech`, `gamepad`, `hopejr`, `lekiwi`, `reachy2`, `unitree_g1`  
Policy extras: `wallx`, `pi`, `smolvla`, `groot`, `xvla`, `hilserl`, `sarm`  
Feature extras: `async`, `peft`  
Sim extras: `aloha`, `pusht`, `libero`, `metaworld`

## Code Style

- Line length: 110 chars (ruff)
- Python 3.10+ (use `X | Y` union syntax, `match` statements)
- No `print()` in library code (use `logging`)
- Ruff rules enforced: E, W, F, I, B, C4, T20, N, UP, SIM
- Bandit skips: B101, B311, B404, B603, B615
