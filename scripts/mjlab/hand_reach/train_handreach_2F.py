#!/usr/bin/env python3
"""
Train a two-fingertip HandReach policy with mjlab + RSL-RL.

Run from the repository root:

    python scripts/mjlab/hand_reach/train_handreach_2F.py \
        --policy-path my_handreach_2F_policy.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from register_handreach_2F import (
    FINGERTIP_POS_DIM,
    FINGERTIP_SITE_NAMES,
    NUM_FINGERTIPS,
    REACH_THD,
    TASK_ID,
    register_hand_reach_task,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

ITERATIONS = 2000
NUM_ENVS = 512
NUM_STEPS_PER_ENV = 64

NUM_EVALS = 100
EVAL_NUM_ENVS = 64
EVAL_STEPS = 200

LEARNING_RATE = 1e-4
MIN_ADAPTIVE_LR = 1e-6
MAX_ADAPTIVE_LR = 3e-4

SCHEDULE = "adaptive"
NUM_LEARNING_EPOCHS = 10
NUM_MINI_BATCHES = 4

CLIP_PARAM = 0.2
DESIRED_KL = 0.01
MAX_GRAD_NORM = 1.0

ENTROPY_COEF = 0.001

# This is only needed when using a fixed schedule and collecting KL values.
LOG_KL_WITH_FIXED_SCHEDULE = False


# =============================================================================
# TARGET CONFIGURATION
# =============================================================================

# Establish the fixed-target baseline before enabling randomization.
RANDOMIZE_TARGET = True

# Observation order:
#     THtip xyz, IFtip xyz
FINGER_NAMES = FINGERTIP_SITE_NAMES

NOMINAL_FINGERTIP_TARGETS = (
    (-0.174, -0.522, 1.482),  # THtip
    (-0.160, -0.574, 1.463),  # IFtip
)

# Used only when RANDOMIZE_TARGET is True.
SHARED_TARGET_OFFSET_LOW = (-0.01, -0.01, -0.01)
SHARED_TARGET_OFFSET_HIGH = (0.01, 0.01, 0.01)

PER_FINGER_TARGET_JITTER_LOW = (-0.03, -0.03, -0.03)
PER_FINGER_TARGET_JITTER_HIGH = (0.03, 0.01, 0.03)


# =============================================================================
# OBSERVATION LAYOUT
# =============================================================================

# qpos        0:23    23 values
# qvel       23:46    23 values
# tip_pos    46:52     6 values
# target_pos 52:58     6 values
# target_err 58:64     6 values
# act        64:103   39 values

QPOS_SLICE = slice(0, 23)
QVEL_SLICE = slice(23, 46)
TIP_POS_SLICE = slice(46, 52)
TARGET_POS_SLICE = slice(52, 58)
TARGET_ERR_SLICE = slice(58, 64)
ACT_SLICE = slice(64, 103)

OBS_DIM = 103
ACTION_DIM = 39


# =============================================================================
# REWARD CONFIGURATION
# =============================================================================

# reward =
#     -DISTANCE_REWARD_SCALE * sum(fingertip distances)
#     +SOLVED_BONUS_VALUE * number of solved fingertips
#     -action penalty
#
# Robust and velocity penalties are retained as configurable terms but disabled.

DISTANCE_REWARD_SCALE = 50.0

ROBUST_MAX_DISTANCE_PENALTY_SCALE = 0.0
ROBUST_MAX_DISTANCE_TH = 0.03

VELOCITY_PENALTY_SCALE = 0.0
ACTION_PENALTY_SCALE = 0.001

SOLVED_DISTANCE_TH = 0.03
VERY_SOLVED_DISTANCE_TH = SOLVED_DISTANCE_TH * 0.5

SOLVED_BONUS_VALUE = 5.0


# =============================================================================
# OUTPUT REDIRECTION
# =============================================================================

@contextlib.contextmanager
def redirect_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            with contextlib.redirect_stderr(file):
                yield


def import_mjlab_tools(quiet: bool):
    with redirect_output(quiet):
        import myosuite  # noqa: F401
        import mjlab.tasks.registry as registry

        from myosuite.core.registry import make_env

        register_hand_reach_task(quiet=True)

    return make_env, registry.load_rl_cfg


# =============================================================================
# MJLAB TO RSL-RL WRAPPER
# =============================================================================

class HandReachRslRlWrapper:
    """
    Adapter between the mjlab HandReach environment and RSL-RL.

    The wrapper:
        1. Creates two targets per environment.
        2. Inserts target positions into the observation.
        3. Calculates target errors.
        4. Calculates the two-fingertip reward.
        5. Converts the mjlab step result to the RSL-RL format.
    """

    def __init__(
        self,
        env: Any,
        seed: int,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(
            getattr(env, "num_envs", 1)
        )
        self.num_actions = int(
            env.action_space.shape[-1]
        )
        self.cfg = getattr(
            env,
            "cfg",
            None,
        )

        if self.num_actions != ACTION_DIM:
            raise RuntimeError(
                f"Expected {ACTION_DIM} actions, "
                f"but the environment exposes "
                f"{self.num_actions}."
            )

        self.target_rng = torch.Generator(
            device="cpu"
        )
        self.target_rng.manual_seed(seed)

        self.target_positions = torch.zeros(
            (
                self.num_envs,
                NUM_FINGERTIPS,
                3,
            ),
            device=self.device,
            dtype=torch.float32,
        )

    def __getattr__(
        self,
        name: str,
    ):
        return getattr(
            self.env,
            name,
        )

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(
        self,
        value,
    ):
        self.env.episode_length_buf = value

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    def close(self) -> None:
        self.env.close()

    def reset(
        self,
        seed: int | None = None,
    ) -> TensorDict:
        if seed is not None:
            self.target_rng.manual_seed(
                int(seed) + 12_345
            )

            try:
                self.env.reset(seed=seed)
            except TypeError:
                self.env.reset()
        else:
            self.env.reset()

        reset_mask = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

        self._resample_targets(reset_mask)

        obs = self.get_observations()

        self._validate_observation(
            obs["policy"]
        )

        return obs

    def get_observations(self) -> TensorDict:
        obs_dict = (
            self.env.observation_manager.compute()
        )

        obs_dict = (
            self._replace_target_observation(
                obs_dict
            )
        )

        obs = TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
        )

        self._validate_observation(
            obs["policy"]
        )

        return obs

    def step(
        self,
        actions: torch.Tensor,
    ):
        actions = actions.to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        expected_action_shape = (
            self.num_envs,
            ACTION_DIM,
        )

        if tuple(actions.shape) != expected_action_shape:
            raise RuntimeError(
                f"Expected action shape "
                f"{expected_action_shape}, "
                f"got {tuple(actions.shape)}."
            )

        (
            obs_dict,
            _native_reward,
            terminated,
            truncated,
            extras,
        ) = self.env.step(actions)

        # These observations still correspond to the target used for this step.
        obs_dict = (
            self._replace_target_observation(
                obs_dict
            )
        )

        rewards, reward_info = self._reward(
            obs_dict=obs_dict,
            actions=actions,
        )

        done_mask = terminated | truncated
        dones = done_mask.to(
            dtype=torch.long
        )

        # These measurements are recorded before target resampling.
        infos: dict[str, Any] = {
            "handreach": reward_info,
            "episode": {
                "reward": (
                    reward_info["reward"].mean()
                ),
                "distance_sum": (
                    reward_info[
                        "distance_sum"
                    ].mean()
                ),
                "distance_mean": (
                    reward_info[
                        "distance_mean"
                    ].mean()
                ),
                "distance_max": (
                    reward_info[
                        "distance_max"
                    ].mean()
                ),
                "thumb_distance": (
                    reward_info[
                        "per_finger_distance"
                    ][:, 0].mean()
                ),
                "index_distance": (
                    reward_info[
                        "per_finger_distance"
                    ][:, 1].mean()
                ),
                "solved_fraction": (
                    reward_info[
                        "solved_fraction"
                    ].mean()
                ),
                "very_solved_fraction": (
                    reward_info[
                        "very_solved_fraction"
                    ].mean()
                ),
                "all_solved": (
                    reward_info[
                        "all_solved"
                    ].mean()
                ),
                "all_very_solved": (
                    reward_info[
                        "all_very_solved"
                    ].mean()
                ),
                "distance_penalty": (
                    reward_info[
                        "distance_penalty"
                    ].mean()
                ),
                "solved_bonus": (
                    reward_info[
                        "solved_bonus"
                    ].mean()
                ),
                "action_penalty": (
                    reward_info[
                        "action_penalty"
                    ].mean()
                ),
                "velocity_penalty": (
                    reward_info[
                        "velocity_penalty"
                    ].mean()
                ),
                "robust_max_penalty": (
                    reward_info[
                        "robust_max_penalty"
                    ].mean()
                ),
            },
        }

        if isinstance(extras, dict):
            if "time_outs" in extras:
                infos["time_outs"] = (
                    extras["time_outs"]
                )

            if "log" in extras:
                infos["log"] = extras["log"]

        # Resampling occurs after measuring the completed step.
        self._resample_targets(done_mask)

        # The returned observation must contain the target for the next step.
        obs_dict = (
            self._replace_target_observation(
                obs_dict
            )
        )

        obs = TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
        )

        self._validate_observation(
            obs["policy"]
        )

        return (
            obs,
            rewards,
            dones,
            infos,
        )

    def _validate_observation(
        self,
        policy_obs: torch.Tensor,
    ) -> None:
        if policy_obs.ndim != 2:
            raise RuntimeError(
                "Expected a two-dimensional "
                "policy observation, got "
                f"{tuple(policy_obs.shape)}."
            )

        expected_shape = (
            self.num_envs,
            OBS_DIM,
        )

        if tuple(policy_obs.shape) != expected_shape:
            raise RuntimeError(
                f"Expected policy observation "
                f"shape {expected_shape}, got "
                f"{tuple(policy_obs.shape)}."
            )

    def _nominal_targets(
        self,
        device: str | torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            NOMINAL_FINGERTIP_TARGETS,
            device=device,
            dtype=torch.float32,
        )

    def _sample_targets(
        self,
        count: int,
    ) -> torch.Tensor:
        nominal = self._nominal_targets(
            device="cpu"
        )

        shared_low = torch.tensor(
            SHARED_TARGET_OFFSET_LOW,
            device="cpu",
            dtype=torch.float32,
        )

        shared_high = torch.tensor(
            SHARED_TARGET_OFFSET_HIGH,
            device="cpu",
            dtype=torch.float32,
        )

        jitter_low = torch.tensor(
            PER_FINGER_TARGET_JITTER_LOW,
            device="cpu",
            dtype=torch.float32,
        )

        jitter_high = torch.tensor(
            PER_FINGER_TARGET_JITTER_HIGH,
            device="cpu",
            dtype=torch.float32,
        )

        shared_samples = torch.rand(
            (
                count,
                1,
                3,
            ),
            generator=self.target_rng,
            device="cpu",
            dtype=torch.float32,
        )

        shared_offsets = (
            shared_low
            + (
                shared_high
                - shared_low
            )
            * shared_samples
        )

        jitter_samples = torch.rand(
            (
                count,
                NUM_FINGERTIPS,
                3,
            ),
            generator=self.target_rng,
            device="cpu",
            dtype=torch.float32,
        )

        per_finger_jitter = (
            jitter_low
            + (
                jitter_high
                - jitter_low
            )
            * jitter_samples
        )

        targets = (
            nominal.unsqueeze(0)
            + shared_offsets
            + per_finger_jitter
        )

        return targets.to(self.device)

    def _resample_targets(
        self,
        mask: torch.Tensor,
    ) -> None:
        if not RANDOMIZE_TARGET:
            nominal = self._nominal_targets(
                device=self.device
            )

            self.target_positions[:] = (
                nominal.unsqueeze(0)
            )

            return

        mask = mask.to(
            device=self.device,
            dtype=torch.bool,
        )

        if not bool(mask.any()):
            return

        count = int(
            mask.sum().detach().cpu()
        )

        self.target_positions[mask] = (
            self._sample_targets(count)
        )

    def _replace_target_observation(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs_dict = dict(obs_dict)

        policy_obs = (
            obs_dict["policy"].clone()
        )

        if policy_obs.shape[-1] != OBS_DIM:
            raise RuntimeError(
                f"Expected raw policy observation "
                f"dimension {OBS_DIM}, got "
                f"{policy_obs.shape[-1]}."
            )

        tip_pos_flat = (
            policy_obs[:, TIP_POS_SLICE]
        )

        tip_pos = tip_pos_flat.reshape(
            self.num_envs,
            NUM_FINGERTIPS,
            3,
        )

        target_error = (
            self.target_positions
            - tip_pos
        )

        policy_obs[
            :,
            TARGET_POS_SLICE,
        ] = self.target_positions.reshape(
            self.num_envs,
            FINGERTIP_POS_DIM,
        )

        policy_obs[
            :,
            TARGET_ERR_SLICE,
        ] = target_error.reshape(
            self.num_envs,
            FINGERTIP_POS_DIM,
        )

        obs_dict["policy"] = policy_obs

        return obs_dict

    def _reward(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        policy_obs = obs_dict["policy"]

        qvel = policy_obs[:, QVEL_SLICE]

        target_error = policy_obs[
            :,
            TARGET_ERR_SLICE,
        ].reshape(
            self.num_envs,
            NUM_FINGERTIPS,
            3,
        )

        # Metres for each fingertip.
        per_finger_distance = (
            torch.linalg.norm(
                target_error,
                dim=-1,
            )
        )

        distance_sum = (
            per_finger_distance.sum(
                dim=-1
            )
        )

        distance_mean = (
            per_finger_distance.mean(
                dim=-1
            )
        )

        distance_max = (
            per_finger_distance.max(
                dim=-1
            ).values
        )

        distance_penalty = (
            -DISTANCE_REWARD_SCALE
            * distance_sum
        )

        robust_max_excess = torch.clamp(
            (
                distance_max
                - ROBUST_MAX_DISTANCE_TH
            ),
            min=0.0,
        )

        robust_max_penalty = (
            ROBUST_MAX_DISTANCE_PENALTY_SCALE
            * robust_max_excess
        )

        velocity_penalty = (
            VELOCITY_PENALTY_SCALE
            * torch.mean(
                torch.abs(qvel),
                dim=-1,
            )
        )

        action_penalty = (
            ACTION_PENALTY_SCALE
            * torch.mean(
                actions.square(),
                dim=-1,
            )
        )

        solved = (
            per_finger_distance
            < SOLVED_DISTANCE_TH
        )

        very_solved = (
            per_finger_distance
            < VERY_SOLVED_DISTANCE_TH
        )

        solved_bonus = (
            SOLVED_BONUS_VALUE
            * solved.float().sum(
                dim=-1
            )
        )

        rewards = (
            distance_penalty
            + solved_bonus
            - robust_max_penalty
            - velocity_penalty
            - action_penalty
        )

        solved_fraction = (
            solved.float().mean(
                dim=-1
            )
        )

        very_solved_fraction = (
            very_solved.float().mean(
                dim=-1
            )
        )

        all_solved = (
            solved.all(
                dim=-1
            ).float()
        )

        all_very_solved = (
            very_solved.all(
                dim=-1
            ).float()
        )

        reward_info = {
            "reward": rewards.detach(),
            "per_finger_distance": (
                per_finger_distance.detach()
            ),
            "distance_sum": (
                distance_sum.detach()
            ),
            "distance_mean": (
                distance_mean.detach()
            ),
            "distance_max": (
                distance_max.detach()
            ),
            "distance_penalty": (
                distance_penalty.detach()
            ),
            "solved_bonus": (
                solved_bonus.detach()
            ),
            "solved_fraction": (
                solved_fraction.detach()
            ),
            "very_solved_fraction": (
                very_solved_fraction.detach()
            ),
            "all_solved": (
                all_solved.detach()
            ),
            "all_very_solved": (
                all_very_solved.detach()
            ),
            "action_penalty": (
                action_penalty.detach()
            ),
            "velocity_penalty": (
                velocity_penalty.detach()
            ),
            "robust_max_penalty": (
                robust_max_penalty.detach()
            ),
        }

        return rewards, reward_info


# =============================================================================
# RSL-RL CONFIGURATION
# =============================================================================

def remove_unsupported_rsl_rl_keys(
    train_cfg: dict[str, Any],
) -> None:
    for section in (
        "actor",
        "critic",
    ):
        section_cfg = train_cfg.get(
            section
        )

        if not isinstance(
            section_cfg,
            dict,
        ):
            continue

        for key in list(
            section_cfg.keys()
        ):
            if (
                key == "cnn_cfg"
                or key.startswith("rnn")
            ):
                section_cfg.pop(key)


def set_key_recursively(
    data: dict[str, Any],
    key: str,
    value: Any,
) -> bool:
    found = False

    if key in data:
        data[key] = value
        found = True

    for child in data.values():
        if isinstance(child, dict):
            child_found = (
                set_key_recursively(
                    child,
                    key,
                    value,
                )
            )

            found = child_found or found

    return found


def set_training_value(
    train_cfg: dict[str, Any],
    possible_keys: tuple[str, ...],
    value: Any,
) -> None:
    if value is None:
        return

    for key in possible_keys:
        if set_key_recursively(
            train_cfg,
            key,
            value,
        ):
            return


def apply_training_config(
    train_cfg: dict[str, Any],
) -> None:
    train_cfg[
        "num_steps_per_env"
    ] = NUM_STEPS_PER_ENV

    train_cfg["logger"] = "tensorboard"

    set_training_value(
        train_cfg,
        (
            "learning_rate",
            "lr",
        ),
        LEARNING_RATE,
    )

    set_training_value(
        train_cfg,
        ("num_learning_epochs",),
        NUM_LEARNING_EPOCHS,
    )

    set_training_value(
        train_cfg,
        (
            "num_mini_batches",
            "num_minibatches",
        ),
        NUM_MINI_BATCHES,
    )

    set_training_value(
        train_cfg,
        (
            "clip_param",
            "clip_range",
        ),
        CLIP_PARAM,
    )

    set_training_value(
        train_cfg,
        ("desired_kl",),
        DESIRED_KL,
    )

    set_training_value(
        train_cfg,
        ("max_grad_norm",),
        MAX_GRAD_NORM,
    )

    set_training_value(
        train_cfg,
        (
            "entropy_coef",
            "entropy_coeff",
        ),
        ENTROPY_COEF,
    )

    algorithm_cfg = train_cfg.get(
        "algorithm"
    )

    if not isinstance(
        algorithm_cfg,
        dict,
    ):
        raise RuntimeError(
            "RSL-RL algorithm configuration "
            "was not found in "
            "train_cfg['algorithm']."
        )

    algorithm_cfg["schedule"] = SCHEDULE


def install_adaptive_lr_bounds(
    algorithm,
) -> None:
    if SCHEDULE != "adaptive":
        return

    if not (
        0.0
        < MIN_ADAPTIVE_LR
        <= LEARNING_RATE
        <= MAX_ADAPTIVE_LR
    ):
        raise ValueError(
            "Learning rates must satisfy "
            "0 < MIN_ADAPTIVE_LR "
            "<= LEARNING_RATE "
            "<= MAX_ADAPTIVE_LR."
        )

    optimizer = getattr(
        algorithm,
        "optimizer",
        None,
    )

    if (
        optimizer is None
        or len(
            optimizer.param_groups
        )
        == 0
    ):
        raise RuntimeError(
            "RSL-RL PPO optimizer "
            "was not found."
        )

    if not hasattr(
        optimizer,
        "register_step_pre_hook",
    ):
        raise RuntimeError(
            "This PyTorch version does not "
            "support optimizer step pre-hooks."
        )

    def clamp_lr_before_optimizer_step(
        _optimizer,
        _args,
        _kwargs,
    ) -> None:
        bounded_lr = min(
            MAX_ADAPTIVE_LR,
            max(
                MIN_ADAPTIVE_LR,
                float(
                    algorithm.learning_rate
                ),
            ),
        )

        algorithm.learning_rate = bounded_lr

        for parameter_group in (
            optimizer.param_groups
        ):
            parameter_group[
                "lr"
            ] = bounded_lr

    clamp_lr_before_optimizer_step(
        optimizer,
        (),
        {},
    )

    optimizer.register_step_pre_hook(
        clamp_lr_before_optimizer_step
    )


# =============================================================================
# FILE HELPERS
# =============================================================================

def resolve_path(
    path_arg: str,
) -> Path:
    path = Path(path_arg)

    if not path.is_absolute():
        path = REPO_ROOT / path

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def default_metrics_path(
    policy_path: Path,
) -> Path:
    return policy_path.with_name(
        f"{policy_path.stem}_metrics.json"
    )


def save_metrics(
    path: Path,
    config: dict[str, Any],
    rows: list[dict[str, float]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "config": config,
                "metrics": rows,
            },
            file,
            indent=2,
        )


# =============================================================================
# PPO METRIC CAPTURE
# =============================================================================

def to_float(
    value: Any,
) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
    ):
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return float(value)

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.numel() == 0:
            return None

        return float(
            value.detach()
            .float()
            .mean()
            .cpu()
        )

    try:
        return float(value)
    except Exception:
        return None


def metric_name(
    name: str,
) -> str | None:
    name = (
        name.lower()
        .strip()
        .replace(" ", "_")
    )

    mapping = {
        "value": "train/value",
        "value_loss": "train/value",
        "surrogate": "train/surrogate",
        "surrogate_loss": "train/surrogate",
        "policy_loss": "train/surrogate",
        "entropy": "train/entropy",
        "entropy_loss": "train/entropy",
        "kl": "train/approx_kl",
        "kl_mean": "train/approx_kl",
        "mean_kl": "train/approx_kl",
        "approx_kl": "train/approx_kl",
        "approx_kl_mean": (
            "train/approx_kl"
        ),
        "kl_divergence": (
            "train/approx_kl"
        ),
    }

    return mapping.get(name)


def find_kl_source(runner):
    algorithm = getattr(
        runner,
        "alg",
        None,
    )

    candidates = []

    if algorithm is not None:
        candidates.extend(
            [
                getattr(
                    algorithm,
                    "actor",
                    None,
                ),
                getattr(
                    algorithm,
                    "actor_critic",
                    None,
                ),
                getattr(
                    algorithm,
                    "policy",
                    None,
                ),
            ]
        )

    candidates.extend(
        [
            getattr(
                runner,
                "actor",
                None,
            ),
            getattr(
                runner,
                "actor_critic",
                None,
            ),
            getattr(
                runner,
                "policy",
                None,
            ),
        ]
    )

    for candidate in candidates:
        if (
            candidate is not None
            and hasattr(
                candidate,
                "get_kl_divergence",
            )
        ):
            return candidate

    return None


def patch_runner_for_metrics(
    runner,
) -> dict[str, list[Any]]:
    storage: dict[str, list[Any]] = {
        "updates": [],
        "kl_values": [],
    }

    algorithm = getattr(
        runner,
        "alg",
        None,
    )

    if (
        algorithm is None
        or not hasattr(
            algorithm,
            "update",
        )
    ):
        return storage

    optimizer = getattr(
        algorithm,
        "optimizer",
        None,
    )

    fixed_lr = None

    if (
        LOG_KL_WITH_FIXED_SCHEDULE
        and SCHEDULE == "fixed"
        and optimizer is not None
        and len(
            optimizer.param_groups
        )
        > 0
    ):
        fixed_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        def force_fixed_lr() -> None:
            algorithm.learning_rate = (
                fixed_lr
            )

            for parameter_group in (
                optimizer.param_groups
            ):
                parameter_group[
                    "lr"
                ] = fixed_lr

        force_fixed_lr()

        if not hasattr(
            optimizer,
            "register_step_pre_hook",
        ):
            raise RuntimeError(
                "This PyTorch version does "
                "not support optimizer "
                "step pre-hooks."
            )

        def fixed_lr_pre_step_hook(
            _optimizer,
            _args,
            _kwargs,
        ) -> None:
            force_fixed_lr()

        optimizer.register_step_pre_hook(
            fixed_lr_pre_step_hook
        )

    kl_source = find_kl_source(runner)

    if kl_source is not None:
        original_kl_function = (
            kl_source.get_kl_divergence
        )

        def wrapped_kl_function(
            *args,
            **kwargs,
        ):
            kl_value = (
                original_kl_function(
                    *args,
                    **kwargs,
                )
            )

            scalar = to_float(kl_value)

            if scalar is not None:
                storage[
                    "kl_values"
                ].append(scalar)

            return kl_value

        kl_source.get_kl_divergence = (
            wrapped_kl_function
        )

    original_update = algorithm.update

    def wrapped_update(
        *args,
        **kwargs,
    ):
        kl_start = len(
            storage["kl_values"]
        )

        previous_schedule = getattr(
            algorithm,
            "schedule",
            None,
        )

        if (
            LOG_KL_WITH_FIXED_SCHEDULE
            and SCHEDULE == "fixed"
        ):
            algorithm.schedule = "adaptive"

        try:
            update_output = original_update(
                *args,
                **kwargs,
            )
        finally:
            if (
                LOG_KL_WITH_FIXED_SCHEDULE
                and SCHEDULE == "fixed"
            ):
                algorithm.schedule = (
                    previous_schedule
                )

                if fixed_lr is not None:
                    force_fixed_lr()

        record: dict[str, float] = {}

        if (
            optimizer is not None
            and len(
                optimizer.param_groups
            )
            > 0
        ):
            record[
                "train/learning_rate"
            ] = float(
                optimizer.param_groups[0][
                    "lr"
                ]
            )

        if isinstance(
            update_output,
            dict,
        ):
            for (
                raw_name,
                raw_value,
            ) in update_output.items():
                clean_name = metric_name(
                    str(raw_name)
                )

                value = to_float(raw_value)

                if (
                    clean_name is not None
                    and value is not None
                ):
                    record[clean_name] = value

        elif isinstance(
            update_output,
            (tuple, list),
        ):
            names = [
                "train/value",
                "train/surrogate",
                "train/entropy",
                "train/approx_kl",
            ]

            for (
                index,
                raw_value,
            ) in enumerate(
                update_output[
                    :len(names)
                ]
            ):
                value = to_float(raw_value)

                if value is not None:
                    record[
                        names[index]
                    ] = value

        new_kl_values = storage[
            "kl_values"
        ][kl_start:]

        if len(new_kl_values) > 0:
            record[
                "train/approx_kl"
            ] = float(
                sum(new_kl_values)
                / len(new_kl_values)
            )

        if len(record) > 0:
            storage[
                "updates"
            ].append(record)

        return update_output

    algorithm.update = wrapped_update

    return storage


def average_update_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    if len(records) == 0:
        return {
            "train/num_updates": 0.0,
        }

    keys = sorted(
        {
            key
            for record in records
            for key in record
        }
    )

    averaged = {
        "train/num_updates": float(
            len(records)
        ),
    }

    for key in keys:
        values = [
            record[key]
            for record in records
            if key in record
        ]

        averaged[key] = float(
            sum(values) / len(values)
        )

    return averaged


# =============================================================================
# POLICY INFERENCE
# =============================================================================

def unpack_action(
    action: Any,
) -> torch.Tensor:
    if isinstance(
        action,
        torch.Tensor,
    ):
        return action

    if isinstance(
        action,
        TensorDict,
    ):
        for key in (
            "action",
            "actions",
        ):
            if key in action:
                return action[key]

    if isinstance(action, dict):
        for key in (
            "action",
            "actions",
        ):
            if key in action:
                return action[key]

    if isinstance(action, tuple):
        return unpack_action(action[0])

    raise TypeError(
        "Unsupported action output "
        f"type: {type(action)}"
    )


def get_policy_action(
    runner,
    obs: TensorDict,
    device: str,
) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(
            runner,
            "get_inference_policy",
        ):
            policy = (
                runner.get_inference_policy(
                    device=device
                )
            )

            for policy_input in (
                obs,
                obs["policy"],
            ):
                try:
                    return unpack_action(
                        policy(policy_input)
                    )
                except Exception:
                    pass

        algorithm = getattr(
            runner,
            "alg",
            None,
        )

        if algorithm is not None:
            for actor_name in (
                "actor",
                "actor_critic",
            ):
                actor = getattr(
                    algorithm,
                    actor_name,
                    None,
                )

                if actor is None:
                    continue

                if hasattr(actor, "eval"):
                    actor.eval()

                for method_name in (
                    "act_inference",
                    "act",
                    "forward",
                    "__call__",
                ):
                    method = getattr(
                        actor,
                        method_name,
                        None,
                    )

                    if method is None:
                        continue

                    for policy_input in (
                        obs,
                        obs["policy"],
                    ):
                        try:
                            return unpack_action(
                                method(
                                    policy_input
                                )
                            )
                        except Exception:
                            pass

    raise RuntimeError(
        "Could not obtain an action "
        "from this RSL-RL policy version."
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_policy(
    runner,
    eval_env: HandReachRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    with redirect_output(quiet):
        obs = eval_env.reset(seed=seed)

    rewards = []
    per_finger_distances = []
    distance_sums = []
    distance_means = []
    distance_maxes = []
    solved_values = []
    very_solved_values = []
    all_solved_values = []
    all_very_solved_values = []

    for _ in range(EVAL_STEPS):
        actions = get_policy_action(
            runner=runner,
            obs=obs,
            device=device,
        )

        (
            obs,
            reward,
            _done,
            infos,
        ) = eval_env.step(actions)

        # These values were captured before target resampling.
        reward_info = infos["handreach"]

        per_finger_distance = (
            reward_info[
                "per_finger_distance"
            ]
        )

        distance_sum = (
            reward_info["distance_sum"]
        )

        distance_mean = (
            reward_info["distance_mean"]
        )

        distance_max = (
            reward_info["distance_max"]
        )

        solved = (
            per_finger_distance
            < SOLVED_DISTANCE_TH
        )

        very_solved = (
            per_finger_distance
            < VERY_SOLVED_DISTANCE_TH
        )

        rewards.append(
            reward.detach()
        )

        per_finger_distances.append(
            per_finger_distance.detach()
        )

        distance_sums.append(
            distance_sum.detach()
        )

        distance_means.append(
            distance_mean.detach()
        )

        distance_maxes.append(
            distance_max.detach()
        )

        solved_values.append(
            solved.float().detach()
        )

        very_solved_values.append(
            very_solved.float().detach()
        )

        all_solved_values.append(
            solved.all(
                dim=-1
            ).float().detach()
        )

        all_very_solved_values.append(
            very_solved.all(
                dim=-1
            ).float().detach()
        )

    reward_tensor = torch.stack(rewards)

    per_finger_distance_tensor = (
        torch.stack(
            per_finger_distances
        )
    )

    distance_sum_tensor = torch.stack(
        distance_sums
    )

    distance_mean_tensor = torch.stack(
        distance_means
    )

    distance_max_tensor = torch.stack(
        distance_maxes
    )

    solved_tensor = torch.stack(
        solved_values
    )

    very_solved_tensor = torch.stack(
        very_solved_values
    )

    all_solved_tensor = torch.stack(
        all_solved_values
    )

    all_very_solved_tensor = torch.stack(
        all_very_solved_values
    )

    metrics = {
        "eval/mean_reward": float(
            reward_tensor.sum(
                dim=0
            ).mean().cpu()
        ),
        "eval/reward_per_step": float(
            reward_tensor.mean().cpu()
        ),
        "eval/distance": float(
            distance_mean_tensor.mean().cpu()
        ),
        "eval/distance_sum": float(
            distance_sum_tensor.mean().cpu()
        ),
        "eval/distance_mean": float(
            distance_mean_tensor.mean().cpu()
        ),
        "eval/distance_max": float(
            distance_max_tensor.mean().cpu()
        ),
        "eval/distance_sum_last": float(
            distance_sum_tensor[
                -1
            ].mean().cpu()
        ),
        "eval/distance_mean_last": float(
            distance_mean_tensor[
                -1
            ].mean().cpu()
        ),
        "eval/distance_max_last": float(
            distance_max_tensor[
                -1
            ].mean().cpu()
        ),
        "eval/solved_fraction": float(
            solved_tensor.mean().cpu()
        ),
        "eval/very_solved_fraction": float(
            very_solved_tensor.mean().cpu()
        ),
        "eval/all_solved_fraction": float(
            all_solved_tensor.mean().cpu()
        ),
        "eval/all_very_solved_fraction": float(
            all_very_solved_tensor.mean().cpu()
        ),
    }

    for (
        finger_index,
        finger_name,
    ) in enumerate(FINGER_NAMES):
        metrics[
            f"eval/{finger_name}_distance"
        ] = float(
            per_finger_distance_tensor[
                :,
                :,
                finger_index,
            ].mean().cpu()
        )

        metrics[
            f"eval/{finger_name}_distance_last"
        ] = float(
            per_finger_distance_tensor[
                -1,
                :,
                finger_index,
            ].mean().cpu()
        )

        metrics[
            f"eval/{finger_name}_solved_fraction"
        ] = float(
            solved_tensor[
                :,
                :,
                finger_index,
            ].mean().cpu()
        )

    return metrics


def make_metric_row(
    *,
    eval_index: int,
    iteration: int,
    env_steps: int,
    elapsed_sec: float,
    eval_metrics: dict[str, float],
    train_metrics: dict[str, float],
) -> dict[str, float]:
    row = {
        "eval_index": float(eval_index),
        "iteration": float(iteration),
        "num_steps": float(env_steps),
        "env_steps": float(env_steps),
        "elapsed_sec": float(elapsed_sec),
    }

    row.update(eval_metrics)
    row.update(train_metrics)

    return row


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def make_wrapped_env(
    make_env,
    env_id: str,
    device: str,
    num_envs: int,
    seed: int,
    quiet: bool,
) -> HandReachRslRlWrapper:
    with redirect_output(quiet):
        env = make_env(
            env_id,
            backend="mjlab",
            device=device,
            num_envs=num_envs,
        )

    wrapper = HandReachRslRlWrapper(
        env=env,
        seed=seed,
    )

    with redirect_output(quiet):
        wrapper.reset(seed=seed)

    return wrapper


# =============================================================================
# SAVED CONFIGURATION
# =============================================================================

def build_saved_config(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "env_id": args.env_id,
        "seed": args.seed,
        "device": args.device,
        "training": {
            "iterations": ITERATIONS,
            "num_envs": NUM_ENVS,
            "num_steps_per_env": (
                NUM_STEPS_PER_ENV
            ),
            "learning_rate": LEARNING_RATE,
            "min_adaptive_lr": (
                MIN_ADAPTIVE_LR
            ),
            "max_adaptive_lr": (
                MAX_ADAPTIVE_LR
            ),
            "schedule": SCHEDULE,
            "num_learning_epochs": (
                NUM_LEARNING_EPOCHS
            ),
            "num_mini_batches": (
                NUM_MINI_BATCHES
            ),
            "clip_param": CLIP_PARAM,
            "desired_kl": DESIRED_KL,
            "max_grad_norm": MAX_GRAD_NORM,
            "entropy_coef": ENTROPY_COEF,
        },
        "evaluation": {
            "num_evals": NUM_EVALS,
            "eval_num_envs": EVAL_NUM_ENVS,
            "eval_steps": EVAL_STEPS,
            "fixed_seed": (
                args.seed + 20_000
            ),
        },
        "target": {
            "randomize_target": (
                RANDOMIZE_TARGET
            ),
            "finger_names": FINGER_NAMES,
            "nominal_fingertip_targets": (
                NOMINAL_FINGERTIP_TARGETS
            ),
            "shared_target_offset_low": (
                SHARED_TARGET_OFFSET_LOW
            ),
            "shared_target_offset_high": (
                SHARED_TARGET_OFFSET_HIGH
            ),
            "per_finger_target_jitter_low": (
                PER_FINGER_TARGET_JITTER_LOW
            ),
            "per_finger_target_jitter_high": (
                PER_FINGER_TARGET_JITTER_HIGH
            ),
        },
        "reward": {
            "distance_reward_scale": (
                DISTANCE_REWARD_SCALE
            ),
            "robust_max_distance_penalty_scale": (
                ROBUST_MAX_DISTANCE_PENALTY_SCALE
            ),
            "robust_max_distance_th": (
                ROBUST_MAX_DISTANCE_TH
            ),
            "velocity_penalty_scale": (
                VELOCITY_PENALTY_SCALE
            ),
            "action_penalty_scale": (
                ACTION_PENALTY_SCALE
            ),
            "solved_distance_th": (
                SOLVED_DISTANCE_TH
            ),
            "very_solved_distance_th": (
                VERY_SOLVED_DISTANCE_TH
            ),
            "solved_bonus_value": (
                SOLVED_BONUS_VALUE
            ),
        },
        "observation_layout": {
            "qpos": [0, 23],
            "qvel": [23, 46],
            "tip_pos": [46, 52],
            "target_pos": [52, 58],
            "target_err": [58, 64],
            "act": [64, 103],
            "total": OBS_DIM,
        },
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    args: argparse.Namespace,
) -> None:
    torch.manual_seed(args.seed)

    policy_path = resolve_path(
        args.policy_path
    )

    if args.metrics_path is None:
        metrics_path = (
            default_metrics_path(
                policy_path
            )
        )
    else:
        metrics_path = resolve_path(
            args.metrics_path
        )

    (
        make_env,
        load_rl_cfg,
    ) = import_mjlab_tools(
        quiet=args.quiet
    )

    from rsl_rl.runners import OnPolicyRunner

    train_env = make_wrapped_env(
        make_env=make_env,
        env_id=args.env_id,
        device=args.device,
        num_envs=NUM_ENVS,
        seed=args.seed,
        quiet=args.quiet,
    )

    eval_env = make_wrapped_env(
        make_env=make_env,
        env_id=args.env_id,
        device=args.device,
        num_envs=EVAL_NUM_ENVS,
        seed=args.seed + 10_000,
        quiet=args.quiet,
    )

    train_cfg = load_rl_cfg(
        args.env_id
    )

    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }

    train_cfg_dict = asdict(
        train_cfg
    )

    apply_training_config(
        train_cfg_dict
    )

    remove_unsupported_rsl_rl_keys(
        train_cfg_dict
    )

    with redirect_output(args.quiet):
        runner = OnPolicyRunner(
            env=train_env,
            train_cfg=train_cfg_dict,
            log_dir=None,
            device=args.device,
        )

    algorithm = getattr(
        runner,
        "alg",
        None,
    )

    if algorithm is None:
        raise RuntimeError(
            "OnPolicyRunner did not "
            "expose runner.alg."
        )

    install_adaptive_lr_bounds(
        algorithm
    )

    effective_schedule = getattr(
        algorithm,
        "schedule",
        None,
    )

    optimizer = getattr(
        algorithm,
        "optimizer",
        None,
    )

    if (
        optimizer is None
        or len(
            optimizer.param_groups
        )
        == 0
    ):
        raise RuntimeError(
            "RSL-RL PPO optimizer "
            "was not found."
        )

    effective_lr = float(
        optimizer.param_groups[0]["lr"]
    )

    if effective_schedule != SCHEDULE:
        raise RuntimeError(
            f"Expected PPO schedule "
            f"{SCHEDULE!r}, but RSL-RL "
            f"is using "
            f"{effective_schedule!r}."
        )

    learning_rate_tolerance = max(
        1e-12,
        abs(LEARNING_RATE) * 1e-6,
    )

    if (
        abs(
            effective_lr
            - LEARNING_RATE
        )
        > learning_rate_tolerance
    ):
        raise RuntimeError(
            f"Expected optimizer learning "
            f"rate {LEARNING_RATE}, but "
            f"RSL-RL is using "
            f"{effective_lr}."
        )

    update_storage = (
        patch_runner_for_metrics(
            runner
        )
    )

    steps_per_iteration = (
        NUM_ENVS
        * NUM_STEPS_PER_ENV
    )

    eval_every = max(
        1,
        ITERATIONS // NUM_EVALS,
    )

    saved_config = build_saved_config(
        args
    )

    metric_rows: list[
        dict[str, float]
    ] = []

    print("=" * 80)
    print(
        "MJLAB TWO-FINGER "
        "HAND REACH TRAINING"
    )
    print("=" * 80)
    print(
        f"environment:           "
        f"{args.env_id}"
    )
    print(
        f"device:                "
        f"{args.device}"
    )
    print(
        f"policy path:           "
        f"{policy_path}"
    )
    print(
        f"metrics path:          "
        f"{metrics_path}"
    )
    print(
        f"iterations:            "
        f"{ITERATIONS}"
    )
    print(
        f"training environments: "
        f"{NUM_ENVS}"
    )
    print(
        f"steps per environment: "
        f"{NUM_STEPS_PER_ENV}"
    )
    print(
        f"PPO schedule:          "
        f"{effective_schedule}"
    )
    print(
        f"adaptive LR range:     "
        f"[{MIN_ADAPTIVE_LR}, "
        f"{MAX_ADAPTIVE_LR}]"
    )
    print(
        f"configured LR:         "
        f"{LEARNING_RATE}"
    )
    print(
        f"internal PPO LR:       "
        f"{getattr(algorithm, 'learning_rate', None)}"
    )
    print(
        f"optimizer LR:          "
        f"{effective_lr}"
    )
    print(
        f"desired KL:            "
        f"{getattr(algorithm, 'desired_kl', None)}"
    )
    print(
        f"learning epochs:       "
        f"{getattr(algorithm, 'num_learning_epochs', None)}"
    )
    print(
        f"mini-batches:          "
        f"{getattr(algorithm, 'num_mini_batches', None)}"
    )
    print(
        f"expected env steps:    "
        f"{ITERATIONS * steps_per_iteration}"
    )
    print(
        f"evaluations:           "
        f"{NUM_EVALS}"
    )
    print(
        f"observation dimension: "
        f"{OBS_DIM}"
    )
    print(
        f"action dimension:      "
        f"{ACTION_DIM}"
    )
    print(
        f"fingertips:            "
        f"{FINGER_NAMES}"
    )
    print(
        f"nominal targets:       "
        f"{NOMINAL_FINGERTIP_TARGETS}"
    )
    print(
        f"randomize target:      "
        f"{RANDOMIZE_TARGET}"
    )
    print("=" * 80)

    start_time = time.time()
    completed_iterations = 0
    eval_index = 0

    initial_evaluation = evaluate_policy(
        runner=runner,
        eval_env=eval_env,
        seed=args.seed + 20_000,
        device=args.device,
        quiet=args.quiet,
    )

    initial_row = make_metric_row(
        eval_index=eval_index,
        iteration=completed_iterations,
        env_steps=0,
        elapsed_sec=0.0,
        eval_metrics=initial_evaluation,
        train_metrics={
            "train/num_updates": 0.0,
            "train/learning_rate": (
                effective_lr
            ),
        },
    )

    metric_rows.append(initial_row)

    save_metrics(
        path=metrics_path,
        config=saved_config,
        rows=metric_rows,
    )

    print(
        f"[eval {eval_index:02d}/"
        f"{NUM_EVALS:02d}] "
        f"iter=0 "
        f"steps=0 "
        f"return="
        f"{initial_evaluation['eval/mean_reward']:.3f} "
        f"reward/step="
        f"{initial_evaluation['eval/reward_per_step']:.4f} "
        f"mean="
        f"{initial_evaluation['eval/distance_mean']:.4f} "
        f"max="
        f"{initial_evaluation['eval/distance_max']:.4f} "
        f"TH="
        f"{initial_evaluation['eval/THtip_distance']:.4f} "
        f"IF="
        f"{initial_evaluation['eval/IFtip_distance']:.4f} "
        f"solved="
        f"{initial_evaluation['eval/solved_fraction']:.3f} "
        f"all="
        f"{initial_evaluation['eval/all_solved_fraction']:.3f}"
    )

    while (
        completed_iterations
        < ITERATIONS
    ):
        remaining_iterations = (
            ITERATIONS
            - completed_iterations
        )

        chunk_iterations = min(
            eval_every,
            remaining_iterations,
        )

        update_storage[
            "updates"
        ].clear()

        update_storage[
            "kl_values"
        ].clear()

        with redirect_output(
            args.quiet
        ):
            runner.learn(
                num_learning_iterations=(
                    chunk_iterations
                ),
                init_at_random_ep_len=(
                    completed_iterations
                    == 0
                ),
            )

        train_metrics = (
            average_update_metrics(
                update_storage["updates"]
            )
        )

        completed_iterations += (
            chunk_iterations
        )

        env_steps = (
            completed_iterations
            * steps_per_iteration
        )

        eval_index += 1

        evaluation = evaluate_policy(
            runner=runner,
            eval_env=eval_env,
            seed=args.seed + 20_000,
            device=args.device,
            quiet=args.quiet,
        )

        elapsed_seconds = (
            time.time() - start_time
        )

        row = make_metric_row(
            eval_index=eval_index,
            iteration=completed_iterations,
            env_steps=env_steps,
            elapsed_sec=elapsed_seconds,
            eval_metrics=evaluation,
            train_metrics=train_metrics,
        )

        metric_rows.append(row)

        runner.save(
            str(policy_path)
        )

        save_metrics(
            path=metrics_path,
            config=saved_config,
            rows=metric_rows,
        )

        current_learning_rate = (
            train_metrics.get(
                "train/learning_rate",
                float(
                    optimizer.param_groups[
                        0
                    ]["lr"]
                ),
            )
        )

        print(
            f"[eval {eval_index:02d}/"
            f"{NUM_EVALS:02d}] "
            f"iter={completed_iterations} "
            f"steps={env_steps} "
            f"return="
            f"{evaluation['eval/mean_reward']:.3f} "
            f"mean="
            f"{evaluation['eval/distance_mean']:.4f} "
            f"max="
            f"{evaluation['eval/distance_max']:.4f} "
            f"TH="
            f"{evaluation['eval/THtip_distance']:.4f} "
            f"IF="
            f"{evaluation['eval/IFtip_distance']:.4f} "
            f"solved="
            f"{evaluation['eval/solved_fraction']:.3f} "
            f"all="
            f"{evaluation['eval/all_solved_fraction']:.3f} "
            f"lr="
            f"{current_learning_rate:.3e} "
            f"time="
            f"{elapsed_seconds:.1f}s"
        )

    runner.save(str(policy_path))

    save_metrics(
        path=metrics_path,
        config=saved_config,
        rows=metric_rows,
    )

    train_env.close()
    eval_env.close()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"policy:  {policy_path}")
    print(f"metrics: {metrics_path}")


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-id",
        type=str,
        default=TASK_ID,
    )

    parser.add_argument(
        "--policy-path",
        type=str,
        default=(
            "my_handreach_2F_policy.pt"
        ),
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--quiet",
        action=(
            argparse.BooleanOptionalAction
        ),
        default=True,
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config(
    args: argparse.Namespace,
) -> None:
    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested device "
            f"{args.device}, but CUDA "
            f"is not available."
        )

    positive_integer_values = {
        "ITERATIONS": ITERATIONS,
        "NUM_ENVS": NUM_ENVS,
        "NUM_STEPS_PER_ENV": (
            NUM_STEPS_PER_ENV
        ),
        "NUM_EVALS": NUM_EVALS,
        "EVAL_NUM_ENVS": EVAL_NUM_ENVS,
        "EVAL_STEPS": EVAL_STEPS,
        "NUM_LEARNING_EPOCHS": (
            NUM_LEARNING_EPOCHS
        ),
        "NUM_MINI_BATCHES": (
            NUM_MINI_BATCHES
        ),
    }

    for (
        name,
        value,
    ) in positive_integer_values.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be "
                "greater than zero."
            )

    if SCHEDULE not in (
        "fixed",
        "adaptive",
    ):
        raise ValueError(
            "SCHEDULE must be either "
            "'fixed' or 'adaptive'."
        )

    rollout_size = (
        NUM_ENVS
        * NUM_STEPS_PER_ENV
    )

    if (
        rollout_size
        % NUM_MINI_BATCHES
        != 0
    ):
        raise ValueError(
            "NUM_ENVS * NUM_STEPS_PER_ENV "
            "must be divisible by "
            "NUM_MINI_BATCHES."
        )

    if NUM_FINGERTIPS != 2:
        raise ValueError(
            "This training script expects "
            "exactly two fingertips."
        )

    if FINGERTIP_POS_DIM != 6:
        raise ValueError(
            "FINGERTIP_POS_DIM must be 6 "
            "for two xyz fingertips."
        )

    if tuple(FINGER_NAMES) != (
        "THtip",
        "IFtip",
    ):
        raise ValueError(
            "The fingertip order must be "
            "('THtip', 'IFtip')."
        )

    if OBS_DIM != 103:
        raise ValueError(
            "OBS_DIM must be 103 for the "
            "two-finger observation."
        )

    expected_slices = {
        "QPOS_SLICE": (
            QPOS_SLICE,
            slice(0, 23),
        ),
        "QVEL_SLICE": (
            QVEL_SLICE,
            slice(23, 46),
        ),
        "TIP_POS_SLICE": (
            TIP_POS_SLICE,
            slice(46, 52),
        ),
        "TARGET_POS_SLICE": (
            TARGET_POS_SLICE,
            slice(52, 58),
        ),
        "TARGET_ERR_SLICE": (
            TARGET_ERR_SLICE,
            slice(58, 64),
        ),
        "ACT_SLICE": (
            ACT_SLICE,
            slice(64, 103),
        ),
    }

    for (
        name,
        (
            actual_slice,
            expected_slice,
        ),
    ) in expected_slices.items():
        if actual_slice != expected_slice:
            raise ValueError(
                f"{name} must be "
                f"{expected_slice}."
            )

    if (
        len(
            NOMINAL_FINGERTIP_TARGETS
        )
        != NUM_FINGERTIPS
    ):
        raise ValueError(
            "NOMINAL_FINGERTIP_TARGETS "
            "must contain one xyz target "
            "per fingertip."
        )

    for target in (
        NOMINAL_FINGERTIP_TARGETS
    ):
        if len(target) != 3:
            raise ValueError(
                "Each nominal fingertip "
                "target must contain xyz."
            )

    target_ranges = (
        (
            "SHARED_TARGET_OFFSET_LOW",
            "SHARED_TARGET_OFFSET_HIGH",
            SHARED_TARGET_OFFSET_LOW,
            SHARED_TARGET_OFFSET_HIGH,
        ),
        (
            "PER_FINGER_TARGET_JITTER_LOW",
            "PER_FINGER_TARGET_JITTER_HIGH",
            PER_FINGER_TARGET_JITTER_LOW,
            PER_FINGER_TARGET_JITTER_HIGH,
        ),
    )

    for (
        low_name,
        high_name,
        low_values,
        high_values,
    ) in target_ranges:
        if (
            len(low_values) != 3
            or len(high_values) != 3
        ):
            raise ValueError(
                f"{low_name} and "
                f"{high_name} must "
                "contain xyz values."
            )

        for (
            low_value,
            high_value,
        ) in zip(
            low_values,
            high_values,
        ):
            if high_value <= low_value:
                raise ValueError(
                    f"Every value in "
                    f"{high_name} must be "
                    f"greater than the "
                    f"matching value in "
                    f"{low_name}."
                )

    if SOLVED_DISTANCE_TH <= 0.0:
        raise ValueError(
            "SOLVED_DISTANCE_TH must "
            "be positive."
        )

    if (
        VERY_SOLVED_DISTANCE_TH
        <= 0.0
    ):
        raise ValueError(
            "VERY_SOLVED_DISTANCE_TH "
            "must be positive."
        )

    if (
        VERY_SOLVED_DISTANCE_TH
        > SOLVED_DISTANCE_TH
    ):
        raise ValueError(
            "VERY_SOLVED_DISTANCE_TH "
            "must not exceed "
            "SOLVED_DISTANCE_TH."
        )

    nonnegative_values = {
        "DISTANCE_REWARD_SCALE": (
            DISTANCE_REWARD_SCALE
        ),
        "ROBUST_MAX_DISTANCE_PENALTY_SCALE": (
            ROBUST_MAX_DISTANCE_PENALTY_SCALE
        ),
        "VELOCITY_PENALTY_SCALE": (
            VELOCITY_PENALTY_SCALE
        ),
        "ACTION_PENALTY_SCALE": (
            ACTION_PENALTY_SCALE
        ),
        "SOLVED_BONUS_VALUE": (
            SOLVED_BONUS_VALUE
        ),
    }

    for (
        name,
        value,
    ) in nonnegative_values.items():
        if value < 0.0:
            raise ValueError(
                f"{name} must be "
                "non-negative."
            )

    if SCHEDULE == "adaptive":
        if not (
            0.0
            < MIN_ADAPTIVE_LR
            <= LEARNING_RATE
            <= MAX_ADAPTIVE_LR
        ):
            raise ValueError(
                "Adaptive learning rates "
                "must satisfy "
                "0 < MIN_ADAPTIVE_LR "
                "<= LEARNING_RATE "
                "<= MAX_ADAPTIVE_LR."
            )


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()