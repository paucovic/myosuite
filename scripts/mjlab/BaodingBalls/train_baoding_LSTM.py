#!/usr/bin/env python3
"""Train the MJLab Baoding static-hold task with RSL-RL PPO-LSTM.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/train_baoding_LSTM.py

Run from the repository root:

    python scripts/mjlab/BaodingBalls/train_baoding_LSTM.py \
        --policy-path my_baoding_static_lstm_policy.pt

This saves the current policy, the best policy, and JSON metrics:

    my_baoding_static_lstm_policy.pt
    my_baoding_static_lstm_policy_best.pt
    my_baoding_static_lstm_policy_metrics.json

The default configuration is a paper-inspired static-target first stage. It
uses the paper's recurrent architecture and main PPO hyperparameters while
retaining GPU-parallel MJLab rollouts. The original survival-only reward can
still be selected with REWARD_MODE = "survival" for an ablation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import register_baoding as baoding_registration

from register_baoding import (
    CTRL_DT,
    DROP_HEIGHT,
    DROP_PENALTY,
    EXPECTED_ACTION_DIM,
    HOLD_REWARD_PER_STEP,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    TASK_ID,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# BAODING ENVIRONMENT CONFIGURATION
# =============================================================================

# The original MyoSuite balls have a diameter of approximately 44 mm.
# A scale of 0.86 produces balls with a diameter of approximately 38 mm.
BALL_SIZE_SCALE = 0.86

# Both balls receive the same random horizontal translation, in metres.
BALL_PAIR_RESET_OFFSET_LOW = (-0.003, -0.003, 0.0)
BALL_PAIR_RESET_OFFSET_HIGH = (0.003, 0.003, 0.0)

# Each ball then receives a smaller independent horizontal displacement.
BALL_INDIVIDUAL_RESET_JITTER_LOW = (-0.001, -0.001, 0.0)
BALL_INDIVIDUAL_RESET_JITTER_HIGH = (0.001, 0.001, 0.0)


def apply_baoding_environment_configuration() -> None:
    """Apply this file's values to register_baoding before registration.

    This supports both register_baoding versions: the newer version with the
    public configure_baoding_balls() helper and the earlier random-reset version
    in which the configuration is stored directly in module constants.
    """
    configure_function = getattr(
        baoding_registration,
        "configure_baoding_balls",
        None,
    )

    if callable(configure_function):
        configure_function(
            ball_size_scale=BALL_SIZE_SCALE,
            pair_reset_offset_low=BALL_PAIR_RESET_OFFSET_LOW,
            pair_reset_offset_high=BALL_PAIR_RESET_OFFSET_HIGH,
            individual_reset_jitter_low=(
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            individual_reset_jitter_high=(
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
        )
        return

    required_names = (
        "ORIGINAL_BALL_NOMINAL_RADIUS",
        "BALL1_CLASSIC_INITIAL_POS",
        "BALL2_CLASSIC_INITIAL_POS",
        "BALL_CONTACT_HEIGHT_COMPENSATION",
        "BALL1_INITIAL_POS",
        "BALL2_INITIAL_POS",
        "BALL_MASS_SCALE",
        "BALL_PAIR_RESET_OFFSET_LOW",
        "BALL_PAIR_RESET_OFFSET_HIGH",
        "BALL_INDIVIDUAL_RESET_JITTER_LOW",
        "BALL_INDIVIDUAL_RESET_JITTER_HIGH",
    )

    missing_names = [
        name
        for name in required_names
        if not hasattr(baoding_registration, name)
    ]

    if missing_names:
        raise RuntimeError(
            "register_baoding.py is older than the random-reset version. "
            "Missing definitions: "
            + ", ".join(missing_names)
        )

    if getattr(
        baoding_registration,
        "_TASK_CONFIGURATION_LOCKED",
        False,
    ):
        raise RuntimeError(
            "The Baoding task was registered before its ball configuration "
            "was applied. Start a new Python process and configure it first."
        )

    if not 0.0 < BALL_SIZE_SCALE <= 1.0:
        raise ValueError("BALL_SIZE_SCALE must be in (0, 1].")

    reset_ranges = (
        (
            "BALL_PAIR_RESET_OFFSET",
            BALL_PAIR_RESET_OFFSET_LOW,
            BALL_PAIR_RESET_OFFSET_HIGH,
        ),
        (
            "BALL_INDIVIDUAL_RESET_JITTER",
            BALL_INDIVIDUAL_RESET_JITTER_LOW,
            BALL_INDIVIDUAL_RESET_JITTER_HIGH,
        ),
    )

    for name, low, high in reset_ranges:
        if len(low) != 3 or len(high) != 3:
            raise ValueError(
                f"{name} bounds must contain exactly three values."
            )

        if any(
            low_value > high_value
            for low_value, high_value in zip(low, high, strict=True)
        ):
            raise ValueError(
                f"Every lower {name} bound must be less than or equal "
                "to its upper bound."
            )

    vertical_bounds = (
        BALL_PAIR_RESET_OFFSET_LOW[2],
        BALL_PAIR_RESET_OFFSET_HIGH[2],
        BALL_INDIVIDUAL_RESET_JITTER_LOW[2],
        BALL_INDIVIDUAL_RESET_JITTER_HIGH[2],
    )

    if any(abs(value) > 1e-12 for value in vertical_bounds):
        raise ValueError(
            "All Baoding reset z offsets must remain 0.0 so the balls "
            "preserve their safe contact height."
        )

    size_scale = float(BALL_SIZE_SCALE)
    contact_height_compensation = (
        baoding_registration.ORIGINAL_BALL_NOMINAL_RADIUS
        * (1.0 - size_scale)
    )

    baoding_registration.BALL_SIZE_SCALE = size_scale
    baoding_registration.BALL_CONTACT_HEIGHT_COMPENSATION = (
        contact_height_compensation
    )
    baoding_registration.BALL1_INITIAL_POS = (
        baoding_registration.BALL1_CLASSIC_INITIAL_POS[0],
        baoding_registration.BALL1_CLASSIC_INITIAL_POS[1],
        baoding_registration.BALL1_CLASSIC_INITIAL_POS[2]
        - contact_height_compensation,
    )
    baoding_registration.BALL2_INITIAL_POS = (
        baoding_registration.BALL2_CLASSIC_INITIAL_POS[0],
        baoding_registration.BALL2_CLASSIC_INITIAL_POS[1],
        baoding_registration.BALL2_CLASSIC_INITIAL_POS[2]
        - contact_height_compensation,
    )
    baoding_registration.BALL_MASS_SCALE = size_scale ** 3
    baoding_registration.BALL_PAIR_RESET_OFFSET_LOW = tuple(
        float(value)
        for value in BALL_PAIR_RESET_OFFSET_LOW
    )
    baoding_registration.BALL_PAIR_RESET_OFFSET_HIGH = tuple(
        float(value)
        for value in BALL_PAIR_RESET_OFFSET_HIGH
    )
    baoding_registration.BALL_INDIVIDUAL_RESET_JITTER_LOW = tuple(
        float(value)
        for value in BALL_INDIVIDUAL_RESET_JITTER_LOW
    )
    baoding_registration.BALL_INDIVIDUAL_RESET_JITTER_HIGH = tuple(
        float(value)
        for value in BALL_INDIVIDUAL_RESET_JITTER_HIGH
    )


# Apply configuration before MJLab creates or registers the environment.
apply_baoding_environment_configuration()

STATIC_TARGET1_POS = tuple(
    float(value)
    for value in baoding_registration.BALL1_INITIAL_POS
)
STATIC_TARGET2_POS = tuple(
    float(value)
    for value in baoding_registration.BALL2_INITIAL_POS
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Preserve the previous number of transitions per iteration:
#
#   old: 512 environments * 64 steps  = 32768
#   new: 256 environments * 128 steps = 32768
#
# The longer sequence gives the LSTM more temporal context.
ITERATIONS = 1000
NUM_ENVS = 256
NUM_STEPS_PER_ENV = 128

RNN_TYPE = "lstm"
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 1
POST_RNN_HIDDEN_DIMS = (256, 256)

NUM_EVALS = 50
EVAL_NUM_ENVS = 64
EVAL_STEPS = MAX_EPISODE_STEPS

# PPO settings from the released first curriculum stage.
LEARNING_RATE = 2.55673e-5
ADAM_EPSILON = 1e-5

MIN_ADAPTIVE_LR = 1e-6
MAX_ADAPTIVE_LR = 3e-4

SCHEDULE = "fixed"
NUM_LEARNING_EPOCHS = 8
NUM_MINI_BATCHES = 4

CLIP_PARAM = 0.3
DESIRED_KL = 0.01
MAX_GRAD_NORM = 0.7

ENTROPY_COEF = 3.62109e-6
VALUE_LOSS_COEF = 0.835671
USE_CLIPPED_VALUE_LOSS = False

GAMMA = 0.99
GAE_LAMBDA = 0.9
NORMALIZE_ADVANTAGE_PER_MINIBATCH = True

MODEL_ACTIVATION = "relu"
INITIAL_ACTION_STD = math.exp(-2.0)
ACTION_STD_TYPE = "log"

# Clip Gaussian actions before the MyoSuite muscle transformation.
ACTION_CLIP = 1.0

# Reward modes:
#
# "paper_static":
#   -distance(ball1, target1)
#   -distance(ball2, target2)
#   +1 while alive
#   +5 while both balls are within 15 mm of their targets
#
# "survival":
#   Use the original registered +1 / -10 survival reward.
REWARD_MODE = "paper_static"

STATIC_POSITION_WEIGHT = 1.0
STATIC_ALIVE_WEIGHT = 1.0
STATIC_SOLVED_WEIGHT = 5.0
STATIC_PROXIMITY_THRESHOLD = 0.015

# VecNormalize-style discounted-return reward normalization. This applies only
# during training. Evaluation always reports raw task rewards.
NORMALIZE_TRAINING_REWARDS = True
REWARD_NORMALIZATION_GAMMA = GAMMA
REWARD_NORMALIZATION_CLIP = 10.0
REWARD_NORMALIZATION_EPSILON = 1e-8

# RSL-RL normally calculates KL only for its adaptive schedule. The metrics
# wrapper temporarily enables that calculation but forces the optimizer back
# to the fixed learning rate before every optimizer step.
LOG_KL_WITH_FIXED_SCHEDULE = True

BEST_POLICY_METRIC = "eval/mean_hold_steps"


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

        baoding_registration.register_baoding_hold_task(
            quiet=True
        )

    return make_env, registry.load_rl_cfg


# =============================================================================
# MJLAB TO RSL-RL WRAPPER
# =============================================================================

class RunningMeanStd:
    """Numerically stable running scalar moments."""

    def __init__(
        self,
        device: torch.device | str,
        epsilon: float = 1e-4,
    ) -> None:
        self.mean = torch.zeros(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.var = torch.ones(
            (),
            device=device,
            dtype=torch.float64,
        )
        self.count = torch.tensor(
            epsilon,
            device=device,
            dtype=torch.float64,
        )

    def update(
        self,
        values: torch.Tensor,
    ) -> None:
        batch = (
            values.detach()
            .to(dtype=torch.float64)
            .reshape(-1)
        )

        if batch.numel() == 0:
            return

        batch_mean = batch.mean()
        batch_var = batch.var(unbiased=False)
        batch_count = torch.tensor(
            float(batch.numel()),
            device=batch.device,
            dtype=torch.float64,
        )

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        old_second_moment = (
            self.var * self.count
        )
        batch_second_moment = (
            batch_var * batch_count
        )
        correction = (
            delta.square()
            * self.count
            * batch_count
            / total_count
        )

        new_var = (
            old_second_moment
            + batch_second_moment
            + correction
        ) / total_count

        self.mean = new_mean
        self.var = new_var.clamp_min(0.0)
        self.count = total_count


class BaodingRslRlWrapper:
    """Adapt MJLab to RSL-RL and enforce the task contract."""

    def __init__(
        self,
        env: Any,
        *,
        normalize_rewards: bool,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(
            getattr(env, "num_envs", 1)
        )
        self.num_actions = int(
            env.action_space.shape[-1]
        )
        self.cfg = getattr(env, "cfg", None)
        self.normalize_rewards = bool(
            normalize_rewards
        )

        if self.num_actions != EXPECTED_ACTION_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_ACTION_DIM} actions, "
                f"but the environment exposes "
                f"{self.num_actions}."
            )

        env_origins = getattr(
            getattr(env, "scene", None),
            "env_origins",
            None,
        )

        if not isinstance(
            env_origins,
            torch.Tensor,
        ):
            raise RuntimeError(
                "MJLab did not expose scene.env_origins. "
                "It is required to convert world-frame "
                "ball positions to local positions."
            )

        expected_origins_shape = (
            self.num_envs,
            3,
        )

        if (
            tuple(env_origins.shape)
            != expected_origins_shape
        ):
            raise RuntimeError(
                "Expected scene.env_origins shape "
                f"{expected_origins_shape}, got "
                f"{tuple(env_origins.shape)}."
            )

        self._env_origins = (
            env_origins.to(
                device=self.device,
                dtype=torch.float32,
            )
            .detach()
            .clone()
        )

        self._target1 = torch.tensor(
            STATIC_TARGET1_POS,
            device=self.device,
            dtype=torch.float32,
        )
        self._target2 = torch.tensor(
            STATIC_TARGET2_POS,
            device=self.device,
            dtype=torch.float32,
        )

        self._return_rms = RunningMeanStd(
            device=self.device
        )
        self._discounted_returns = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        self._episode_returns = torch.zeros_like(
            self._discounted_returns
        )
        self._episode_lengths = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        self._last_policy_obs: (
            torch.Tensor | None
        ) = None

        self._reset_interval_diagnostics()

    def __getattr__(
        self,
        name: str,
    ):
        return getattr(self.env, name)

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
            try:
                self.env.reset(seed=seed)
            except TypeError:
                self.env.reset()
        else:
            self.env.reset()

        self._discounted_returns.zero_()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()

        obs = self.get_observations()
        self._last_policy_obs = (
            obs["policy"].detach().clone()
        )
        return obs

    def get_observations(self) -> TensorDict:
        obs_dict = (
            self.env.observation_manager.compute()
        )
        return self._build_local_observation(
            obs_dict
        )

    def step(
        self,
        actions: torch.Tensor,
    ):
        actions = actions.to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        expected_action_shape = (
            self.num_envs,
            EXPECTED_ACTION_DIM,
        )

        if (
            tuple(actions.shape)
            != expected_action_shape
        ):
            raise RuntimeError(
                f"Expected action shape "
                f"{expected_action_shape}, got "
                f"{tuple(actions.shape)}."
            )

        if not torch.isfinite(actions).all():
            raise RuntimeError(
                "The PPO actor produced an action "
                "containing NaN or infinity."
            )

        raw_actions = actions
        clipped_actions = torch.clamp(
            raw_actions,
            -ACTION_CLIP,
            ACTION_CLIP,
        )

        self._record_action_diagnostics(
            raw_actions=raw_actions,
            clipped_actions=clipped_actions,
        )

        (
            obs_dict,
            native_reward,
            terminated,
            truncated,
            extras,
        ) = self.env.step(clipped_actions)

        native_reward = native_reward.reshape(
            self.num_envs
        )
        terminated = terminated.reshape(
            self.num_envs
        ).bool()
        truncated = truncated.reshape(
            self.num_envs
        ).bool()

        done_mask = terminated | truncated
        dones = done_mask.to(dtype=torch.long)

        obs = self._build_local_observation(
            obs_dict
        )
        policy_obs = obs["policy"]
        reward_obs = policy_obs

        # MJLab returns post-reset observations for completed
        # environments. Use the previous observation for their
        # terminal reward instead of scoring the next reset.
        if self._last_policy_obs is not None:
            reward_obs = policy_obs.clone()
            reward_obs[done_mask] = (
                self._last_policy_obs[done_mask]
            )

        reward_components = (
            self._compute_reward_components(
                policy_obs=reward_obs,
                terminated=terminated,
            )
        )

        if REWARD_MODE == "paper_static":
            raw_task_reward = reward_components[
                "paper_static_reward"
            ]
        elif REWARD_MODE == "survival":
            raw_task_reward = native_reward
        else:
            raise RuntimeError(
                f"Unsupported REWARD_MODE: "
                f"{REWARD_MODE!r}"
            )

        if not torch.isfinite(
            raw_task_reward
        ).all():
            raise RuntimeError(
                "The Baoding task reward contains "
                "NaN or infinity."
            )

        rewards = self._normalize_reward(
            raw_rewards=raw_task_reward,
            done_mask=done_mask,
        )

        both_held = (~terminated).float()
        ball_dropped = terminated.float()

        self._record_episode_diagnostics(
            raw_task_reward=raw_task_reward,
            terminated=terminated,
            truncated=truncated,
            done_mask=done_mask,
        )

        self._last_policy_obs = (
            policy_obs.detach().clone()
        )

        infos: dict[str, Any] = {
            "baoding": {
                "reward": (
                    raw_task_reward.detach()
                ),
                "normalized_reward": (
                    rewards.detach()
                ),
                "native_survival_reward": (
                    native_reward.detach()
                ),
                "both_held": (
                    both_held.detach()
                ),
                "ball_dropped": (
                    ball_dropped.detach()
                ),
                "terminated": (
                    terminated.detach()
                ),
                "truncated": (
                    truncated.detach()
                ),
                "target1_distance": (
                    reward_components[
                        "target1_distance"
                    ].detach()
                ),
                "target2_distance": (
                    reward_components[
                        "target2_distance"
                    ].detach()
                ),
                "static_solved": (
                    reward_components[
                        "static_solved"
                    ].detach()
                ),
            },
            "episode": {
                "reward": (
                    raw_task_reward.detach().mean()
                ),
                "both_held": (
                    both_held.detach().mean()
                ),
                "ball_dropped": (
                    ball_dropped.detach().mean()
                ),
                "static_solved": (
                    reward_components[
                        "static_solved"
                    ]
                    .detach()
                    .mean()
                ),
            },
        }

        if isinstance(extras, dict):
            if "time_outs" in extras:
                infos["time_outs"] = (
                    extras["time_outs"]
                )
            else:
                infos["time_outs"] = truncated

            if "log" in extras:
                infos["log"] = extras["log"]
        else:
            infos["time_outs"] = truncated

        return obs, rewards, dones, infos

    def pop_training_diagnostics(
        self,
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self._action_value_count > 0:
            metrics.update(
                {
                    "train/action_clip_fraction": (
                        float(
                            self
                            ._action_clipped_count
                            .cpu()
                        )
                        / self._action_value_count
                    ),
                    "train/action_abs_mean_pre_clip": (
                        float(
                            self
                            ._action_abs_sum
                            .cpu()
                        )
                        / self._action_value_count
                    ),
                    "train/action_abs_max_pre_clip": (
                        float(
                            self
                            ._action_abs_max
                            .cpu()
                        )
                    ),
                }
            )

        completed_count = float(
            self._completed_episode_count.cpu()
        )

        if completed_count > 0:
            metrics.update(
                {
                    "train/completed_episodes": (
                        completed_count
                    ),
                    "train/episode_return_raw": (
                        float(
                            self
                            ._completed_return_sum
                            .cpu()
                        )
                        / completed_count
                    ),
                    "train/episode_length": (
                        float(
                            self
                            ._completed_length_sum
                            .cpu()
                        )
                        / completed_count
                    ),
                    "train/drop_fraction": (
                        float(
                            self
                            ._completed_drop_count
                            .cpu()
                        )
                        / completed_count
                    ),
                    "train/time_limit_fraction": (
                        float(
                            self
                            ._completed_timeout_count
                            .cpu()
                        )
                        / completed_count
                    ),
                }
            )

        metrics.update(
            {
                "train/reward_rms_mean": float(
                    self._return_rms.mean
                    .detach()
                    .cpu()
                ),
                "train/reward_rms_var": float(
                    self._return_rms.var
                    .detach()
                    .cpu()
                ),
                "train/reward_rms_std": float(
                    torch.sqrt(
                        self._return_rms.var
                    )
                    .detach()
                    .cpu()
                ),
                "train/reward_rms_count": float(
                    self._return_rms.count
                    .detach()
                    .cpu()
                ),
            }
        )

        self._reset_interval_diagnostics()
        return metrics

    def checkpoint_infos(
        self,
    ) -> dict[str, Any]:
        """Return non-model state needed to resume training."""
        return {
            "reward_mode": REWARD_MODE,
            "action_clip": ACTION_CLIP,
            "reward_normalizer": {
                "enabled": self.normalize_rewards,
                "mean": float(
                    self._return_rms.mean
                    .detach()
                    .cpu()
                ),
                "var": float(
                    self._return_rms.var
                    .detach()
                    .cpu()
                ),
                "count": float(
                    self._return_rms.count
                    .detach()
                    .cpu()
                ),
            },
        }

    def _compute_reward_components(
        self,
        *,
        policy_obs: torch.Tensor,
        terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ball1_pos = policy_obs[:, 46:49]
        ball2_pos = policy_obs[:, 52:55]

        target1_distance = (
            torch.linalg.vector_norm(
                ball1_pos - self._target1,
                dim=-1,
            )
        )
        target2_distance = (
            torch.linalg.vector_norm(
                ball2_pos - self._target2,
                dim=-1,
            )
        )

        alive = (~terminated).float()

        static_solved = (
            (~terminated)
            & (
                target1_distance
                < STATIC_PROXIMITY_THRESHOLD
            )
            & (
                target2_distance
                < STATIC_PROXIMITY_THRESHOLD
            )
        ).float()

        paper_static_reward = (
            -STATIC_POSITION_WEIGHT
            * target1_distance
            -STATIC_POSITION_WEIGHT
            * target2_distance
            +STATIC_ALIVE_WEIGHT
            * alive
            +STATIC_SOLVED_WEIGHT
            * static_solved
        )

        return {
            "target1_distance": (
                target1_distance
            ),
            "target2_distance": (
                target2_distance
            ),
            "static_solved": static_solved,
            "paper_static_reward": (
                paper_static_reward
            ),
        }

    def _build_local_observation(
        self,
        obs_dict: Any,
    ) -> TensorDict:
        """Convert world-frame ball positions to local positions."""
        obs = TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
        )
        policy_obs = obs["policy"].clone()

        policy_obs[:, 46:49] -= (
            self._env_origins
        )
        policy_obs[:, 52:55] -= (
            self._env_origins
        )

        obs["policy"] = policy_obs

        self._validate_observation(
            policy_obs
        )
        return obs

    def _normalize_reward(
        self,
        *,
        raw_rewards: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.normalize_rewards:
            return raw_rewards

        self._discounted_returns.mul_(
            REWARD_NORMALIZATION_GAMMA
        )
        self._discounted_returns.add_(
            raw_rewards
        )

        self._return_rms.update(
            self._discounted_returns
        )

        normalized = (
            raw_rewards
            / torch.sqrt(
                self._return_rms.var.to(
                    dtype=raw_rewards.dtype
                )
                + REWARD_NORMALIZATION_EPSILON
            )
        )

        normalized = torch.clamp(
            normalized,
            -REWARD_NORMALIZATION_CLIP,
            REWARD_NORMALIZATION_CLIP,
        )

        self._discounted_returns[
            done_mask
        ] = 0.0

        return normalized

    def _record_action_diagnostics(
        self,
        *,
        raw_actions: torch.Tensor,
        clipped_actions: torch.Tensor,
    ) -> None:
        raw_detached = raw_actions.detach()
        clipped_detached = (
            clipped_actions.detach()
        )
        abs_actions = raw_detached.abs()

        self._action_value_count += int(
            raw_detached.numel()
        )

        self._action_clipped_count.add_(
            torch.count_nonzero(
                raw_detached
                != clipped_detached
            )
        )
        self._action_abs_sum.add_(
            abs_actions.sum()
        )
        self._action_abs_max.copy_(
            torch.maximum(
                self._action_abs_max,
                abs_actions.max(),
            )
        )

    def _record_episode_diagnostics(
        self,
        *,
        raw_task_reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> None:
        self._episode_returns.add_(
            raw_task_reward
        )
        self._episode_lengths.add_(1)

        completed_returns = (
            self._episode_returns[done_mask]
        )
        completed_lengths = (
            self._episode_lengths[done_mask]
        )

        self._completed_episode_count.add_(
            done_mask.sum()
        )
        self._completed_return_sum.add_(
            completed_returns.sum()
        )
        self._completed_length_sum.add_(
            completed_lengths.sum()
        )
        self._completed_drop_count.add_(
            terminated[done_mask].sum()
        )
        self._completed_timeout_count.add_(
            (
                truncated
                & (~terminated)
            )[done_mask].sum()
        )

        self._episode_returns[
            done_mask
        ] = 0.0
        self._episode_lengths[
            done_mask
        ] = 0

    def _reset_interval_diagnostics(
        self,
    ) -> None:
        self._action_value_count = 0

        self._action_clipped_count = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.long,
            )
        )
        self._action_abs_sum = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )
        self._action_abs_max = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float32,
        )

        self._completed_episode_count = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.long,
            )
        )
        self._completed_return_sum = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.float64,
            )
        )
        self._completed_length_sum = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.float64,
            )
        )
        self._completed_drop_count = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.long,
            )
        )
        self._completed_timeout_count = (
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.long,
            )
        )

    def _validate_observation(
        self,
        policy_obs: torch.Tensor,
    ) -> None:
        expected_shape = (
            self.num_envs,
            OBS_DIM,
        )

        if (
            tuple(policy_obs.shape)
            != expected_shape
        ):
            raise RuntimeError(
                f"Expected policy observation "
                f"shape {expected_shape}, got "
                f"{tuple(policy_obs.shape)}."
            )

        if not torch.isfinite(
            policy_obs
        ).all():
            raise RuntimeError(
                "The Baoding policy observation "
                "contains NaN or infinity."
            )


# =============================================================================
# RSL-RL CONFIGURATION
# =============================================================================

def configure_lstm_models(
    train_cfg: dict[str, Any],
) -> None:
    """Configure the recurrent actor and critic."""
    for section in ("actor", "critic"):
        section_cfg = train_cfg.get(
            section
        )

        if not isinstance(
            section_cfg,
            dict,
        ):
            raise RuntimeError(
                f"RSL-RL {section!r} model "
                "configuration was not found."
            )

        section_cfg.pop("cnn_cfg", None)

        section_cfg["class_name"] = (
            "RNNModel"
        )
        section_cfg["hidden_dims"] = list(
            POST_RNN_HIDDEN_DIMS
        )
        section_cfg["activation"] = (
            MODEL_ACTIVATION
        )
        section_cfg["obs_normalization"] = (
            True
        )
        section_cfg["rnn_type"] = (
            RNN_TYPE
        )
        section_cfg["rnn_hidden_dim"] = (
            RNN_HIDDEN_DIM
        )
        section_cfg["rnn_num_layers"] = (
            RNN_NUM_LAYERS
        )

        if section == "actor":
            distribution_cfg = (
                section_cfg.get(
                    "distribution_cfg"
                )
            )

            if not isinstance(
                distribution_cfg,
                dict,
            ):
                raise RuntimeError(
                    "The RSL-RL actor "
                    "distribution configuration "
                    "was not found."
                )

            distribution_cfg[
                "class_name"
            ] = "GaussianDistribution"
            distribution_cfg[
                "init_std"
            ] = INITIAL_ACTION_STD
            distribution_cfg[
                "std_type"
            ] = ACTION_STD_TYPE


def validate_lstm_runner(
    runner,
) -> tuple[Any, Any]:
    """Verify the constructed recurrent models."""
    algorithm = getattr(
        runner,
        "alg",
        None,
    )

    if algorithm is None:
        raise RuntimeError(
            "OnPolicyRunner did not expose "
            "runner.alg."
        )

    actor = getattr(
        algorithm,
        "actor",
        None,
    )
    critic = getattr(
        algorithm,
        "critic",
        None,
    )

    if actor is None or critic is None:
        raise RuntimeError(
            "RSL-RL did not expose separate "
            "actor and critic models."
        )

    for name, model in (
        ("actor", actor),
        ("critic", critic),
    ):
        if not bool(
            getattr(
                model,
                "is_recurrent",
                False,
            )
        ):
            raise RuntimeError(
                f"The constructed {name} is "
                f"{type(model).__name__}, not "
                "a recurrent RNNModel."
            )

        model_rnn = getattr(
            model,
            "rnn",
            None,
        )
        torch_rnn = getattr(
            model_rnn,
            "rnn",
            None,
        )

        if not isinstance(
            torch_rnn,
            torch.nn.LSTM,
        ):
            raise RuntimeError(
                f"The constructed {name} "
                "recurrent module is "
                f"{type(torch_rnn).__name__}, "
                "not torch.nn.LSTM."
            )

        if not bool(
            getattr(
                model,
                "obs_normalization",
                False,
            )
        ):
            raise RuntimeError(
                "Observation normalization "
                f"is disabled for the {name}."
            )

    distribution = getattr(
        actor,
        "distribution",
        None,
    )

    if distribution is None:
        raise RuntimeError(
            "The recurrent actor has no "
            "action distribution."
        )

    effective_std_type = getattr(
        distribution,
        "std_type",
        None,
    )

    if (
        effective_std_type
        != ACTION_STD_TYPE
    ):
        raise RuntimeError(
            "Expected actor std_type="
            f"{ACTION_STD_TYPE!r}, got "
            f"{effective_std_type!r}."
        )

    log_std_parameter = getattr(
        distribution,
        "log_std_param",
        None,
    )

    if not isinstance(
        log_std_parameter,
        torch.Tensor,
    ):
        raise RuntimeError(
            "The actor is not using a "
            "learnable log-standard-deviation "
            "parameter."
        )

    initial_std = torch.exp(
        log_std_parameter.detach()
    )
    expected_std = torch.full_like(
        initial_std,
        INITIAL_ACTION_STD,
    )

    if not torch.allclose(
        initial_std,
        expected_std,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError(
            "The actor's initial standard "
            "deviation does not equal "
            f"{INITIAL_ACTION_STD}."
        )

    return actor, critic


def validate_effective_ppo_config(
    algorithm,
    optimizer,
) -> None:
    expected_values: dict[
        str,
        Any,
    ] = {
        "clip_param": CLIP_PARAM,
        "desired_kl": DESIRED_KL,
        "max_grad_norm": MAX_GRAD_NORM,
        "entropy_coef": ENTROPY_COEF,
        "value_loss_coef": VALUE_LOSS_COEF,
        "gamma": GAMMA,
        "lam": GAE_LAMBDA,
        "num_learning_epochs": (
            NUM_LEARNING_EPOCHS
        ),
        "num_mini_batches": (
            NUM_MINI_BATCHES
        ),
        "use_clipped_value_loss": (
            USE_CLIPPED_VALUE_LOSS
        ),
        "normalize_advantage_per_mini_batch": (
            NORMALIZE_ADVANTAGE_PER_MINIBATCH
        ),
    }

    for (
        name,
        expected,
    ) in expected_values.items():
        actual = getattr(
            algorithm,
            name,
            None,
        )

        if isinstance(expected, float):
            matches = (
                actual is not None
                and math.isclose(
                    float(actual),
                    expected,
                    rel_tol=1e-7,
                    abs_tol=1e-12,
                )
            )
        else:
            matches = actual == expected

        if not matches:
            raise RuntimeError(
                f"Expected PPO {name}="
                f"{expected!r}, got "
                f"{actual!r}."
            )

    for parameter_group in (
        optimizer.param_groups
    ):
        if not math.isclose(
            float(parameter_group["eps"]),
            ADAM_EPSILON,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "The Adam optimizer does not "
                "use the requested epsilon "
                f"{ADAM_EPSILON}."
            )


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
    train_cfg["num_steps_per_env"] = (
        NUM_STEPS_PER_ENV
    )
    train_cfg["logger"] = "tensorboard"

    set_training_value(
        train_cfg,
        ("learning_rate", "lr"),
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
        ("clip_param", "clip_range"),
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
    set_training_value(
        train_cfg,
        (
            "value_loss_coef",
            "value_coef",
            "vf_coef",
        ),
        VALUE_LOSS_COEF,
    )
    set_training_value(
        train_cfg,
        ("use_clipped_value_loss",),
        USE_CLIPPED_VALUE_LOSS,
    )
    set_training_value(
        train_cfg,
        ("gamma",),
        GAMMA,
    )
    set_training_value(
        train_cfg,
        ("lam", "gae_lambda"),
        GAE_LAMBDA,
    )
    set_training_value(
        train_cfg,
        (
            "normalize_advantage_per_mini_batch",
        ),
        NORMALIZE_ADVANTAGE_PER_MINIBATCH,
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
        or len(optimizer.param_groups) == 0
    ):
        raise RuntimeError(
            "RSL-RL PPO optimizer was not found."
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


def best_policy_path(
    policy_path: Path,
) -> Path:
    return policy_path.with_name(
        f"{policy_path.stem}_best"
        f"{policy_path.suffix}"
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
        "surrogate_loss": (
            "train/surrogate"
        ),
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


def find_kl_source(
    runner,
):
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
    storage: dict[
        str,
        list[Any],
    ] = {
        "updates": [],
        "kl_values": [],
        "grad_norm_values": [],
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

    if optimizer is not None:
        if not hasattr(
            optimizer,
            "register_step_pre_hook",
        ):
            raise RuntimeError(
                "This PyTorch version does "
                "not support optimizer "
                "step pre-hooks."
            )

        model_parameters = (
            list(
                algorithm.actor.parameters()
            )
            + list(
                algorithm.critic.parameters()
            )
        )

        def capture_post_clip_gradient_norm(
            _optimizer,
            _args,
            _kwargs,
        ) -> None:
            squared_norm = torch.zeros(
                (),
                device=algorithm.device,
                dtype=torch.float32,
            )

            for parameter in model_parameters:
                if parameter.grad is not None:
                    squared_norm.add_(
                        parameter.grad
                        .detach()
                        .float()
                        .square()
                        .sum()
                    )

            storage[
                "grad_norm_values"
            ].append(
                float(
                    torch.sqrt(
                        squared_norm
                    ).cpu()
                )
            )

        optimizer.register_step_pre_hook(
            capture_post_clip_gradient_norm
        )

    if (
        LOG_KL_WITH_FIXED_SCHEDULE
        and SCHEDULE == "fixed"
        and optimizer is not None
        and len(optimizer.param_groups) > 0
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

        def fixed_lr_pre_step_hook(
            _optimizer,
            _args,
            _kwargs,
        ) -> None:
            force_fixed_lr()

        optimizer.register_step_pre_hook(
            fixed_lr_pre_step_hook
        )

    kl_source = find_kl_source(
        runner
    )

    if kl_source is not None:
        original_kl_function = (
            kl_source.get_kl_divergence
        )

        def wrapped_kl_function(
            *args,
            **kwargs,
        ):
            kl_value = original_kl_function(
                *args,
                **kwargs,
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
        grad_start = len(
            storage["grad_norm_values"]
        )
        previous_schedule = getattr(
            algorithm,
            "schedule",
            None,
        )
        explained_variance = None

        rollout_storage = getattr(
            algorithm,
            "storage",
            None,
        )

        if rollout_storage is not None:
            returns = getattr(
                rollout_storage,
                "returns",
                None,
            )
            values = getattr(
                rollout_storage,
                "values",
                None,
            )

            if (
                isinstance(
                    returns,
                    torch.Tensor,
                )
                and isinstance(
                    values,
                    torch.Tensor,
                )
                and returns.numel() > 1
            ):
                with torch.no_grad():
                    target_variance = (
                        torch.var(
                            returns.float(),
                            unbiased=False,
                        )
                    )

                    if (
                        float(
                            target_variance.cpu()
                        )
                        > 1e-12
                    ):
                        residual_variance = (
                            torch.var(
                                (
                                    returns
                                    - values
                                ).float(),
                                unbiased=False,
                            )
                        )
                        explained_variance = (
                            float(
                                (
                                    1.0
                                    - residual_variance
                                    / target_variance
                                ).cpu()
                            )
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

        if explained_variance is not None:
            record[
                "train/explained_variance"
            ] = explained_variance

        if (
            optimizer is not None
            and len(
                optimizer.param_groups
            ) > 0
        ):
            record[
                "train/learning_rate"
            ] = float(
                optimizer.param_groups[
                    0
                ]["lr"]
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
                value = to_float(
                    raw_value
                )

                if (
                    clean_name is not None
                    and value is not None
                ):
                    record[
                        clean_name
                    ] = value

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
                value = to_float(
                    raw_value
                )

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

        new_grad_norm_values = storage[
            "grad_norm_values"
        ][grad_start:]

        if len(
            new_grad_norm_values
        ) > 0:
            record[
                "train/grad_norm_after_clip"
            ] = float(
                sum(new_grad_norm_values)
                / len(
                    new_grad_norm_values
                )
            )
            record[
                "train/grad_norm_after_clip_max"
            ] = float(
                max(new_grad_norm_values)
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
            "train/num_updates": 0.0
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


def summarize_samples(
    values: list[Any],
    *,
    prefix: str,
) -> dict[str, float]:
    clean_values: list[float] = []

    for value in values:
        scalar = to_float(value)

        if scalar is not None:
            clean_values.append(scalar)

    if len(clean_values) == 0:
        return {}

    samples = torch.tensor(
        clean_values,
        dtype=torch.float64,
    )

    return {
        f"{prefix}_mean": float(
            samples.mean()
        ),
        f"{prefix}_median": float(
            samples.median()
        ),
        f"{prefix}_p95": float(
            torch.quantile(
                samples,
                0.95,
            )
        ),
        f"{prefix}_max": float(
            samples.max()
        ),
    }


def actor_std_metrics(
    actor,
) -> dict[str, float]:
    try:
        std = (
            actor.output_std
            .detach()
            .float()
            .reshape(-1)
        )
    except Exception:
        return {}

    if std.numel() == 0:
        return {}

    return {
        "train/action_std_mean": float(
            std.mean().cpu()
        ),
        "train/action_std_min": float(
            std.min().cpu()
        ),
        "train/action_std_max": float(
            std.max().cpu()
        ),
    }


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
        "Unsupported action output type: "
        f"{type(action)}"
    )


def clone_hidden_state(
    hidden_state: Any,
) -> Any:
    """Clone an RNN state without retaining its graph."""
    if hidden_state is None:
        return None

    if isinstance(
        hidden_state,
        torch.Tensor,
    ):
        return (
            hidden_state.detach().clone()
        )

    if isinstance(
        hidden_state,
        tuple,
    ):
        return tuple(
            clone_hidden_state(value)
            for value in hidden_state
        )

    raise TypeError(
        "Unsupported recurrent hidden-state "
        f"type: {type(hidden_state)}"
    )


@contextlib.contextmanager
def preserve_global_rng_state():
    """Prevent evaluation from changing training RNG streams."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_cpu_state = (
        torch.random.get_rng_state()
    )

    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    previous_python_hash_seed = (
        os.environ.get("PYTHONHASHSEED")
    )

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(
            torch_cpu_state
        )

        if cuda_states is not None:
            torch.cuda.set_rng_state_all(
                cuda_states
            )

        if previous_python_hash_seed is None:
            os.environ.pop(
                "PYTHONHASHSEED",
                None,
            )
        else:
            os.environ[
                "PYTHONHASHSEED"
            ] = previous_python_hash_seed


def patch_recurrent_critic_bootstrap(
    algorithm,
) -> None:
    """Prevent a boundary observation from advancing critic memory.

    The bootstrap value is computed, but any persistent recurrent-state
    mutation is discarded so the next rollout processes the boundary
    observation exactly once.
    """
    critic = getattr(
        algorithm,
        "critic",
        None,
    )
    original_compute_returns = getattr(
        algorithm,
        "compute_returns",
        None,
    )

    if (
        critic is None
        or not bool(
            getattr(
                critic,
                "is_recurrent",
                False,
            )
        )
        or not callable(
            original_compute_returns
        )
    ):
        raise RuntimeError(
            "A recurrent critic with "
            "compute_returns() is required."
        )

    def compute_returns_without_state_advance(
        obs: TensorDict,
    ) -> None:
        critic_state = clone_hidden_state(
            critic.get_hidden_state()
        )

        try:
            original_compute_returns(obs)
        finally:
            critic.reset(
                hidden_state=critic_state
            )

    algorithm.compute_returns = (
        compute_returns_without_state_advance
    )


def get_deterministic_action(
    policy,
    obs: TensorDict,
) -> torch.Tensor:
    """Run one deterministic recurrent inference step."""
    with torch.no_grad():
        return unpack_action(
            policy(obs)
        )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_policy(
    runner,
    eval_env: BaodingRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    """Evaluate without perturbing training RNG streams."""
    with preserve_global_rng_state():
        return _evaluate_policy_once(
            runner=runner,
            eval_env=eval_env,
            seed=seed,
            device=device,
            quiet=quiet,
        )


def _evaluate_policy_once(
    runner,
    eval_env: BaodingRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    """Evaluate first episodes while isolating recurrent state."""
    with redirect_output(quiet):
        obs = eval_env.reset(seed=seed)

    algorithm = getattr(
        runner,
        "alg",
        None,
    )

    if algorithm is None:
        raise RuntimeError(
            "OnPolicyRunner did not expose "
            "runner.alg."
        )

    actor = getattr(
        algorithm,
        "actor",
        None,
    )

    if actor is None:
        raise RuntimeError(
            "RSL-RL PPO actor was not found."
        )

    training_hidden_state = (
        clone_hidden_state(
            actor.get_hidden_state()
        )
    )
    actor_was_training = bool(
        actor.training
    )

    policy = runner.get_inference_policy(
        device=device
    )
    policy.reset()

    episode_returns = torch.zeros(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.float32,
    )
    episode_lengths = torch.zeros(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.long,
    )
    held_steps = torch.zeros(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.long,
    )
    static_solved_steps = (
        torch.zeros_like(held_steps)
    )

    target1_distance_sum = torch.zeros(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.float32,
    )
    target2_distance_sum = (
        torch.zeros_like(
            target1_distance_sum
        )
    )

    active = torch.ones(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.bool,
    )
    dropped = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)

    try:
        for _ in range(EVAL_STEPS):
            actions = (
                get_deterministic_action(
                    policy=policy,
                    obs=obs,
                )
            )

            (
                obs,
                reward,
                done,
                infos,
            ) = eval_env.step(actions)

            baoding_info = infos[
                "baoding"
            ]
            terminated = (
                baoding_info[
                    "terminated"
                ].bool()
            )
            truncated = (
                baoding_info[
                    "truncated"
                ].bool()
            )

            policy.reset(done)

            active_before_step = (
                active.clone()
            )

            episode_returns[
                active_before_step
            ] += reward[
                active_before_step
            ]
            episode_lengths[
                active_before_step
            ] += 1

            held_this_step = (
                active_before_step
                & (~terminated)
            )
            held_steps[
                held_this_step
            ] += 1

            static_solved_steps[
                active_before_step
            ] += (
                baoding_info[
                    "static_solved"
                ][active_before_step]
                .long()
            )

            target1_distance_sum[
                active_before_step
            ] += (
                baoding_info[
                    "target1_distance"
                ][active_before_step]
            )
            target2_distance_sum[
                active_before_step
            ] += (
                baoding_info[
                    "target2_distance"
                ][active_before_step]
            )

            done_this_step = (
                active_before_step
                & (
                    terminated
                    | truncated
                )
            )

            dropped[
                done_this_step
                & terminated
            ] = True
            timed_out[
                done_this_step
                & truncated
                & (~terminated)
            ] = True

            active = (
                active_before_step
                & (~done_this_step)
            )

            if not bool(active.any()):
                break

    finally:
        policy.reset(
            hidden_state=(
                training_hidden_state
            )
        )

        if actor_was_training:
            algorithm.train_mode()

    completed = ~active

    full_hold = (
        (held_steps >= EVAL_STEPS)
        & (~dropped)
    )

    episode_lengths_float = (
        episode_lengths.float()
    )
    held_steps_float = (
        held_steps.float()
    )
    static_solved_steps_float = (
        static_solved_steps.float()
    )

    return_per_executed_step = (
        episode_returns
        / episode_lengths_float.clamp_min(
            1.0
        )
    )

    metrics = {
        "eval/mean_reward": float(
            episode_returns.mean().cpu()
        ),
        "eval/reward_per_step": float(
            return_per_executed_step
            .mean()
            .cpu()
        ),
        "eval/reward_std": float(
            episode_returns.std(
                unbiased=False
            ).cpu()
        ),
        "eval/mean_episode_length": float(
            episode_lengths_float
            .mean()
            .cpu()
        ),
        "eval/mean_hold_steps": float(
            held_steps_float.mean().cpu()
        ),
        "eval/mean_hold_time_sec": float(
            (
                held_steps_float
                * CTRL_DT
            ).mean().cpu()
        ),
        "eval/mean_hold_fraction": float(
            (
                held_steps_float
                / EVAL_STEPS
            ).mean().cpu()
        ),
        "eval/mean_static_solved_fraction": float(
            (
                static_solved_steps_float
                / episode_lengths_float
                .clamp_min(1.0)
            ).mean().cpu()
        ),
        "eval/mean_target1_distance": float(
            (
                target1_distance_sum
                / episode_lengths_float
                .clamp_min(1.0)
            ).mean().cpu()
        ),
        "eval/mean_target2_distance": float(
            (
                target2_distance_sum
                / episode_lengths_float
                .clamp_min(1.0)
            ).mean().cpu()
        ),
        "eval/mean_total_target_distance": float(
            (
                (
                    target1_distance_sum
                    + target2_distance_sum
                )
                / episode_lengths_float
                .clamp_min(1.0)
            ).mean().cpu()
        ),
        "eval/full_hold_fraction": float(
            full_hold.float().mean().cpu()
        ),
        "eval/drop_fraction": float(
            dropped.float().mean().cpu()
        ),
        "eval/time_limit_fraction": float(
            timed_out.float().mean().cpu()
        ),
        "eval/completed_fraction": float(
            completed.float().mean().cpu()
        ),
    }

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
        "elapsed_sec": float(
            elapsed_sec
        ),
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
    normalize_rewards: bool,
) -> BaodingRslRlWrapper:
    with redirect_output(quiet):
        env = make_env(
            env_id,
            backend="mjlab",
            device=device,
            num_envs=num_envs,
        )

    wrapper = BaodingRslRlWrapper(
        env=env,
        normalize_rewards=(
            normalize_rewards
        ),
    )

    with redirect_output(quiet):
        wrapper.reset(seed=seed)

    if (
        int(wrapper.max_episode_length)
        != MAX_EPISODE_STEPS
    ):
        wrapper.close()
        raise RuntimeError(
            f"Expected an episode length of "
            f"{MAX_EPISODE_STEPS}, but the "
            "environment exposes "
            f"{wrapper.max_episode_length}."
        )

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
            "learning_rate": (
                LEARNING_RATE
            ),
            "adam_epsilon": (
                ADAM_EPSILON
            ),
            "min_adaptive_lr": (
                MIN_ADAPTIVE_LR
                if SCHEDULE == "adaptive"
                else None
            ),
            "max_adaptive_lr": (
                MAX_ADAPTIVE_LR
                if SCHEDULE == "adaptive"
                else None
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
            "max_grad_norm": (
                MAX_GRAD_NORM
            ),
            "entropy_coef": ENTROPY_COEF,
            "value_loss_coef": (
                VALUE_LOSS_COEF
            ),
            "use_clipped_value_loss": (
                USE_CLIPPED_VALUE_LOSS
            ),
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "normalize_advantage_per_mini_batch": (
                NORMALIZE_ADVANTAGE_PER_MINIBATCH
            ),
            "reward_normalization": (
                NORMALIZE_TRAINING_REWARDS
            ),
            "reward_normalization_gamma": (
                REWARD_NORMALIZATION_GAMMA
            ),
            "reward_normalization_clip": (
                REWARD_NORMALIZATION_CLIP
            ),
        },
        "model": {
            "algorithm": "PPO",
            "actor_class": "RNNModel",
            "critic_class": "RNNModel",
            "rnn_type": RNN_TYPE,
            "rnn_hidden_dim": (
                RNN_HIDDEN_DIM
            ),
            "rnn_num_layers": (
                RNN_NUM_LAYERS
            ),
            "post_rnn_hidden_dims": list(
                POST_RNN_HIDDEN_DIMS
            ),
            "activation": MODEL_ACTIVATION,
            "observation_normalization": (
                True
            ),
            "initial_action_std": (
                INITIAL_ACTION_STD
            ),
            "action_std_type": (
                ACTION_STD_TYPE
            ),
        },
        "evaluation": {
            "num_evals": NUM_EVALS,
            "eval_num_envs": (
                EVAL_NUM_ENVS
            ),
            "eval_steps": EVAL_STEPS,
            "fixed_seed": (
                args.seed + 20_000
            ),
            "first_episode_only": True,
        },
        "task": {
            "max_episode_steps": (
                MAX_EPISODE_STEPS
            ),
            "control_dt": CTRL_DT,
            "episode_duration_sec": (
                MAX_EPISODE_STEPS
                * CTRL_DT
            ),
            "drop_height": DROP_HEIGHT,
        },
        "ball_configuration": {
            "size_scale": BALL_SIZE_SCALE,
            "mass_scale": (
                BALL_SIZE_SCALE ** 3
            ),
            "pair_reset_offset_low": list(
                BALL_PAIR_RESET_OFFSET_LOW
            ),
            "pair_reset_offset_high": list(
                BALL_PAIR_RESET_OFFSET_HIGH
            ),
            "individual_reset_jitter_low": list(
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            "individual_reset_jitter_high": list(
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
            "vertical_randomization": False,
        },
        "reward": {
            "mode": REWARD_MODE,
            "hold_reward_per_step": (
                HOLD_REWARD_PER_STEP
            ),
            "drop_penalty": DROP_PENALTY,
            "static_target1": list(
                STATIC_TARGET1_POS
            ),
            "static_target2": list(
                STATIC_TARGET2_POS
            ),
            "static_position_weight": (
                STATIC_POSITION_WEIGHT
            ),
            "static_alive_weight": (
                STATIC_ALIVE_WEIGHT
            ),
            "static_solved_weight": (
                STATIC_SOLVED_WEIGHT
            ),
            "static_proximity_threshold": (
                STATIC_PROXIMITY_THRESHOLD
            ),
            "maximum_static_return_ignoring_distance": (
                MAX_EPISODE_STEPS
                * (
                    STATIC_ALIVE_WEIGHT
                    + STATIC_SOLVED_WEIGHT
                )
            ),
        },
        "observation_layout": {
            "hand_qpos": [0, 23],
            "hand_qvel": [23, 46],
            "ball1_pos": [46, 49],
            "ball1_lin_vel": [49, 52],
            "ball2_pos": [52, 55],
            "ball2_lin_vel": [55, 58],
            "activation": [58, 97],
            "total": OBS_DIM,
            "ball_position_frame": (
                "environment_local"
            ),
        },
        "action_dimension": (
            EXPECTED_ACTION_DIM
        ),
        "action_clip": ACTION_CLIP,
        "best_policy_metric": (
            BEST_POLICY_METRIC
        ),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    args: argparse.Namespace,
) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    policy_path = resolve_path(
        args.policy_path
    )
    best_path = best_policy_path(
        policy_path
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

    make_env, load_rl_cfg = (
        import_mjlab_tools(
            quiet=args.quiet
        )
    )

    from rsl_rl.runners import OnPolicyRunner

    train_env: (
        BaodingRslRlWrapper | None
    ) = None
    eval_env: (
        BaodingRslRlWrapper | None
    ) = None

    try:
        train_env = make_wrapped_env(
            make_env=make_env,
            env_id=args.env_id,
            device=args.device,
            num_envs=NUM_ENVS,
            seed=args.seed,
            quiet=args.quiet,
            normalize_rewards=(
                NORMALIZE_TRAINING_REWARDS
            ),
        )

        # Do not let evaluation construction
        # alter model-initialization randomness.
        with preserve_global_rng_state():
            eval_env = make_wrapped_env(
                make_env=make_env,
                env_id=args.env_id,
                device=args.device,
                num_envs=EVAL_NUM_ENVS,
                seed=args.seed + 10_000,
                quiet=args.quiet,
                normalize_rewards=False,
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
        configure_lstm_models(
            train_cfg_dict
        )

        with redirect_output(args.quiet):
            runner = OnPolicyRunner(
                env=train_env,
                train_cfg=train_cfg_dict,
                log_dir=None,
                device=args.device,
            )

        actor, critic = (
            validate_lstm_runner(runner)
        )
        algorithm = runner.alg

        install_adaptive_lr_bounds(
            algorithm
        )
        patch_recurrent_critic_bootstrap(
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
            ) == 0
        ):
            raise RuntimeError(
                "RSL-RL PPO optimizer "
                "was not found."
            )

        if not isinstance(
            optimizer,
            torch.optim.Adam,
        ):
            raise RuntimeError(
                "The paper-style configuration "
                "requires torch.optim.Adam, but "
                "RSL-RL constructed "
                f"{type(optimizer).__name__}."
            )

        optimizer.defaults[
            "eps"
        ] = ADAM_EPSILON

        for parameter_group in (
            optimizer.param_groups
        ):
            parameter_group[
                "eps"
            ] = ADAM_EPSILON

        effective_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        if (
            effective_schedule
            != SCHEDULE
        ):
            raise RuntimeError(
                f"Expected PPO schedule "
                f"{SCHEDULE!r}, but RSL-RL "
                "is using "
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
                "Expected optimizer learning "
                f"rate {LEARNING_RATE}, but "
                "RSL-RL is using "
                f"{effective_lr}."
            )

        validate_effective_ppo_config(
            algorithm=algorithm,
            optimizer=optimizer,
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
        eval_every = (
            ITERATIONS // NUM_EVALS
        )

        saved_config = (
            build_saved_config(args)
        )
        saved_config[
            "checkpointing"
        ] = {
            "current_policy_path": str(
                policy_path
            ),
            "best_policy_path": str(
                best_path
            ),
            "best_metric": (
                BEST_POLICY_METRIC
            ),
            "best_metric_value": None,
            "best_iteration": None,
        }

        metric_rows: list[
            dict[str, float]
        ] = []

        print("=" * 80)
        print(
            "MJLAB BAODING HOLD "
            "PPO-LSTM TRAINING"
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
            f"best policy path:      "
            f"{best_path}"
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
            f"actor model:           "
            f"{type(actor).__name__}"
        )
        print(
            f"critic model:          "
            f"{type(critic).__name__}"
        )
        print(
            f"recurrent type:        "
            f"{RNN_TYPE}"
        )
        print(
            f"recurrent hidden dim:  "
            f"{RNN_HIDDEN_DIM}"
        )
        print(
            f"recurrent layers:      "
            f"{RNN_NUM_LAYERS}"
        )
        print(
            f"post-LSTM MLP:         "
            f"{POST_RNN_HIDDEN_DIMS}"
        )
        print(
            f"activation:            "
            f"{MODEL_ACTIVATION}"
        )
        print(
            f"initial action std:    "
            f"{INITIAL_ACTION_STD:.6f}"
        )
        print(
            f"action std type:       "
            f"{ACTION_STD_TYPE}"
        )
        print(
            "action clipping:       "
            f"[-{ACTION_CLIP}, "
            f"{ACTION_CLIP}]"
        )
        print(
            f"PPO schedule:          "
            f"{effective_schedule}"
        )

        if (
            effective_schedule
            == "adaptive"
        ):
            print(
                "adaptive LR range:     "
                f"[{MIN_ADAPTIVE_LR}, "
                f"{MAX_ADAPTIVE_LR}]"
            )

        print(
            f"configured LR:         "
            f"{LEARNING_RATE}"
        )
        print(
            f"Adam epsilon:          "
            f"{ADAM_EPSILON}"
        )
        print(
            f"internal PPO LR:       "
            f"{algorithm.learning_rate}"
        )
        print(
            f"optimizer LR:          "
            f"{effective_lr}"
        )
        print(
            f"desired KL:            "
            f"{algorithm.desired_kl}"
        )
        print(
            f"learning epochs:       "
            f"{algorithm.num_learning_epochs}"
        )
        print(
            f"mini-batches:          "
            f"{algorithm.num_mini_batches}"
        )
        print(
            f"PPO clip:              "
            f"{CLIP_PARAM}"
        )
        print(
            f"GAE lambda:            "
            f"{GAE_LAMBDA}"
        )
        print(
            "per-minibatch adv norm: "
            f"{NORMALIZE_ADVANTAGE_PER_MINIBATCH}"
        )
        print(
            f"value loss coef:       "
            f"{VALUE_LOSS_COEF}"
        )
        print(
            f"clipped value loss:    "
            f"{USE_CLIPPED_VALUE_LOSS}"
        )
        print(
            f"entropy coef:          "
            f"{ENTROPY_COEF}"
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
            f"evaluation horizon:    "
            f"{EVAL_STEPS} steps"
        )
        print(
            "episode duration:      "
            f"{MAX_EPISODE_STEPS * CTRL_DT:.1f} "
            "seconds"
        )
        print(
            f"reward mode:           "
            f"{REWARD_MODE}"
        )
        print(
            "reward normalization:  "
            f"{NORMALIZE_TRAINING_REWARDS}"
        )
        print(
            "maximum static return: "
            f"{MAX_EPISODE_STEPS * (STATIC_ALIVE_WEIGHT + STATIC_SOLVED_WEIGHT)}"
        )
        print(
            f"static target 1:       "
            f"{STATIC_TARGET1_POS}"
        )
        print(
            f"static target 2:       "
            f"{STATIC_TARGET2_POS}"
        )
        print(
            f"ball size scale:       "
            f"{BALL_SIZE_SCALE}"
        )
        print(
            f"ball mass scale:       "
            f"{BALL_SIZE_SCALE ** 3:.6f}"
        )
        print(
            "pair reset range:      "
            f"{BALL_PAIR_RESET_OFFSET_LOW} "
            "to "
            f"{BALL_PAIR_RESET_OFFSET_HIGH}"
        )
        print(
            "individual jitter:     "
            f"{BALL_INDIVIDUAL_RESET_JITTER_LOW} "
            "to "
            f"{BALL_INDIVIDUAL_RESET_JITTER_HIGH}"
        )
        print(
            f"observation dimension: "
            f"{OBS_DIM}"
        )
        print(
            f"action dimension:      "
            f"{EXPECTED_ACTION_DIM}"
        )
        print("=" * 80)

        start_time = time.time()
        completed_iterations = 0
        eval_index = 0

        initial_evaluation = (
            evaluate_policy(
                runner=runner,
                eval_env=eval_env,
                seed=(
                    args.seed + 20_000
                ),
                device=args.device,
                quiet=args.quiet,
            )
        )

        initial_row = make_metric_row(
            eval_index=eval_index,
            iteration=(
                completed_iterations
            ),
            env_steps=0,
            elapsed_sec=0.0,
            eval_metrics=(
                initial_evaluation
            ),
            train_metrics={
                "train/num_updates": 0.0,
                "train/learning_rate": (
                    effective_lr
                ),
            },
        )

        metric_rows.append(initial_row)

        best_metric_value = float(
            initial_evaluation[
                BEST_POLICY_METRIC
            ]
        )
        best_iteration = 0

        runner.save(
            str(policy_path),
            infos=(
                train_env.checkpoint_infos()
            ),
        )
        runner.save(
            str(best_path),
            infos=(
                train_env.checkpoint_infos()
            ),
        )

        saved_config[
            "checkpointing"
        ][
            "best_metric_value"
        ] = best_metric_value
        saved_config[
            "checkpointing"
        ][
            "best_iteration"
        ] = best_iteration

        save_metrics(
            path=metrics_path,
            config=saved_config,
            rows=metric_rows,
        )

        print(
            f"[eval {eval_index:03d}/"
            f"{NUM_EVALS:03d}] "
            "iter=0 steps=0 "
            f"return="
            f"{initial_evaluation['eval/mean_reward']:.2f} "
            f"held="
            f"{initial_evaluation['eval/mean_hold_steps']:.1f}/"
            f"{EVAL_STEPS} "
            f"hold_time="
            f"{initial_evaluation['eval/mean_hold_time_sec']:.2f}s "
            f"full="
            f"{initial_evaluation['eval/full_hold_fraction']:.3f} "
            f"dropped="
            f"{initial_evaluation['eval/drop_fraction']:.3f} "
            f"solved="
            f"{initial_evaluation['eval/mean_static_solved_fraction']:.3f} "
            "best=yes"
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
            update_storage[
                "grad_norm_values"
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
                    update_storage[
                        "updates"
                    ]
                )
            )

            train_metrics.update(
                summarize_samples(
                    update_storage[
                        "kl_values"
                    ],
                    prefix=(
                        "train/approx_kl"
                    ),
                )
            )
            train_metrics.update(
                summarize_samples(
                    update_storage[
                        "grad_norm_values"
                    ],
                    prefix=(
                        "train/"
                        "grad_norm_after_clip"
                    ),
                )
            )
            train_metrics.update(
                actor_std_metrics(actor)
            )
            train_metrics.update(
                train_env
                .pop_training_diagnostics()
            )

            train_metrics[
                "train/learning_rate_final"
            ] = float(
                optimizer.param_groups[
                    0
                ]["lr"]
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
                seed=(
                    args.seed + 20_000
                ),
                device=args.device,
                quiet=args.quiet,
            )

            elapsed_seconds = (
                time.time() - start_time
            )

            row = make_metric_row(
                eval_index=eval_index,
                iteration=(
                    completed_iterations
                ),
                env_steps=env_steps,
                elapsed_sec=(
                    elapsed_seconds
                ),
                eval_metrics=evaluation,
                train_metrics=(
                    train_metrics
                ),
            )

            metric_rows.append(row)

            # Always retain the latest policy.
            runner.save(
                str(policy_path),
                infos=(
                    train_env
                    .checkpoint_infos()
                ),
            )

            current_best_metric = float(
                evaluation[
                    BEST_POLICY_METRIC
                ]
            )
            is_new_best = (
                current_best_metric
                > best_metric_value
            )

            if is_new_best:
                best_metric_value = (
                    current_best_metric
                )
                best_iteration = (
                    completed_iterations
                )

                runner.save(
                    str(best_path),
                    infos=(
                        train_env
                        .checkpoint_infos()
                    ),
                )

            saved_config[
                "checkpointing"
            ][
                "best_metric_value"
            ] = best_metric_value
            saved_config[
                "checkpointing"
            ][
                "best_iteration"
            ] = best_iteration

            save_metrics(
                path=metrics_path,
                config=saved_config,
                rows=metric_rows,
            )

            current_learning_rate = float(
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            approx_kl = train_metrics.get(
                "train/approx_kl_mean"
            )
            kl_text = (
                "n/a"
                if approx_kl is None
                else f"{approx_kl:.3e}"
            )

            action_std = train_metrics.get(
                "train/action_std_mean"
            )
            std_text = (
                "n/a"
                if action_std is None
                else f"{action_std:.3e}"
            )

            action_clip_fraction = (
                train_metrics.get(
                    "train/"
                    "action_clip_fraction"
                )
            )
            action_clip_text = (
                "n/a"
                if (
                    action_clip_fraction
                    is None
                )
                else (
                    f"{action_clip_fraction:.3f}"
                )
            )

            print(
                f"[eval {eval_index:03d}/"
                f"{NUM_EVALS:03d}] "
                f"iter={completed_iterations} "
                f"steps={env_steps} "
                f"return="
                f"{evaluation['eval/mean_reward']:.2f} "
                f"held="
                f"{evaluation['eval/mean_hold_steps']:.1f}/"
                f"{EVAL_STEPS} "
                f"hold_time="
                f"{evaluation['eval/mean_hold_time_sec']:.2f}s "
                f"full="
                f"{evaluation['eval/full_hold_fraction']:.3f} "
                f"dropped="
                f"{evaluation['eval/drop_fraction']:.3f} "
                f"solved="
                f"{evaluation['eval/mean_static_solved_fraction']:.3f} "
                f"lr="
                f"{current_learning_rate:.3e} "
                f"kl={kl_text} "
                f"std={std_text} "
                f"act_clip="
                f"{action_clip_text} "
                f"best="
                f"{'yes' if is_new_best else 'no'} "
                f"time="
                f"{elapsed_seconds:.1f}s"
            )

        runner.save(
            str(policy_path),
            infos=(
                train_env.checkpoint_infos()
            ),
        )

        save_metrics(
            path=metrics_path,
            config=saved_config,
            rows=metric_rows,
        )

        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(
            f"current policy: "
            f"{policy_path}"
        )
        print(
            f"best policy:    "
            f"{best_path}"
        )
        print(
            f"best {BEST_POLICY_METRIC}: "
            f"{best_metric_value:.6f} "
            f"at iteration "
            f"{best_iteration}"
        )
        print(
            f"metrics:        "
            f"{metrics_path}"
        )

    finally:
        if train_env is not None:
            train_env.close()

        if eval_env is not None:
            eval_env.close()


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
            "my_baoding_static_lstm_policy.pt"
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
            f"Requested device {args.device}, "
            "but CUDA is not available."
        )

    if not (
        0.0 < BALL_SIZE_SCALE <= 1.0
    ):
        raise ValueError(
            "BALL_SIZE_SCALE must be "
            "in (0, 1]."
        )

    reset_ranges = (
        (
            "BALL_PAIR_RESET_OFFSET",
            BALL_PAIR_RESET_OFFSET_LOW,
            BALL_PAIR_RESET_OFFSET_HIGH,
        ),
        (
            "BALL_INDIVIDUAL_RESET_JITTER",
            BALL_INDIVIDUAL_RESET_JITTER_LOW,
            BALL_INDIVIDUAL_RESET_JITTER_HIGH,
        ),
    )

    for name, low, high in reset_ranges:
        if (
            len(low) != 3
            or len(high) != 3
        ):
            raise ValueError(
                f"{name} bounds must contain "
                "exactly three values."
            )

        if any(
            low_value > high_value
            for (
                low_value,
                high_value,
            ) in zip(
                low,
                high,
                strict=True,
            )
        ):
            raise ValueError(
                f"Every lower {name} bound "
                "must be less than or equal "
                "to its upper bound."
            )

    vertical_bounds = (
        BALL_PAIR_RESET_OFFSET_LOW[2],
        BALL_PAIR_RESET_OFFSET_HIGH[2],
        BALL_INDIVIDUAL_RESET_JITTER_LOW[2],
        BALL_INDIVIDUAL_RESET_JITTER_HIGH[2],
    )

    if any(
        abs(value) > 1e-12
        for value in vertical_bounds
    ):
        raise ValueError(
            "All Baoding reset z offsets "
            "must remain 0.0 so the balls "
            "preserve their safe contact "
            "height."
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
        "RNN_HIDDEN_DIM": (
            RNN_HIDDEN_DIM
        ),
        "RNN_NUM_LAYERS": (
            RNN_NUM_LAYERS
        ),
    }

    for (
        name,
        value,
    ) in positive_integer_values.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be greater "
                "than zero."
            )

    if ITERATIONS % NUM_EVALS != 0:
        raise ValueError(
            "ITERATIONS must be divisible "
            "by NUM_EVALS."
        )

    if (
        EVAL_STEPS
        != MAX_EPISODE_STEPS
    ):
        raise ValueError(
            "EVAL_STEPS must equal "
            "MAX_EPISODE_STEPS."
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

    if (
        NUM_ENVS
        % NUM_MINI_BATCHES
        != 0
    ):
        raise ValueError(
            "For recurrent minibatches, "
            "NUM_ENVS must be divisible "
            "by NUM_MINI_BATCHES."
        )

    if OBS_DIM != 97:
        raise ValueError(
            "OBS_DIM must be 97 for the "
            "Baoding observation."
        )

    if EXPECTED_ACTION_DIM != 39:
        raise ValueError(
            "EXPECTED_ACTION_DIM must be "
            "39 for the MyoHand muscles."
        )

    if CTRL_DT <= 0.0:
        raise ValueError(
            "CTRL_DT must be positive."
        )

    if HOLD_REWARD_PER_STEP <= 0.0:
        raise ValueError(
            "HOLD_REWARD_PER_STEP must "
            "be positive."
        )

    if DROP_PENALTY < 0.0:
        raise ValueError(
            "DROP_PENALTY must be "
            "non-negative."
        )

    if CLIP_PARAM <= 0.0:
        raise ValueError(
            "CLIP_PARAM must be positive."
        )

    if DESIRED_KL <= 0.0:
        raise ValueError(
            "DESIRED_KL must be positive."
        )

    if MAX_GRAD_NORM <= 0.0:
        raise ValueError(
            "MAX_GRAD_NORM must be positive."
        )

    if ENTROPY_COEF < 0.0:
        raise ValueError(
            "ENTROPY_COEF must be "
            "non-negative."
        )

    if ADAM_EPSILON <= 0.0:
        raise ValueError(
            "ADAM_EPSILON must be positive."
        )

    if VALUE_LOSS_COEF <= 0.0:
        raise ValueError(
            "VALUE_LOSS_COEF must be positive."
        )

    if not (
        0.0 < GAMMA <= 1.0
    ):
        raise ValueError(
            "GAMMA must be in (0, 1]."
        )

    if not (
        0.0 < GAE_LAMBDA <= 1.0
    ):
        raise ValueError(
            "GAE_LAMBDA must be in (0, 1]."
        )

    if ACTION_CLIP <= 0.0:
        raise ValueError(
            "ACTION_CLIP must be positive."
        )

    if INITIAL_ACTION_STD <= 0.0:
        raise ValueError(
            "INITIAL_ACTION_STD must be "
            "positive."
        )

    if ACTION_STD_TYPE not in (
        "scalar",
        "log",
    ):
        raise ValueError(
            "ACTION_STD_TYPE must be "
            "'scalar' or 'log'."
        )

    if MODEL_ACTIVATION != "relu":
        raise ValueError(
            "MODEL_ACTIVATION must be "
            "'relu' for this run."
        )

    if REWARD_MODE not in (
        "paper_static",
        "survival",
    ):
        raise ValueError(
            "REWARD_MODE must be "
            "'paper_static' or 'survival'."
        )

    if STATIC_POSITION_WEIGHT < 0.0:
        raise ValueError(
            "STATIC_POSITION_WEIGHT must "
            "be non-negative."
        )

    if STATIC_ALIVE_WEIGHT < 0.0:
        raise ValueError(
            "STATIC_ALIVE_WEIGHT must "
            "be non-negative."
        )

    if STATIC_SOLVED_WEIGHT < 0.0:
        raise ValueError(
            "STATIC_SOLVED_WEIGHT must "
            "be non-negative."
        )

    if (
        STATIC_PROXIMITY_THRESHOLD
        <= 0.0
    ):
        raise ValueError(
            "STATIC_PROXIMITY_THRESHOLD "
            "must be positive."
        )

    if (
        REWARD_NORMALIZATION_CLIP
        <= 0.0
    ):
        raise ValueError(
            "REWARD_NORMALIZATION_CLIP "
            "must be positive."
        )

    if (
        REWARD_NORMALIZATION_EPSILON
        <= 0.0
    ):
        raise ValueError(
            "REWARD_NORMALIZATION_EPSILON "
            "must be positive."
        )

    if RNN_TYPE != "lstm":
        raise ValueError(
            "RNN_TYPE must be 'lstm'."
        )

    if (
        len(POST_RNN_HIDDEN_DIMS) == 0
        or any(
            value <= 0
            for value in (
                POST_RNN_HIDDEN_DIMS
            )
        )
    ):
        raise ValueError(
            "POST_RNN_HIDDEN_DIMS must "
            "contain positive dimensions."
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
    elif LEARNING_RATE <= 0.0:
        raise ValueError(
            "LEARNING_RATE must be positive."
        )


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()