#!/usr/bin/env python3
"""Train the first five Baoding SDS stages with gated progression in MJLab.

This adapts the task to the user's MJLab/RSL-RL GPU trainer. The published
agent instead used SB3 RecurrentPPO with state-dependent exploration and
different recurrent minibatches. The gates are experiment choices in this
port, not advancement criteria published by the authors.

Sources: Chiappa et al., Neuron 2024, doi:10.1016/j.neuron.2024.09.002;
https://github.com/amathislab/myochallenge/blob/main/docs/summary.md.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/train_baoding_rsi_static.py

Place register_baoding_rsi_static.py alongside this script; neither file
replaces your existing curriculum trainer or register_baoding.py.

Run from the repository root:

    python scripts/mjlab/BaodingBalls/train_baoding_rsi_static.py \
        --policy-path my_baoding_rsi_static_policy.pt

With no --stage, train stages 1..5 in order. Each stage advances only after
three consecutive passing evaluations, and training stops at its budget if it
never passes. --stage N runs only one stage. Each stage saves a latest
checkpoint and a gate-passing checkpoint; progression uses the passing one.

Evaluation uses 64 evenly spaced phases and the full 200-step denominator for
solved fraction. Moving stages balance hold/CW/CCW across those phases.
Training continues to sample independent phases, tasks and RSI placement.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import register_baoding_rsi_static as baoding_registration
from register_baoding_rsi_static import (
    CTRL_DT,
    DROP_HEIGHT,
    DROP_PENALTY,
    EVAL_PHASE_COUNT,
    EXPECTED_ACTION_DIM,
    HOLD_REWARD_PER_STEP,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    TASK_ID,
    evaluation_phase,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Ball size and episode horizon follow the paper. The PPO optimizer itself
# remains the current MJLab/RSL-RL implementation, not the paper's SB3 PPO.
BALL_SIZE_SCALE = 1.0

# The stage-01 reset samples a full ellipse phase instead of positional jitter.
ZERO_RESET_OFFSET = (0.0, 0.0, 0.0)
ROBUST_PAIR_RESET_OFFSET_LOW = ZERO_RESET_OFFSET
ROBUST_PAIR_RESET_OFFSET_HIGH = ZERO_RESET_OFFSET
ROBUST_INDIVIDUAL_RESET_JITTER_LOW = ZERO_RESET_OFFSET
ROBUST_INDIVIDUAL_RESET_JITTER_HIGH = ZERO_RESET_OFFSET

# Training batch and per-stage iteration limit from the uploaded stage-one
# trainer. Change MAX_ITERATIONS_PER_STAGE to adjust every stage together.
NUM_ENVS = 256
NUM_STEPS_PER_ENV = 128
MAX_ITERATIONS_PER_STAGE = 2000

# Recurrent actor and critic.
RNN_TYPE = "lstm"
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 1
POST_RNN_HIDDEN_DIMS = (256, 256)
MODEL_ACTIVATION = "relu"

# PPO.
LEARNING_RATE = 2.55e-5
ADAM_EPSILON = 1e-5
MIN_ADAPTIVE_LR = 1e-6
MAX_ADAPTIVE_LR = 2.55e-5
SCHEDULE = "adaptive"
NUM_LEARNING_EPOCHS = 10
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

# Gaussian action distribution.
INITIAL_ACTION_STD = math.exp(-2.0)
ACTION_STD_TYPE = "log"
ACTION_CLIP = 1.0

# Keep the uploaded stage-one reward in every stage. The paper's Reward
# Engineering section specifies distance weights -1 each, held weight +1,
# and both-balls-within-1.5-cm weight +5. MyoSuite baoding_v1.py supplies
# the distance/solved signals; its stock DEFAULT_RWD_KEYS_AND_WEIGHTS is a
# different reward, so these particular weights come from Chiappa et al.
STATIC_PROXIMITY_THRESHOLD = 0.015

# VecNormalize-style discounted-return normalization is training-only.
NORMALIZE_TRAINING_REWARDS = True
REWARD_NORMALIZATION_GAMMA = GAMMA
REWARD_NORMALIZATION_CLIP = 10.0
REWARD_NORMALIZATION_EPSILON = 1e-8

# Evaluation.
EVAL_STEPS = MAX_EPISODE_STEPS

# Keep KL diagnostics with the fixed schedule without allowing KL adaptation to
# change the fixed optimizer learning rate.
LOG_KL_WITH_FIXED_SCHEDULE = True


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    slug: str
    description: str
    max_iterations: int
    min_iterations: int
    eval_every: int
    eval_num_envs: int
    period_sec: float | None
    rsi_probability: float
    random_task_choice: bool
    drop_height: float
    position_weight: float
    alive_weight: float
    solved_weight: float
    pair_reset_low: tuple[float, float, float]
    pair_reset_high: tuple[float, float, float]
    individual_jitter_low: tuple[float, float, float]
    individual_jitter_high: tuple[float, float, float]
    required_full_hold_fraction: float
    required_mean_hold_steps: float
    required_solved_fraction: float
    required_consecutive_passes: int = 3


CURRICULUM_STAGES = (
    CurriculumStage(
        index=1,
        slug="rsi_static",
        description=(
            "Uniform phase along the Baoding ellipse; goals remain fixed "
            "relative to the palm while the balls start at their x/y."
        ),
        max_iterations=MAX_ITERATIONS_PER_STAGE,
        min_iterations=200,
        eval_every=20,
        eval_num_envs=EVAL_PHASE_COUNT,
        period_sec=None,
        rsi_probability=1.0,
        random_task_choice=False,
        drop_height=DROP_HEIGHT,
        position_weight=1.0,
        alive_weight=1.0,
        solved_weight=5.0,
        pair_reset_low=ZERO_RESET_OFFSET,
        pair_reset_high=ZERO_RESET_OFFSET,
        individual_jitter_low=ZERO_RESET_OFFSET,
        individual_jitter_high=ZERO_RESET_OFFSET,
        required_full_hold_fraction=0.95,
        required_mean_hold_steps=190.0,
        required_solved_fraction=0.80,
    ),
    *(
        CurriculumStage(
            index=index,
            slug=f"rsi_period_{period}",
            description=(
                f"Hold, clockwise, or counterclockwise with a {period} s "
                f"rotation period and {probability:.0%} RSI placement."
            ),
            max_iterations=MAX_ITERATIONS_PER_STAGE,
            min_iterations=200,
            eval_every=20,
            eval_num_envs=EVAL_PHASE_COUNT,
            period_sec=float(period),
            rsi_probability=probability,
            random_task_choice=True,
            drop_height=DROP_HEIGHT,
            position_weight=1.0,
            alive_weight=1.0,
            solved_weight=5.0,
            pair_reset_low=ZERO_RESET_OFFSET,
            pair_reset_high=ZERO_RESET_OFFSET,
            individual_jitter_low=ZERO_RESET_OFFSET,
            individual_jitter_high=ZERO_RESET_OFFSET,
            # The user-selected advancement gate is identical at every stage.
            # These thresholds are not author-reported curriculum gates.
            required_full_hold_fraction=0.95,
            required_mean_hold_steps=190.0,
            required_solved_fraction=0.80,
        )
        for index, period, probability in (
            (2, 30, 0.95), (3, 25, 0.90),
            (4, 20, 0.90), (5, 15, 0.90),
        )
    ),
)


def apply_baoding_environment_configuration() -> None:
    """Apply original-size balls and disable positional reset jitter."""
    baoding_registration.configure_baoding_balls(
        ball_size_scale=BALL_SIZE_SCALE,
        pair_reset_offset_low=ROBUST_PAIR_RESET_OFFSET_LOW,
        pair_reset_offset_high=ROBUST_PAIR_RESET_OFFSET_HIGH,
        individual_reset_jitter_low=ROBUST_INDIVIDUAL_RESET_JITTER_LOW,
        individual_reset_jitter_high=ROBUST_INDIVIDUAL_RESET_JITTER_HIGH,
    )


# This must happen before MJLab registers or creates the environment. Importing
# this module from the visualizer therefore applies the same ball configuration.
apply_baoding_environment_configuration()

def set_runtime_reset_randomization(stage: CurriculumStage) -> None:
    """Update the shared reset, reward and termination functions for a stage."""
    baoding_registration.configure_baoding_stage(
        index=stage.index,
        period_sec=stage.period_sec,
        rsi_probability=stage.rsi_probability,
        random_task_choice=stage.random_task_choice,
        drop_height=stage.drop_height,
    )
    baoding_registration.BALL_PAIR_RESET_OFFSET_LOW = stage.pair_reset_low
    baoding_registration.BALL_PAIR_RESET_OFFSET_HIGH = stage.pair_reset_high
    baoding_registration.BALL_INDIVIDUAL_RESET_JITTER_LOW = (
        stage.individual_jitter_low
    )
    baoding_registration.BALL_INDIVIDUAL_RESET_JITTER_HIGH = (
        stage.individual_jitter_high
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

@contextlib.contextmanager
def redirect_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
            yield


@contextlib.contextmanager
def preserve_global_rng_state():
    """Prevent evaluation and evaluation-env creation changing training RNGs."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def clone_hidden_state(hidden_state: Any) -> Any:
    if hidden_state is None:
        return None
    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach().clone()
    if isinstance(hidden_state, tuple):
        return tuple(clone_hidden_state(value) for value in hidden_state)
    raise TypeError(f"Unsupported recurrent hidden state: {type(hidden_state)}")


def import_mjlab_tools(quiet: bool):
    with redirect_output(quiet):
        import myosuite  # noqa: F401
        import mjlab.tasks.registry as registry

        from myosuite.core.registry import make_env

        baoding_registration.register_baoding_hold_task(quiet=True)

    return make_env, registry.load_rl_cfg


def unpack_action(action: Any) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        return action
    if isinstance(action, (TensorDict, dict)):
        for key in ("action", "actions"):
            if key in action:
                return action[key]
    if isinstance(action, tuple):
        return unpack_action(action[0])

    raise TypeError(f"Unsupported action output type: {type(action)}")


def get_deterministic_action(policy, obs: TensorDict) -> torch.Tensor:
    with torch.no_grad():
        return unpack_action(policy(obs))


# =============================================================================
# MJLAB TO RSL-RL WRAPPER
# =============================================================================

class RunningMeanStd:
    """Running scalar moments used by reward normalization."""

    def __init__(self, device: torch.device | str) -> None:
        self.mean = torch.zeros((), device=device, dtype=torch.float64)
        self.var = torch.ones((), device=device, dtype=torch.float64)
        self.count = torch.tensor(1e-4, device=device, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        batch = values.detach().to(dtype=torch.float64).reshape(-1)

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
        new_mean = self.mean + delta * batch_count / total_count
        new_var = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total_count
        ) / total_count

        self.mean = new_mean
        self.var = new_var.clamp_min(0.0)
        self.count = total_count


class BaodingRslRlWrapper:
    """Adapt MJLab to RSL-RL and score palm-local hold targets."""

    def __init__(
        self, env: Any, *, normalize_rewards: bool,
        stage: CurriculumStage | None = None,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.num_actions = int(env.action_space.shape[-1])
        self.cfg = getattr(env, "cfg", None)
        self.normalize_rewards = bool(normalize_rewards)

        self.stage = stage or CURRICULUM_STAGES[0]

        if self.num_actions != EXPECTED_ACTION_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_ACTION_DIM} actions, got {self.num_actions}."
            )

        env_origins = env.scene.env_origins
        expected_shape = (self.num_envs, 3)

        if tuple(env_origins.shape) != expected_shape:
            raise RuntimeError(
                f"Expected env_origins shape {expected_shape}, "
                f"got {tuple(env_origins.shape)}."
            )

        self._env_origins = env_origins.to(
            device=self.device,
            dtype=torch.float32,
        ).detach().clone()

        self._return_rms = RunningMeanStd(self.device)
        self._discounted_returns = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        self._episode_returns = torch.zeros_like(self._discounted_returns)
        self._episode_lengths = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._reset_interval_diagnostics()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    def close(self) -> None:
        self.env.close()

    def set_curriculum_stage(self, stage: CurriculumStage) -> None:
        self.stage = stage

    def reset(self, seed: int | None = None) -> TensorDict:
        if seed is None:
            self.env.reset()
        else:
            self.env.reset(seed=seed)

        self._discounted_returns.zero_()
        self._episode_returns.zero_()
        self._episode_lengths.zero_()

        obs = self.get_observations()
        policy_obs = obs["policy"]
        if not (
            torch.allclose(
                policy_obs[:, 41:44],
                policy_obs[:, 35:38] - policy_obs[:, 23:26],
                atol=1e-5,
            )
            and torch.allclose(
                policy_obs[:, 44:47],
                policy_obs[:, 38:41] - policy_obs[:, 29:32],
                atol=1e-5,
            )
        ):
            raise RuntimeError("RSI target/error observation order is incorrect.")
        # The first-stage RSI copies target x/y into the two ball free joints.
        # A mismatched hand-parent transform here would invalidate training.
        initial_xy_error = torch.maximum(
            torch.linalg.vector_norm(policy_obs[:, 41:43], dim=-1),
            torch.linalg.vector_norm(policy_obs[:, 44:46], dim=-1),
        )
        placed = getattr(self.env, "_rsi_reset_placed", None)
        if placed is None:
            raise RuntimeError("The RSI reset event did not run.")
        if bool(placed.any()) and float(initial_xy_error[placed].max()) > 0.0125:
            raise RuntimeError(
                "RSI reset put a ball more than 1.25 cm from its "
                "palm-relative target in x/y; check the target parent frame. "
                f"Maximum placed-ball x/y error: "
                f"{float(initial_xy_error[placed].max()):.4f} m."
            )
        return obs

    def get_observations(self) -> TensorDict:
        obs_dict = self.env.observation_manager.compute()
        return self._build_local_observation(obs_dict)

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        raw_actions = actions
        clipped_actions = torch.clamp(
            raw_actions,
            -ACTION_CLIP,
            ACTION_CLIP,
        )
        self._record_action_diagnostics(
            raw_actions,
            clipped_actions,
        )

        (
            obs_dict,
            native_reward,
            terminated,
            truncated,
            extras,
        ) = self.env.step(clipped_actions)

        native_reward = native_reward.reshape(self.num_envs)
        terminated = terminated.reshape(self.num_envs).bool()
        truncated = truncated.reshape(self.num_envs).bool()
        done_mask = terminated | truncated

        # The reward used the current target. Advance survivors for the next
        # observation; MJLab's reset event already set completed envs to zero.
        self.env._rsi_episode_step[~done_mask] += 1
        if self.stage.period_sec is not None:
            obs_dict = self.env.observation_manager.compute()

        obs = self._build_local_observation(obs_dict)

        # MJLab's reward manager runs on the just-stepped state, before it
        # auto-resets done environments. The returned observation for a done
        # environment is already from its next episode, so use the native
        # pre-reset reward and the diagnostics cached by that reward term.
        reward_components = getattr(self.env, "_rsi_step_components", None)
        if reward_components is None:
            raise RuntimeError(
                "The Baoding reward term did not cache pre-reset diagnostics. "
                "Check that register_baoding_rsi_static.py is being imported."
            )
        raw_task_reward = native_reward

        rewards = self._normalize_reward(
            raw_task_reward,
            done_mask,
        )
        both_held = (~terminated).float()
        ball_dropped = terminated.float()

        self._record_episode_diagnostics(
            raw_task_reward,
            terminated,
            truncated,
            done_mask,
        )

        infos: dict[str, Any] = {
            "baoding": {
                "reward": raw_task_reward.detach(),
                "normalized_reward": rewards.detach(),
                "native_static_reward": native_reward.detach(),
                "both_held": both_held.detach(),
                "ball_dropped": ball_dropped.detach(),
                "terminated": terminated.detach(),
                "truncated": truncated.detach(),
                "target1_distance": reward_components[
                    "target1_distance"
                ].detach(),
                "target2_distance": reward_components[
                    "target2_distance"
                ].detach(),
                "ball1_speed": reward_components[
                    "ball1_speed"
                ].detach(),
                "ball2_speed": reward_components[
                    "ball2_speed"
                ].detach(),
                "static_solved": reward_components[
                    "static_solved"
                ].detach(),
            },
            "episode": {
                "reward": raw_task_reward.detach().mean(),
                "both_held": both_held.detach().mean(),
                "ball_dropped": ball_dropped.detach().mean(),
                "static_solved": reward_components[
                    "static_solved"
                ].detach().mean(),
            },
        }

        if isinstance(extras, dict):
            infos["time_outs"] = extras.get(
                "time_outs",
                truncated,
            )

            if "log" in extras:
                infos["log"] = extras["log"]
        else:
            infos["time_outs"] = truncated

        return (
            obs,
            rewards,
            done_mask.long(),
            infos,
        )

    def pop_training_diagnostics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}

        if self._action_value_count > 0:
            metrics.update(
                {
                    "train/action_clip_fraction": (
                        float(self._action_clipped_count.cpu())
                        / self._action_value_count
                    ),
                    "train/action_abs_mean_pre_clip": (
                        float(self._action_abs_sum.cpu())
                        / self._action_value_count
                    ),
                    "train/action_abs_max_pre_clip": float(
                        self._action_abs_max.cpu()
                    ),
                }
            )

        completed_count = float(
            self._completed_episode_count.cpu()
        )

        if completed_count > 0:
            metrics.update(
                {
                    "train/completed_episodes": completed_count,
                    "train/episode_return_raw": (
                        float(
                            self._completed_return_sum.cpu()
                        )
                        / completed_count
                    ),
                    "train/episode_length": (
                        float(
                            self._completed_length_sum.cpu()
                        )
                        / completed_count
                    ),
                    "train/drop_fraction": (
                        float(
                            self._completed_drop_count.cpu()
                        )
                        / completed_count
                    ),
                    "train/time_limit_fraction": (
                        float(
                            self._completed_timeout_count.cpu()
                        )
                        / completed_count
                    ),
                }
            )

        metrics.update(
            {
                "train/reward_rms_mean": float(
                    self._return_rms.mean.cpu()
                ),
                "train/reward_rms_var": float(
                    self._return_rms.var.cpu()
                ),
                "train/reward_rms_std": float(
                    torch.sqrt(
                        self._return_rms.var
                    ).cpu()
                ),
                "train/reward_rms_count": float(
                    self._return_rms.count.cpu()
                ),
            }
        )

        self._reset_interval_diagnostics()
        return metrics

    def checkpoint_infos(self) -> dict[str, Any]:
        return {
            "reward_mode": "curriculum_baoding",
            "curriculum_stage": asdict(self.stage),
            "action_clip": ACTION_CLIP,
            "reward_normalizer": {
                "enabled": self.normalize_rewards,
                "mean": float(
                    self._return_rms.mean.cpu()
                ),
                "var": float(
                    self._return_rms.var.cpu()
                ),
                "count": float(
                    self._return_rms.count.cpu()
                ),
            },
        }

    def restore_reward_normalizer(self, infos: dict[str, Any]) -> None:
        state = infos.get("reward_normalizer")
        if not isinstance(state, dict):
            raise RuntimeError("Prior checkpoint lacks reward normalizer state.")
        for name in ("mean", "var", "count"):
            if name not in state:
                raise RuntimeError(f"Prior checkpoint lacks reward_normalizer/{name}.")
        self._return_rms.mean.fill_(float(state["mean"]))
        self._return_rms.var.fill_(float(state["var"]))
        self._return_rms.count.fill_(float(state["count"]))
        self._discounted_returns.zero_()

    def _build_local_observation(
        self,
        obs_dict: Any,
    ) -> TensorDict:
        obs = TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
        )
        policy_obs = obs["policy"].clone()

        policy_obs[:, 23:26] -= self._env_origins
        policy_obs[:, 29:32] -= self._env_origins
        policy_obs[:, 35:38] -= self._env_origins
        policy_obs[:, 38:41] -= self._env_origins

        expected_shape = (
            self.num_envs,
            OBS_DIM,
        )

        if tuple(policy_obs.shape) != expected_shape:
            raise RuntimeError(
                "Expected policy observation "
                f"shape {expected_shape}, "
                f"got {tuple(policy_obs.shape)}."
            )

        obs["policy"] = policy_obs
        return obs

    def _normalize_reward(
        self,
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
        raw_actions: torch.Tensor,
        clipped_actions: torch.Tensor,
    ) -> None:
        raw_detached = raw_actions.detach()
        abs_actions = raw_detached.abs()

        self._action_value_count += int(
            raw_detached.numel()
        )

        self._action_clipped_count.add_(
            torch.count_nonzero(
                raw_detached
                != clipped_actions.detach()
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
        raw_task_reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> None:
        self._episode_returns.add_(
            raw_task_reward
        )
        self._episode_lengths.add_(1)

        self._completed_episode_count.add_(
            done_mask.sum()
        )
        self._completed_return_sum.add_(
            self._episode_returns[
                done_mask
            ].sum()
        )
        self._completed_length_sum.add_(
            self._episode_lengths[
                done_mask
            ].sum()
        )
        self._completed_drop_count.add_(
            terminated[
                done_mask
            ].sum()
        )
        self._completed_timeout_count.add_(
            (
                truncated
                & (~terminated)
            )[
                done_mask
            ].sum()
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

        self._action_clipped_count = torch.zeros(
            (),
            device=self.device,
            dtype=torch.long,
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

        self._completed_episode_count = torch.zeros(
            (),
            device=self.device,
            dtype=torch.long,
        )
        self._completed_return_sum = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )
        self._completed_length_sum = torch.zeros(
            (),
            device=self.device,
            dtype=torch.float64,
        )
        self._completed_drop_count = torch.zeros(
            (),
            device=self.device,
            dtype=torch.long,
        )
        self._completed_timeout_count = torch.zeros(
            (),
            device=self.device,
            dtype=torch.long,
        )


# =============================================================================
# RSL-RL CONFIGURATION
# =============================================================================

def configure_lstm_models(
    train_cfg: dict[str, Any],
) -> None:
    """Configure separate recurrent actor and critic models."""
    for section in (
        "actor",
        "critic",
    ):
        section_cfg = train_cfg[section]
        section_cfg.pop(
            "cnn_cfg",
            None,
        )
        section_cfg.update(
            {
                "class_name": "RNNModel",
                "hidden_dims": list(
                    POST_RNN_HIDDEN_DIMS
                ),
                "activation": MODEL_ACTIVATION,
                "obs_normalization": True,
                "rnn_type": RNN_TYPE,
                "rnn_hidden_dim": RNN_HIDDEN_DIM,
                "rnn_num_layers": RNN_NUM_LAYERS,
            }
        )

    train_cfg[
        "actor"
    ][
        "distribution_cfg"
    ].update(
        {
            "class_name": (
                "GaussianDistribution"
            ),
            "init_std": INITIAL_ACTION_STD,
            "std_type": ACTION_STD_TYPE,
        }
    )


def validate_lstm_runner(
    runner,
) -> tuple[Any, Any]:
    """Return the recurrent models, with one useful construction check."""
    actor = runner.alg.actor
    critic = runner.alg.critic

    if not all(
        isinstance(
            model.rnn.rnn,
            torch.nn.LSTM,
        )
        for model in (
            actor,
            critic,
        )
    ):
        raise RuntimeError(
            "RSL-RL did not construct "
            "LSTM actor and critic models."
        )

    return actor, critic


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
            found = (
                set_key_recursively(
                    child,
                    key,
                    value,
                )
                or found
            )

    return found


def set_training_value(
    train_cfg: dict[str, Any],
    possible_keys: tuple[str, ...],
    value: Any,
) -> None:
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
    """Apply this experiment's PPO settings to the loaded configuration."""
    train_cfg[
        "num_steps_per_env"
    ] = NUM_STEPS_PER_ENV

    train_cfg["logger"] = "tensorboard"

    values = (
        (
            (
                "learning_rate",
                "lr",
            ),
            LEARNING_RATE,
        ),
        (
            (
                "num_learning_epochs",
            ),
            NUM_LEARNING_EPOCHS,
        ),
        (
            (
                "num_mini_batches",
                "num_minibatches",
            ),
            NUM_MINI_BATCHES,
        ),
        (
            (
                "clip_param",
                "clip_range",
            ),
            CLIP_PARAM,
        ),
        (
            (
                "desired_kl",
            ),
            DESIRED_KL,
        ),
        (
            (
                "max_grad_norm",
            ),
            MAX_GRAD_NORM,
        ),
        (
            (
                "entropy_coef",
                "entropy_coeff",
            ),
            ENTROPY_COEF,
        ),
        (
            (
                "value_loss_coef",
                "value_coef",
                "vf_coef",
            ),
            VALUE_LOSS_COEF,
        ),
        (
            (
                "use_clipped_value_loss",
            ),
            USE_CLIPPED_VALUE_LOSS,
        ),
        (
            (
                "gamma",
            ),
            GAMMA,
        ),
        (
            (
                "lam",
                "gae_lambda",
            ),
            GAE_LAMBDA,
        ),
        (
            (
                "normalize_advantage_per_mini_batch",
            ),
            NORMALIZE_ADVANTAGE_PER_MINIBATCH,
        ),
    )

    for possible_keys, value in values:
        set_training_value(
            train_cfg,
            possible_keys,
            value,
        )

    train_cfg[
        "algorithm"
    ][
        "schedule"
    ] = SCHEDULE


def install_adaptive_lr_bounds(
    algorithm,
) -> dict[str, float] | None:
    """Force the requested adaptive-LR bounds on every optimizer step.

    RSL-RL currently hard-codes 1e-5 and 1e-2 as its adaptive bounds. The
    mutable state returned here is updated from each measured minibatch KL in
    patch_runner_for_metrics(). The optimizer hook then overwrites RSL-RL's
    internally clamped value immediately before the parameter update.
    """
    if SCHEDULE != "adaptive":
        return None

    optimizer = algorithm.optimizer

    adaptive_lr_state = {
        "learning_rate": min(
            MAX_ADAPTIVE_LR,
            max(
                MIN_ADAPTIVE_LR,
                float(
                    algorithm.learning_rate
                ),
            ),
        )
    }

    def force_bounded_lr_before_step(
        _optimizer,
        _args,
        _kwargs,
    ) -> None:
        bounded_lr = min(
            MAX_ADAPTIVE_LR,
            max(
                MIN_ADAPTIVE_LR,
                adaptive_lr_state[
                    "learning_rate"
                ],
            ),
        )

        adaptive_lr_state[
            "learning_rate"
        ] = bounded_lr

        algorithm.learning_rate = (
            bounded_lr
        )

        for parameter_group in (
            optimizer.param_groups
        ):
            parameter_group[
                "lr"
            ] = bounded_lr

    force_bounded_lr_before_step(
        optimizer,
        (),
        {},
    )

    optimizer.register_step_pre_hook(
        force_bounded_lr_before_step
    )

    return adaptive_lr_state


def patch_recurrent_critic_bootstrap(
    algorithm,
) -> None:
    """Do not advance the persistent critic state during return bootstrap."""
    critic = algorithm.critic
    original_compute_returns = (
        algorithm.compute_returns
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


# =============================================================================
# FILES AND SAVED CONFIGURATION
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
    rows: list[dict[str, Any]],
) -> None:
    temporary_path = path.with_name(
        f"{path.name}.tmp"
    )

    with temporary_path.open(
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

    temporary_path.replace(path)


def build_saved_config(
    args: argparse.Namespace,
    stage: CurriculumStage,
) -> dict[str, Any]:
    return {
        "env_id": args.env_id,
        "seed": args.seed,
        "device": args.device,
        "training": {
            "maximum_total_iterations": args.max_iterations or stage.max_iterations,
            "num_envs": args.num_envs,
            "num_steps_per_env": (
                NUM_STEPS_PER_ENV
            ),
            "learning_rate": LEARNING_RATE,
            "adam_epsilon": ADAM_EPSILON,
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
            "max_grad_norm": MAX_GRAD_NORM,
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
            "rnn_hidden_dim": RNN_HIDDEN_DIM,
            "rnn_num_layers": RNN_NUM_LAYERS,
            "post_rnn_hidden_dims": list(
                POST_RNN_HIDDEN_DIMS
            ),
            "activation": MODEL_ACTIVATION,
            "observation_normalization": True,
            "initial_action_std": (
                INITIAL_ACTION_STD
            ),
            "action_std_type": ACTION_STD_TYPE,
        },
        "evaluation": {
            "maximum_num_evals": 1 + math.ceil(
                (args.max_iterations or stage.max_iterations) / stage.eval_every
            ),
            "eval_num_envs": stage.eval_num_envs,
            "eval_steps": EVAL_STEPS,
            "stage_seed_base": (
                args.seed
                + 20_000
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
            "drop_height": stage.drop_height,
        },
        "ball_configuration": {
            "size_scale": BALL_SIZE_SCALE,
            "mass_scale": (
                BALL_SIZE_SCALE ** 3
            ),
            "vertical_randomization": False,
        },
        "reward": {
            "mode": "rsi_curriculum",
            "hold_reward_per_step": (
                HOLD_REWARD_PER_STEP
            ),
            "drop_penalty": DROP_PENALTY,
            "targets": "palm-relative, moving on hold/CW/CCW episodes",
            "eval_phases": stage.eval_num_envs,
            "rsi_ellipse_x_radius": baoding_registration.RSI_ELLIPSE_X_RADIUS,
            "rsi_ellipse_y_radius": baoding_registration.RSI_ELLIPSE_Y_RADIUS,
            "static_proximity_threshold": (
                STATIC_PROXIMITY_THRESHOLD
            ),
        },
        "curriculum": [asdict(stage)],
        "observation_layout": {
            "hand_qpos": [0, 23],
            "ball1_pos": [23, 26],
            "ball1_lin_vel": [26, 29],
            "ball2_pos": [29, 32],
            "ball2_lin_vel": [32, 35],
            "target1_pos": [35, 38],
            "target2_pos": [38, 41],
            "target1_err": [41, 44],
            "target2_err": [44, 47],
            "activation": [47, 86],
            "total": OBS_DIM,
            "ball_position_frame": (
                "environment_local"
            ),
        },
        "action_dimension": (
            EXPECTED_ACTION_DIM
        ),
        "action_clip": ACTION_CLIP,
        "checkpoint_policy": "latest per stage; passing checkpoint for progression",
    }


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
        (
            int,
            float,
        ),
    ):
        return float(value)

    if isinstance(
        value,
        torch.Tensor,
    ):
        if value.numel() == 0:
            return None

        return float(
            value
            .detach()
            .float()
            .mean()
            .cpu()
        )

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def metric_name(
    name: str,
) -> str | None:
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
        "approx_kl_mean": "train/approx_kl",
        "kl_divergence": "train/approx_kl",
    }

    return mapping.get(
        name
        .lower()
        .strip()
        .replace(
            " ",
            "_",
        )
    )


def find_kl_source(
    runner,
):
    candidates = [
        getattr(
            runner.alg,
            "actor",
            None,
        ),
        getattr(
            runner.alg,
            "actor_critic",
            None,
        ),
        getattr(
            runner.alg,
            "policy",
            None,
        ),
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
    adaptive_lr_state: (
        dict[str, float]
        | None
    ),
) -> dict[str, list[Any]]:
    """Capture PPO diagnostics and apply the requested adaptive LR bounds."""
    storage: dict[
        str,
        list[Any],
    ] = {
        "updates": [],
        "kl_values": [],
        "grad_norm_values": [],
    }

    algorithm = runner.alg
    optimizer = algorithm.optimizer

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
                    parameter
                    .grad
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

    fixed_lr: float | None = None

    if (
        LOG_KL_WITH_FIXED_SCHEDULE
        and SCHEDULE == "fixed"
    ):
        fixed_lr = float(
            optimizer
            .param_groups[0][
                "lr"
            ]
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

    kl_source = find_kl_source(runner)

    if (
        adaptive_lr_state is not None
        and kl_source is None
    ):
        raise RuntimeError(
            "Adaptive LR requires an actor "
            "exposing get_kl_divergence()."
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

                if (
                    adaptive_lr_state
                    is not None
                ):
                    desired_kl = to_float(
                        algorithm.desired_kl
                    )

                    if desired_kl is not None:
                        learning_rate = (
                            adaptive_lr_state[
                                "learning_rate"
                            ]
                        )

                        if (
                            scalar
                            > 2.0
                            * desired_kl
                        ):
                            learning_rate = max(
                                MIN_ADAPTIVE_LR,
                                learning_rate
                                / 1.5,
                            )
                        elif (
                            0.0
                            < scalar
                            < 0.5
                            * desired_kl
                        ):
                            learning_rate = min(
                                MAX_ADAPTIVE_LR,
                                learning_rate
                                * 1.5,
                            )

                        adaptive_lr_state[
                            "learning_rate"
                        ] = learning_rate

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
            storage[
                "grad_norm_values"
            ]
        )
        previous_schedule = (
            algorithm.schedule
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

                        explained_variance = float(
                            (
                                1.0
                                - residual_variance
                                / target_variance
                            ).cpu()
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

        record[
            "train/learning_rate"
        ] = float(
            optimizer
            .param_groups[0][
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
            (
                tuple,
                list,
            ),
        ):
            names = (
                "train/value",
                "train/surrogate",
                "train/entropy",
                "train/approx_kl",
            )

            for (
                name,
                raw_value,
            ) in zip(
                names,
                update_output,
            ):
                value = to_float(
                    raw_value
                )

                if value is not None:
                    record[name] = value

        new_kl_values = storage[
            "kl_values"
        ][
            kl_start:
        ]

        if new_kl_values:
            record[
                "train/approx_kl"
            ] = float(
                sum(new_kl_values)
                / len(new_kl_values)
            )

        new_grad_values = storage[
            "grad_norm_values"
        ][
            grad_start:
        ]

        if new_grad_values:
            record[
                "train/grad_norm_after_clip"
            ] = float(
                sum(new_grad_values)
                / len(new_grad_values)
            )

            record[
                "train/grad_norm_after_clip_max"
            ] = float(
                max(new_grad_values)
            )

        storage[
            "updates"
        ].append(record)

        return update_output

    algorithm.update = wrapped_update

    return storage


def average_update_metrics(
    records: list[
        dict[str, float]
    ],
) -> dict[str, float]:
    if not records:
        return {
            "train/num_updates": 0.0
        }

    result = {
        "train/num_updates": float(
            len(records)
        )
    }

    keys = {
        key
        for record in records
        for key in record
    }

    for key in keys:
        values = [
            record[key]
            for record in records
            if key in record
        ]

        result[key] = float(
            sum(values)
            / len(values)
        )

    return result


def summarize_samples(
    values: list[Any],
    *,
    prefix: str,
) -> dict[str, float]:
    clean_values = [
        scalar
        for value in values
        if (
            scalar := to_float(value)
        ) is not None
    ]

    if not clean_values:
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
    std = (
        actor.output_std
        .detach()
        .float()
        .reshape(-1)
    )

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
# EVALUATION
# =============================================================================

def evaluate_policy(
    runner,
    eval_env: BaodingRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    """Evaluate one episode per evenly spaced phase with a fixed stage seed."""
    with preserve_global_rng_state():
        count = eval_env.num_envs
        eval_env.env._rsi_forced_eval_phase = torch.as_tensor(
            [evaluation_phase(i, count) for i in range(count)],
            device=eval_env.device, dtype=torch.float32,
        )
        eval_env.env._rsi_forced_eval_direction = (
            torch.as_tensor(
                [(i % 3) - 1 for i in range(count)],
                device=eval_env.device, dtype=torch.long,
            )
            if eval_env.stage.random_task_choice else None
        )
        with redirect_output(quiet):
            obs = eval_env.reset(
                seed=seed
            )

        # Snapshot phases before auto-resetting completed environments.
        phase = getattr(eval_env.env, "_rsi_static_phase", None)
        if phase is None:
            raise RuntimeError("Evaluation environment did not sample RSI phases.")
        phase_quadrant = torch.remainder(
            torch.floor(phase.detach().clone() * (2.0 / torch.pi)).long(),
            4,
        )
        directions = eval_env.env._rsi_direction.detach().clone()

        algorithm = runner.alg
        actor = algorithm.actor

        training_hidden_state = (
            clone_hidden_state(
                actor.get_hidden_state()
            )
        )

        actor_was_training = bool(
            actor.training
        )

        policy = (
            runner
            .get_inference_policy(
                device=device
            )
        )
        policy.reset()

        num_eval_envs = (
            eval_env.num_envs
        )

        episode_returns = torch.zeros(
            num_eval_envs,
            device=eval_env.device,
            dtype=torch.float32,
        )
        episode_lengths = torch.zeros(
            num_eval_envs,
            device=eval_env.device,
            dtype=torch.long,
        )
        held_steps = torch.zeros_like(
            episode_lengths
        )
        static_solved_steps = torch.zeros_like(
            episode_lengths
        )

        target1_distance_sum = torch.zeros(
            num_eval_envs,
            device=eval_env.device,
            dtype=torch.float32,
        )
        target2_distance_sum = torch.zeros_like(
            target1_distance_sum
        )
        ball_speed_sum = torch.zeros_like(
            target1_distance_sum
        )
        ball_speed_max = torch.zeros_like(
            target1_distance_sum
        )
        action_delta_sum = torch.zeros_like(
            target1_distance_sum
        )
        action_delta_count = torch.zeros_like(
            episode_lengths
        )

        active = torch.ones(
            num_eval_envs,
            device=eval_env.device,
            dtype=torch.bool,
        )
        dropped = torch.zeros_like(active)
        timed_out = torch.zeros_like(active)

        previous_actions: (
            torch.Tensor
            | None
        ) = None

        try:
            for _ in range(EVAL_STEPS):
                actions = (
                    get_deterministic_action(
                        policy,
                        obs,
                    )
                )

                applied_actions = torch.clamp(
                    actions,
                    -ACTION_CLIP,
                    ACTION_CLIP,
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
                terminated = baoding_info[
                    "terminated"
                ].bool()
                truncated = baoding_info[
                    "truncated"
                ].bool()

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

                held_steps[
                    active_before_step
                    & (~terminated)
                ] += 1

                static_solved_steps[
                    active_before_step
                ] += (
                    baoding_info[
                        "static_solved"
                    ][
                        active_before_step
                    ].long()
                )

                target1_distance_sum[
                    active_before_step
                ] += (
                    baoding_info[
                        "target1_distance"
                    ][
                        active_before_step
                    ]
                )

                target2_distance_sum[
                    active_before_step
                ] += (
                    baoding_info[
                        "target2_distance"
                    ][
                        active_before_step
                    ]
                )

                mean_ball_speed = (
                    0.5
                    * (
                        baoding_info[
                            "ball1_speed"
                        ]
                        + baoding_info[
                            "ball2_speed"
                        ]
                    )
                )

                maximum_ball_speed = (
                    torch.maximum(
                        baoding_info[
                            "ball1_speed"
                        ],
                        baoding_info[
                            "ball2_speed"
                        ],
                    )
                )

                ball_speed_sum[
                    active_before_step
                ] += mean_ball_speed[
                    active_before_step
                ]

                ball_speed_max[
                    active_before_step
                ] = torch.maximum(
                    ball_speed_max[
                        active_before_step
                    ],
                    maximum_ball_speed[
                        active_before_step
                    ],
                )

                if (
                    previous_actions
                    is not None
                ):
                    action_delta = (
                        torch.mean(
                            torch.abs(
                                applied_actions
                                - previous_actions
                            ),
                            dim=-1,
                        )
                    )

                    action_delta_sum[
                        active_before_step
                    ] += action_delta[
                        active_before_step
                    ]

                    action_delta_count[
                        active_before_step
                    ] += 1

                previous_actions = (
                    applied_actions
                    .detach()
                    .clone()
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

                if not bool(
                    active.any()
                ):
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
            held_steps
            >= EVAL_STEPS
        ) & (~dropped)

        lengths_float = (
            episode_lengths.float()
        )
        held_float = (
            held_steps.float()
        )
        solved_float = (
            static_solved_steps.float()
        )

        safe_lengths = (
            lengths_float.clamp_min(
                1.0
            )
        )
        safe_action_delta_count = (
            action_delta_count
            .float()
            .clamp_min(1.0)
        )

        results = {
            "eval/mean_reward": float(
                episode_returns
                .mean()
                .cpu()
            ),
            "eval/reward_per_step": float(
                (
                    episode_returns
                    / safe_lengths
                )
                .mean()
                .cpu()
            ),
            "eval/reward_std": float(
                episode_returns
                .std(unbiased=False)
                .cpu()
            ),
            "eval/mean_episode_length": float(
                lengths_float
                .mean()
                .cpu()
            ),
            "eval/mean_hold_steps": float(
                held_float
                .mean()
                .cpu()
            ),
            "eval/mean_hold_time_sec": float(
                (
                    held_float
                    * CTRL_DT
                )
                .mean()
                .cpu()
            ),
            "eval/mean_hold_fraction": float(
                (
                    held_float
                    / EVAL_STEPS
                )
                .mean()
                .cpu()
            ),
            "eval/mean_static_solved_fraction": float(
                (
                    solved_float
                    / EVAL_STEPS
                )
                .mean()
                .cpu()
            ),
            "eval/mean_target1_distance": float(
                (
                    target1_distance_sum
                    / safe_lengths
                )
                .mean()
                .cpu()
            ),
            "eval/mean_target2_distance": float(
                (
                    target2_distance_sum
                    / safe_lengths
                )
                .mean()
                .cpu()
            ),
            "eval/mean_total_target_distance": float(
                (
                    (
                        target1_distance_sum
                        + target2_distance_sum
                    )
                    / safe_lengths
                )
                .mean()
                .cpu()
            ),
            "eval/mean_ball_speed": float(
                (
                    ball_speed_sum
                    / safe_lengths
                )
                .mean()
                .cpu()
            ),
            "eval/max_ball_speed": float(
                ball_speed_max
                .max()
                .cpu()
            ),
            "eval/mean_action_delta": float(
                (
                    action_delta_sum
                    / safe_action_delta_count
                )
                .mean()
                .cpu()
            ),
            "eval/full_hold_fraction": float(
                full_hold
                .float()
                .mean()
                .cpu()
            ),
            "eval/drop_fraction": float(
                dropped
                .float()
                .mean()
                .cpu()
            ),
            "eval/time_limit_fraction": float(
                timed_out
                .float()
                .mean()
                .cpu()
            ),
            "eval/completed_fraction": float(
                completed
                .float()
                .mean()
                .cpu()
            ),
        }

        for quadrant in range(4):
            mask = phase_quadrant == quadrant
            results[f"eval/phase_q{quadrant}/count"] = int(mask.sum().item())
            if bool(mask.any()):
                results[f"eval/phase_q{quadrant}/solved_fraction"] = float(
                    (solved_float[mask] / EVAL_STEPS).mean().cpu()
                )
                results[f"eval/phase_q{quadrant}/full_hold_fraction"] = float(
                    full_hold[mask].float().mean().cpu()
                )

        for direction, label in ((-1, "cw"), (0, "hold"), (1, "ccw")):
            mask = directions == direction
            results[f"eval/task_{label}/count"] = int(mask.sum().item())
            if bool(mask.any()):
                results[f"eval/task_{label}/solved_fraction"] = float(
                    (solved_float[mask] / EVAL_STEPS).mean().cpu()
                )
                results[f"eval/task_{label}/full_hold_fraction"] = float(
                    full_hold[mask].float().mean().cpu()
                )

        return results


def make_metric_row(
    *,
    stage: CurriculumStage,
    stage_eval_index: int,
    stage_iteration: int,
    gate_passed: bool,
    consecutive_passes: int,
    eval_index: int,
    iteration: int,
    env_steps: int,
    elapsed_sec: float,
    eval_metrics: dict[str, float],
    train_metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage_index": float(
            stage.index
        ),
        "stage_name": stage.slug,
        "stage_eval_index": float(
            stage_eval_index
        ),
        "stage_iteration": float(
            stage_iteration
        ),
        "stage_gate_passed": float(gate_passed),
        "stage_consecutive_passes": float(consecutive_passes),
        "eval_index": float(
            eval_index
        ),
        "iteration": float(
            iteration
        ),
        "num_steps": float(
            env_steps
        ),
        "env_steps": float(
            env_steps
        ),
        "elapsed_sec": float(
            elapsed_sec
        ),
    }

    row.update(eval_metrics)
    row.update(train_metrics)

    return row


def print_evaluation(
    *,
    stage: CurriculumStage,
    stage_eval_index: int,
    maximum_stage_evals: int,
    stage_iteration: int,
    consecutive_passes: int,
    eval_index: int,
    iteration: int,
    env_steps: int,
    evaluation: dict[str, float],
    learning_rate: float,
    approx_kl: float | None,
    elapsed_sec: float,
) -> None:
    kl_text = (
        "n/a"
        if approx_kl is None
        else f"{approx_kl:.3e}"
    )

    print(
        f"[stage {stage.index}/"
        f"{len(CURRICULUM_STAGES)} "
        f"eval {stage_eval_index:02d}/"
        f"{maximum_stage_evals:02d}] "
        f"stage_iter={stage_iteration} "
        f"global_iter={iteration} "
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
        f"worst_quadrant="
        f"{min(evaluation[f'eval/phase_q{q}/solved_fraction'] for q in range(4) if evaluation[f'eval/phase_q{q}/count']):.3f} "
        f"distance="
        f"{evaluation['eval/mean_total_target_distance']:.4f} "
        f"gate={consecutive_passes}/{stage.required_consecutive_passes} "
        + (
            f"worst_task={min(evaluation[f'eval/task_{task}/solved_fraction'] for task in ('cw', 'hold', 'ccw')):.3f} "
            if stage.random_task_choice else ""
        )
        + f"speed="
        f"{evaluation['eval/mean_ball_speed']:.3f} "
        f"lr={learning_rate:.3e} "
        f"kl={kl_text} "
        f"time={elapsed_sec:.1f}s"
    )


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
    stage: CurriculumStage | None = None,
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
        stage=stage,
    )

    with redirect_output(quiet):
        wrapper.reset(
            seed=seed
        )

    if (
        int(
            wrapper.max_episode_length
        )
        != MAX_EPISODE_STEPS
    ):
        wrapper.close()

        raise RuntimeError(
            f"Expected "
            f"{MAX_EPISODE_STEPS} "
            "episode steps, got "
            f"{wrapper.max_episode_length}."
        )

    return wrapper


# =============================================================================
# CURRICULUM HELPERS
# =============================================================================

def stage_gate_passed(
    stage: CurriculumStage,
    stage_iteration: int,
    evaluation: dict[str, float],
) -> bool:
    """Our advancement rule; the authors' scripts do not define a gate."""
    if stage_iteration < stage.min_iterations:
        return False
    if evaluation["eval/full_hold_fraction"] < stage.required_full_hold_fraction:
        return False
    if evaluation["eval/mean_hold_steps"] < stage.required_mean_hold_steps:
        return False
    if evaluation["eval/mean_static_solved_fraction"] < stage.required_solved_fraction:
        return False
    return True


def activate_curriculum_stage(
    *,
    stage: CurriculumStage,
    runner,
    train_env: BaodingRslRlWrapper,
    eval_env: BaodingRslRlWrapper,
    seed: int,
    quiet: bool,
) -> None:
    train_env.set_curriculum_stage(
        stage
    )
    eval_env.set_curriculum_stage(
        stage
    )

    with redirect_output(quiet):
        train_env.reset(
            seed=(
                seed
                + stage.index
                * 1_000
            )
        )

    runner.alg.actor.reset()
    runner.alg.critic.reset()

    # Clear diagnostics belonging to any partial episode from the previous
    # curriculum stage.
    train_env.pop_training_diagnostics()


# =============================================================================
# TRAINING
# =============================================================================

def stage_policy_name(stage: CurriculumStage) -> str:
    if stage.index == 1:
        return "my_baoding_rsi_static_policy.pt"
    return f"my_baoding_rsi_stage{stage.index:02d}_policy.pt"


def load_previous_stage(runner, env: BaodingRslRlWrapper, path: Path, stage: CurriculumStage) -> None:
    """Continue the actor, critic, normalizers and optimizer from stage N-1."""
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    infos = checkpoint.get("infos", {})
    previous_stage = infos.get("curriculum_stage", {})
    if previous_stage.get("index") != stage.index - 1:
        raise RuntimeError(
            f"{path} belongs to stage {previous_stage.get('index')!r}; "
            f"stage {stage.index} requires a stage {stage.index - 1} checkpoint."
        )
    if "optimizer_state_dict" not in checkpoint:
        raise RuntimeError("Previous checkpoint has no optimizer state.")
    try:
        runner.load(str(path), load_optimizer=True)
    except TypeError:
        runner.load(str(path))
    env.restore_reward_normalizer(infos)


def train(args: argparse.Namespace) -> None:
    # Omitting --stage runs the complete curriculum. An explicit stage runs
    # only that stage and requires the preceding checkpoint when stage > 1.
    stages = (
        CURRICULUM_STAGES if args.stage is None
        else (CURRICULUM_STAGES[args.stage - 1],)
    )
    first_stage = stages[0]
    set_runtime_reset_randomization(first_stage)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.stage is None and args.policy_path is None:
        # A no-argument curriculum run must preserve an earlier stage-one
        # checkpoint rather than replacing it at the initial evaluation.
        run_name = time.strftime("run_%Y%m%d_%H%M%S", time.gmtime())
        run_name += f"_{os.getpid()}"
        first_policy_path = resolve_path(str(
            REPO_ROOT / "baoding_rsi_curriculum_runs" / run_name
            / stage_policy_name(first_stage)
        ))
    else:
        first_policy_path = resolve_path(
            args.policy_path or stage_policy_name(first_stage)
        )
    output_dir = first_policy_path.parent
    metrics_path = (
        default_metrics_path(first_policy_path)
        if args.metrics_path is None else resolve_path(args.metrics_path)
    )
    source_path = None
    if first_stage.index > 1:
        preceding = output_dir / stage_policy_name(
            CURRICULUM_STAGES[first_stage.index - 2]
        )
        candidates = (
            [resolve_path(args.from_policy)] if args.from_policy else
            [preceding.with_name(f"{preceding.stem}_passed{preceding.suffix}")]
        )
        source_path = next((path for path in candidates if path.is_file()), None)
        if source_path is None:
            raise FileNotFoundError(
                f"Stage {first_stage.index} requires a stage "
                f"{first_stage.index - 1} checkpoint. Looked for: {candidates}. "
                "Specify --from-policy if it has a different filename."
            )
    elif args.from_policy:
        raise ValueError("--from-policy requires an explicit --stage 2..5.")

    make_env, load_rl_cfg = import_mjlab_tools(quiet=args.quiet)
    from rsl_rl.runners import OnPolicyRunner
    train_env = None
    eval_env = None
    try:
        train_env = make_wrapped_env(
            make_env, args.env_id, args.device, args.num_envs, args.seed,
            args.quiet, NORMALIZE_TRAINING_REWARDS, stage=first_stage,
        )
        with preserve_global_rng_state():
            eval_env = make_wrapped_env(
                make_env, args.env_id, args.device, EVAL_PHASE_COUNT,
                args.seed + 10_000, args.quiet, False, stage=first_stage,
            )
        train_cfg = load_rl_cfg(args.env_id)
        train_cfg.obs_groups = {
            "policy": ("policy",), "critic": ("policy",), "actor": ("policy",),
        }
        train_cfg_dict = asdict(train_cfg)
        apply_training_config(train_cfg_dict)
        configure_lstm_models(train_cfg_dict)
        with redirect_output(args.quiet):
            runner = OnPolicyRunner(
                env=train_env, train_cfg=train_cfg_dict, log_dir=None,
                device=args.device,
            )
        actor, _ = validate_lstm_runner(runner)
        algorithm = runner.alg
        adaptive_lr_state = install_adaptive_lr_bounds(algorithm)
        patch_recurrent_critic_bootstrap(algorithm)
        optimizer = algorithm.optimizer
        optimizer.defaults["eps"] = ADAM_EPSILON
        for group in optimizer.param_groups:
            group["eps"] = ADAM_EPSILON
        if source_path is not None:
            load_previous_stage(runner, train_env, source_path, first_stage)
        update_storage = patch_runner_for_metrics(
            runner, adaptive_lr_state=adaptive_lr_state
        )

        saved_config = build_saved_config(args, first_stage)
        saved_config["curriculum"] = [
            {**asdict(stage), "effective_max_iterations": (
                args.max_iterations or stage.max_iterations
            )} for stage in stages
        ]
        saved_config["training"]["maximum_total_iterations"] = sum(
            args.max_iterations or stage.max_iterations for stage in stages
        )
        saved_config["evaluation"]["maximum_num_evals"] = sum(
            1 + math.ceil(
                (args.max_iterations or stage.max_iterations) / stage.eval_every
            )
            for stage in stages
        )
        saved_config["checkpointing"] = {
            "current_policy_path": str(first_policy_path),
            "stage_policy_path": str(first_policy_path),
            "active_stage": None,
            "selection_metric": None,
            "progression_checkpoint": "passed gate checkpoint",
            "source_policy_path": str(source_path) if source_path else None,
            "stage_outputs": {},
        }
        saved_config["curriculum_results"] = []
        metric_rows: list[dict[str, Any]] = []
        steps_per_iteration = args.num_envs * NUM_STEPS_PER_ENV
        global_iteration = 0
        global_eval_index = -1
        start_time = time.time()
        previous_passed_path = source_path
        curriculum_completed = True

        print("=" * 80)
        print("MJLAB BAODING CURRICULUM PPO-LSTM TRAINING")
        print("=" * 80)
        print(f"environment:             {args.env_id}")
        print(f"device:                  {args.device}")
        print(f"policy path:             {first_policy_path}")
        print(f"metrics path:            {metrics_path}")
        print(f"curriculum stages:       {len(stages)}")
        print(f"maximum iterations:      {saved_config['training']['maximum_total_iterations']}")
        print(f"training environments:   {args.num_envs}")
        print(f"steps per environment:   {NUM_STEPS_PER_ENV}")
        print(f"evaluation environments: {first_stage.eval_num_envs}")
        print(f"LSTM hidden size:         {RNN_HIDDEN_DIM}")
        print(f"learning rate:            {LEARNING_RATE}")
        print(f"learning epochs:          {NUM_LEARNING_EPOCHS}")
        print(f"mini-batches:             {NUM_MINI_BATCHES}")
        print(f"PPO clip:                 {CLIP_PARAM}")
        print("=" * 80)

        for stage_index, stage in enumerate(stages):
            if stage_index:
                # The previous stage's passing policy, optimizer and reward
                # normalizer are the exact starting point for the new task.
                set_runtime_reset_randomization(stage)
                load_previous_stage(runner, train_env, previous_passed_path, stage)
            activate_curriculum_stage(
                stage=stage, runner=runner, train_env=train_env, eval_env=eval_env,
                seed=args.seed, quiet=args.quiet,
            )
            policy_path = (
                first_policy_path if stage_index == 0
                else output_dir / stage_policy_name(stage)
            )
            passed_path = policy_path.with_name(
                f"{policy_path.stem}_passed{policy_path.suffix}"
            )
            if source_path in (policy_path, passed_path):
                raise ValueError("The input checkpoint must not be overwritten.")
            saved_config["checkpointing"]["current_policy_path"] = str(policy_path)
            saved_config["checkpointing"]["stage_policy_path"] = str(policy_path)
            saved_config["checkpointing"]["active_stage"] = stage.slug
            maximum_iterations = args.max_iterations or stage.max_iterations
            evaluation_seed = args.seed + 20_000 + stage.index * 1_000
            stage_iteration = 0
            stage_eval_index = 0
            consecutive_passes = 0
            train_metrics = {
                "train/num_updates": 0.0,
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            print("=" * 80)
            print(f"MJLAB BAODING STAGE {stage.index}/5: {stage.slug}")
            print(stage.description)
            print("reward weights:         position=1.0, alive=1.0, "
                  f"solved=5.0, drop={DROP_PENALTY}")
            print(f"Input: {previous_passed_path or 'new policy'}")
            print(f"Latest: {policy_path} | passed: {passed_path}")
            print(f"Budget: up to {maximum_iterations} updates x {steps_per_iteration} "
                  f"= {maximum_iterations * steps_per_iteration} interactions")
            print(f"Gate after {stage.min_iterations} updates, "
                  f"{stage.required_consecutive_passes} evaluations: "
                  f"full>={stage.required_full_hold_fraction:.2f}, "
                  f"held>={stage.required_mean_hold_steps:.0f}, "
                  f"solved>={stage.required_solved_fraction:.2f}")
            print(f"Evaluation: {stage.eval_num_envs} episodes, seed {evaluation_seed}")
            print("=" * 80)

            while True:
                global_eval_index += 1
                evaluation = evaluate_policy(
                    runner=runner, eval_env=eval_env, seed=evaluation_seed,
                    device=args.device, quiet=args.quiet,
                )
                gate_passed = stage_gate_passed(stage, stage_iteration, evaluation)
                consecutive_passes = consecutive_passes + 1 if gate_passed else 0
                stage_completed = consecutive_passes >= stage.required_consecutive_passes
                if not math.isfinite(evaluation["eval/mean_reward"]):
                    raise RuntimeError("Evaluation mean reward is not finite.")
                elapsed = time.time() - start_time
                row = make_metric_row(
                    stage=stage, stage_eval_index=stage_eval_index,
                    stage_iteration=stage_iteration, gate_passed=gate_passed,
                    consecutive_passes=consecutive_passes,
                    eval_index=global_eval_index, iteration=global_iteration,
                    env_steps=global_iteration * steps_per_iteration,
                    elapsed_sec=elapsed, eval_metrics=evaluation,
                    train_metrics=train_metrics,
                )
                row["eval/stage_completed"] = float(stage_completed)
                metric_rows.append(row)
                infos = train_env.checkpoint_infos()
                runner.save(str(policy_path), infos=infos)
                if stage_completed:
                    runner.save(str(passed_path), infos=infos)
                saved_config["checkpointing"]["stage_outputs"][stage.slug] = {
                    "latest": str(policy_path),
                    "passed": str(passed_path) if stage_completed else None,
                }
                save_metrics(metrics_path, saved_config, metric_rows)
                print_evaluation(
                    stage=stage, stage_eval_index=stage_eval_index,
                    maximum_stage_evals=math.ceil(maximum_iterations / stage.eval_every),
                    stage_iteration=stage_iteration,
                    consecutive_passes=consecutive_passes,
                    eval_index=global_eval_index, iteration=global_iteration,
                    env_steps=global_iteration * steps_per_iteration,
                    evaluation=evaluation,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    approx_kl=train_metrics.get("train/approx_kl_mean"),
                    elapsed_sec=elapsed,
                )
                if stage_completed or stage_iteration >= maximum_iterations:
                    break

                chunk = min(stage.eval_every, maximum_iterations - stage_iteration)
                for values in update_storage.values():
                    values.clear()
                with redirect_output(args.quiet):
                    runner.learn(
                        num_learning_iterations=chunk,
                        init_at_random_ep_len=(stage_iteration == 0),
                    )
                train_metrics = average_update_metrics(update_storage["updates"])
                train_metrics.update(summarize_samples(
                    update_storage["kl_values"], prefix="train/approx_kl"
                ))
                train_metrics.update(summarize_samples(
                    update_storage["grad_norm_values"],
                    prefix="train/grad_norm_after_clip"
                ))
                train_metrics.update(actor_std_metrics(actor))
                train_metrics.update(train_env.pop_training_diagnostics())
                train_metrics["train/learning_rate_final"] = float(
                    optimizer.param_groups[0]["lr"]
                )
                stage_iteration += chunk
                global_iteration += chunk
                stage_eval_index += 1

            saved_config["curriculum_results"].append({
                "stage_index": stage.index, "stage_name": stage.slug,
                "completed": stage_completed,
                "stage_iterations": stage_iteration,
                "global_iteration": global_iteration,
                "env_steps": global_iteration * steps_per_iteration,
                "consecutive_passes": consecutive_passes,
                "checkpoint_path": str(policy_path),
                "passed_checkpoint_path": str(passed_path) if stage_completed else None,
                "latest_checkpoint_path": str(policy_path),
                "final_evaluation": evaluation,
            })
            save_metrics(metrics_path, saved_config, metric_rows)
            if not stage_completed:
                curriculum_completed = False
                print(f"Stage {stage.index} did not pass its gate within "
                      f"{maximum_iterations} updates. Curriculum stopped.")
                break
            print(f"Stage {stage.index} passed. Continuing from: {passed_path}")
            previous_passed_path = passed_path

        saved_config["checkpointing"]["active_stage"] = None
        saved_config["selected_stages_completed"] = curriculum_completed
        saved_config["curriculum_completed"] = (
            curriculum_completed and args.stage is None
        )
        save_metrics(metrics_path, saved_config, metric_rows)
        print("=" * 80)
        label = "CURRICULUM" if args.stage is None else f"STAGE {args.stage}"
        print(f"{label} {'COMPLETE' if curriculum_completed else 'INCOMPLETE'}")
        print(f"Metrics: {metrics_path}")
        print("=" * 80)
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
        default=None,
    )
    parser.add_argument(
        "--stage", type=int, choices=range(1, 6), default=None,
        help="Run only this stage; omitted runs stages 1 through 5 in order.",
    )
    parser.add_argument(
        "--from-policy", default=None,
        help="Input checkpoint for a single-stage run (N>1); defaults to prior passed policy.",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Override the 1000-update maximum for each selected stage.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=NUM_ENVS,
        help="Training environments (default: 256, as in the uploaded stage-one script).",
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
            argparse
            .BooleanOptionalAction
        ),
        default=True,
    )

    return parser.parse_args()


def validate_config(
    args: argparse.Namespace,
) -> None:
    """Keep only checks that prevent an invalid run from starting."""
    if (
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested device "
            f"{args.device}, but CUDA "
            "is not available."
        )

    if args.num_envs <= 0 or args.num_envs % NUM_MINI_BATCHES != 0:
        raise ValueError(
            "--num-envs must be positive and divisible by NUM_MINI_BATCHES."
        )

    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")
    for stage in CURRICULUM_STAGES:
        invalid_schedule = (
            stage.max_iterations <= 0
            or stage.min_iterations < 0
            or stage.min_iterations > stage.max_iterations
            or stage.eval_every <= 0
            or stage.eval_num_envs <= 0
            or stage.required_consecutive_passes <= 0
        )

        if invalid_schedule:
            raise ValueError(
                "Invalid iteration schedule "
                "for curriculum stage "
                f"{stage.index}."
            )
        if args.stage is None or args.stage == stage.index:
            earliest_pass = (
                math.ceil(stage.min_iterations / stage.eval_every)
                + stage.required_consecutive_passes - 1
            ) * stage.eval_every
            budget = args.max_iterations or stage.max_iterations
            if budget < earliest_pass:
                raise ValueError(
                    f"Stage {stage.index} needs at least {earliest_pass} updates "
                    "for its minimum training and consecutive gate evaluations; "
                    f"the configured budget is {budget}. Increase --max-iterations."
                )


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()
