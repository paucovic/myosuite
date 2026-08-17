#!/usr/bin/env python3
"""
Train a five-fingertip HandReach policy with mjlab + RSL-RL.

Run from the repo root:

    python scripts/mjlab/hand_reach/train_handreach.py --policy-path my_handreach_policy.pt

This saves:
    <policy_name>.pt
    <policy_name>_metrics.json
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

from register_handreach import (
    TASK_ID,
    REACH_THD,
    FINGERTIP_SITE_NAMES,
    NUM_FINGERTIPS,
    FINGERTIP_POS_DIM,
    register_hand_reach_task,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# USER TRAINING CONFIG
# =============================================================================

ITERATIONS = 1000
NUM_ENVS = 256
NUM_STEPS_PER_ENV = 64

NUM_EVALS = 50
EVAL_NUM_ENVS = 64
EVAL_STEPS = 200

LEARNING_RATE = 1e-4
MIN_ADAPTIVE_LR = 1e-6
MAX_ADAPTIVE_LR = 3e-4

SCHEDULE = "adaptive"
NUM_LEARNING_EPOCHS = 5
NUM_MINI_BATCHES = 4

CLIP_PARAM = 0.2
DESIRED_KL = 0.01
MAX_GRAD_NORM = 1.0

# Set to None to keep the default value from the registered RSL-RL config.
ENTROPY_COEF = 0.001


# =============================================================================
# USER TARGET / REWARD CONFIG
# =============================================================================

RANDOMIZE_TARGET = False

# Must match register_handreach.py:
#   THtip xyz, IFtip xyz, MFtip xyz, RFtip xyz, LFtip xyz
FINGER_NAMES = FINGERTIP_SITE_NAMES

# Nominal five-fingertip targets from the MJX HandReach fixed target.
NOMINAL_FINGERTIP_TARGETS = (
    (-0.165, -0.537, 1.495),
    (-0.151, -0.547, 1.455),
    (-0.146, -0.547, 1.447),
    (-0.148, -0.543, 1.445),
    (-0.148, -0.528, 1.434),
)


# Structured target sampling:
#   target_i = nominal_i + shared_hand_offset + per_finger_jitter_i
# This avoids asking for five independent, possibly impossible fingertip targets.
SHARED_TARGET_OFFSET_LOW = (-0.01, -0.01, -0.010)
SHARED_TARGET_OFFSET_HIGH = (0.01, 0.01, 0.010)

PER_FINGER_TARGET_JITTER_LOW = (-0.03, -0.03, -0.03)
PER_FINGER_TARGET_JITTER_HIGH = (0.03, 0.01, 0.03)

# obs["policy"] layout after the five-fingertip register_handreach.py:
#   qpos        0:23
#   qvel       23:46
#   tip_pos    46:61   # 15 = 5 fingertips * xyz
#   target_pos 61:76   # 15 = 5 targets * xyz
#   target_err 76:91   # 15 = 5 target errors * xyz
#   act        91:130  # 39 muscles
QPOS_SLICE = slice(0, 23)
QVEL_SLICE = slice(23, 46)
TIP_POS_SLICE = slice(46, 61)
TARGET_POS_SLICE = slice(61, 76)
TARGET_ERR_SLICE = slice(76, 91)
ACT_SLICE = slice(91, 130)

OBS_DIM = 130
ACTION_DIM = 39

# Additive five-fingertip reward.
#MAX_DISTANCE_PENALTY_SCALE = 20.0
DISTANCE_REWARD_SCALE = 20.0
#PROXIMITY_REWARD_SCALE = 2.0
#PROXIMITY_REWARD_SHARPNESS = 25.0
ROBUST_MAX_DISTANCE_PENALTY_SCALE = 0.0

VELOCITY_PENALTY_SCALE = 0.0
ACTION_PENALTY_SCALE = 0.001

SOLVED_DISTANCE_TH = 0.03
#VERY_SOLVED_DISTANCE_TH = REACH_THD * 0.5
ROBUST_MAX_DISTANCE_TH = 0.03

SOLVED_BONUS_VALUE = 5.0
#VERY_SOLVED_BONUS_VALUE = 2.0
#ALL_SOLVED_BONUS_VALUE = 5.0




# =============================================================================
# OUTPUT REDIRECTION
# =============================================================================

@contextlib.contextmanager
def redirect_output(enabled: bool):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
            yield


def import_mjlab_tools(quiet: bool):
    with redirect_output(quiet):
        import myosuite  # noqa: F401
        import mjlab.tasks.registry as registry

        from myosuite.core.registry import make_env

        register_hand_reach_task(quiet=True)

    return make_env, registry.load_rl_cfg


# =============================================================================
# MJLAB -> RSL-RL WRAPPER
# =============================================================================

class HandReachRslRlWrapper:
    """
    Adapter between mjlab HandReach and RSL-RL.

    mjlab step returns:
        obs_dict, reward, terminated, truncated, extras

    RSL-RL expects:
        obs, rewards, dones, infos

    This wrapper:
        1. samples five 3D targets per env
        2. inserts flattened target_pos and target_err into obs["policy"]
        3. computes an additive five-fingertip shaped reach reward
    """

    def __init__(self, env: Any, seed: int) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.num_actions = int(env.action_space.shape[-1])
        self.cfg = getattr(env, "cfg", None)

        self.target_rng = torch.Generator(device="cpu")
        self.target_rng.manual_seed(seed)

        self.target_positions = torch.zeros(
            (self.num_envs, NUM_FINGERTIPS, 3),
            device=self.device,
            dtype=torch.float32,
        )

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

    def reset(self, seed: int | None = None) -> TensorDict:
        if seed is not None:
            self.target_rng.manual_seed(int(seed) + 12345)

            try:
                self.env.reset(seed=seed)
            except TypeError:
                self.env.reset()
        else:
            self.env.reset()

        self._resample_targets(torch.ones(self.num_envs, dtype=torch.bool))
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        obs_dict = self.env.observation_manager.compute()
        obs_dict = self._replace_target_observation(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        obs_dict, _, terminated, truncated, extras = self.env.step(actions)

        obs_dict = self._replace_target_observation(obs_dict)
        rewards, reward_info = self._reward(obs_dict, actions)

        done_mask = terminated | truncated
        dones = done_mask.to(dtype=torch.long)

        self._resample_targets(done_mask)

        obs_dict = self._replace_target_observation(obs_dict)
        obs = TensorDict(obs_dict, batch_size=[self.num_envs])

        infos = {
            "episode": {
                "reward": reward_info["reward"].mean(),
                "distance_sum": reward_info["distance_sum"].mean(),
                "distance_mean": reward_info["distance_mean"].mean(),
                "distance_max": reward_info["distance_max"].mean(),
                "solved": reward_info["solved"].mean(),
                "distance_penalty": reward_info["distance_penalty"].mean(),
                "solved_bonus": reward_info["solved_bonus"].mean(),
                "robust_worst_finger_penalty": reward_info[
                    "robust_worst_finger_penalty"
                ].mean(),
                "action_penalty": reward_info["action_penalty"].mean(),
                "velocity_penalty": reward_info["velocity_penalty"].mean(),
            }
        }

        if isinstance(extras, dict):
            if "time_outs" in extras:
                infos["time_outs"] = extras["time_outs"]

            if "log" in extras:
                infos["log"] = extras["log"]

        return obs, rewards, dones, infos

    def _nominal_targets(self, device: str | torch.device) -> torch.Tensor:
        return torch.tensor(
            NOMINAL_FINGERTIP_TARGETS,
            device=device,
            dtype=torch.float32,
        )

    def _sample_targets(self, count: int) -> torch.Tensor:
        nominal = self._nominal_targets(device="cpu")

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
            (count, 1, 3),
            generator=self.target_rng,
            device="cpu",
            dtype=torch.float32,
        )
        shared_offsets = shared_low + (shared_high - shared_low) * shared_samples

        jitter_samples = torch.rand(
            (count, NUM_FINGERTIPS, 3),
            generator=self.target_rng,
            device="cpu",
            dtype=torch.float32,
        )
        per_finger_jitter = jitter_low + (jitter_high - jitter_low) * jitter_samples

        targets = nominal.unsqueeze(0) + shared_offsets + per_finger_jitter
        return targets.to(self.device)

    def _resample_targets(self, mask: torch.Tensor) -> None:
        if not RANDOMIZE_TARGET:
            nominal = self._nominal_targets(device=self.device)
            self.target_positions[:] = nominal.unsqueeze(0)
            return

        mask = mask.to(device=self.device, dtype=torch.bool)

        if not bool(mask.any()):
            return

        count = int(mask.sum().detach().cpu())
        self.target_positions[mask] = self._sample_targets(count)

    def _replace_target_observation(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        obs_dict = dict(obs_dict)
        policy_obs = obs_dict["policy"].clone()

        tip_pos_flat = policy_obs[:, TIP_POS_SLICE]
        tip_pos = tip_pos_flat.reshape(self.num_envs, NUM_FINGERTIPS, 3)
        target_err = self.target_positions - tip_pos

        policy_obs[:, TARGET_POS_SLICE] = self.target_positions.reshape(
            self.num_envs,
            FINGERTIP_POS_DIM,
        )
        policy_obs[:, TARGET_ERR_SLICE] = target_err.reshape(
            self.num_envs,
            FINGERTIP_POS_DIM,
        )

        obs_dict["policy"] = policy_obs
        return obs_dict

    def _reward(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_obs = obs_dict["policy"]

        qvel = policy_obs[:, QVEL_SLICE]
        target_err = policy_obs[:, TARGET_ERR_SLICE].reshape(
            self.num_envs,
            NUM_FINGERTIPS,
            3,
        )

        per_finger_distance = torch.linalg.norm(target_err, dim=-1)
        distance_sum = per_finger_distance.sum(dim=-1)
        distance_mean = per_finger_distance.mean(dim=-1)
        distance_max = per_finger_distance.max(dim=-1).values

        # Dense shaping: make all fingertips closer.
        distance_penalty = -DISTANCE_REWARD_SCALE * distance_sum

        # Single solved condition: every fingertip must be under the threshold.
        solved = distance_max < SOLVED_DISTANCE_TH
        solved_bonus = SOLVED_BONUS_VALUE * solved.float()

        # One hinge penalty for the same solved threshold.
        # Below the threshold: no penalty.
        # Above the threshold: linearly punish the worst fingertip.
        robust_max_distance_excess = torch.clamp(
            distance_max - SOLVED_DISTANCE_TH,
            min=0.0,
        )

        robust_worst_finger_penalty = (
            ROBUST_MAX_DISTANCE_PENALTY_SCALE * robust_max_distance_excess
        )

        velocity_penalty = VELOCITY_PENALTY_SCALE * torch.mean(
            torch.abs(qvel),
            dim=-1,
        )

        action_penalty = ACTION_PENALTY_SCALE * torch.mean(
            actions.square(),
            dim=-1,
        )

        rewards = (
            distance_penalty
            + solved_bonus
            - robust_worst_finger_penalty
            - velocity_penalty
            - action_penalty
        )

        info = {
            "reward": rewards.detach(),
            "per_finger_distance": per_finger_distance.detach(),
            "distance_sum": distance_sum.detach(),
            "distance_mean": distance_mean.detach(),
            "distance_max": distance_max.detach(),
            "distance_penalty": distance_penalty.detach(),
            "solved": solved.float().detach(),
            "solved_bonus": solved_bonus.detach(),
            "robust_worst_finger_penalty": robust_worst_finger_penalty.detach(),
            "action_penalty": action_penalty.detach(),
            "velocity_penalty": velocity_penalty.detach(),
        }

        return rewards, info


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def remove_unsupported_rsl_rl_keys(train_cfg: dict[str, Any]) -> None:
    for section in ("actor", "critic"):
        section_cfg = train_cfg.get(section)

        if not isinstance(section_cfg, dict):
            continue

        for key in list(section_cfg.keys()):
            if key == "cnn_cfg" or key.startswith("rnn"):
                section_cfg.pop(key)


def set_key_recursively(data: dict[str, Any], key: str, value: Any) -> bool:
    found = False

    if key in data:
        data[key] = value
        found = True

    for child in data.values():
        if isinstance(child, dict):
            found = set_key_recursively(child, key, value) or found

    return found


def set_training_value(
    train_cfg: dict[str, Any],
    possible_keys: tuple[str, ...],
    value: Any,
) -> None:
    if value is None:
        return

    for key in possible_keys:
        if set_key_recursively(train_cfg, key, value):
            return


def apply_training_config(train_cfg: dict[str, Any]) -> None:
    train_cfg["num_steps_per_env"] = NUM_STEPS_PER_ENV
    train_cfg["logger"] = "tensorboard"

    set_training_value(train_cfg, ("learning_rate", "lr"), LEARNING_RATE)
    set_training_value(train_cfg, ("num_learning_epochs",), NUM_LEARNING_EPOCHS)
    set_training_value(train_cfg, ("num_mini_batches", "num_minibatches"), NUM_MINI_BATCHES)
    set_training_value(train_cfg, ("clip_param", "clip_range"), CLIP_PARAM)
    set_training_value(train_cfg, ("desired_kl",), DESIRED_KL)
    set_training_value(train_cfg, ("max_grad_norm",), MAX_GRAD_NORM)
    set_training_value(train_cfg, ("entropy_coef", "entropy_coeff"), ENTROPY_COEF)

    algorithm_cfg = train_cfg.get("algorithm")

    if not isinstance(algorithm_cfg, dict):
        raise RuntimeError(
            "RSL-RL algorithm configuration was not found in "
            "train_cfg['algorithm']."
        )

    algorithm_cfg["schedule"] = SCHEDULE

def install_adaptive_lr_bounds(alg) -> None:
    """Keep every adaptive PPO optimizer step inside the requested LR range."""
    if SCHEDULE != "adaptive":
        return

    if not (
        0.0
        < MIN_ADAPTIVE_LR
        <= LEARNING_RATE
        <= MAX_ADAPTIVE_LR
    ):
        raise ValueError(
            "Learning rates must satisfy: "
            "0 < MIN_ADAPTIVE_LR <= LEARNING_RATE <= MAX_ADAPTIVE_LR."
        )

    optimizer = getattr(alg, "optimizer", None)

    if optimizer is None or len(optimizer.param_groups) == 0:
        raise RuntimeError("RSL-RL PPO optimizer was not found.")

    if not hasattr(optimizer, "register_step_pre_hook"):
        raise RuntimeError(
            "This PyTorch version does not support optimizer step pre-hooks."
        )

    def clamp_lr_before_optimizer_step(
        _optimizer,
        _args,
        _kwargs,
    ) -> None:
        bounded_lr = min(
            MAX_ADAPTIVE_LR,
            max(MIN_ADAPTIVE_LR, float(alg.learning_rate)),
        )

        # RSL-RL reads this value during the next KL adaptation.
        alg.learning_rate = bounded_lr

        # Adam uses the values stored in its parameter groups.
        for param_group in optimizer.param_groups:
            param_group["lr"] = bounded_lr

    # Clamp the initial value immediately.
    clamp_lr_before_optimizer_step(optimizer, (), {})

    # RSL-RL changes the LR before optimizer.step().
    # This hook then enforces our limits immediately before the update.
    optimizer.register_step_pre_hook(clamp_lr_before_optimizer_step)
# =============================================================================
# FILE HELPERS
# =============================================================================

def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)

    if not path.is_absolute():
        path = REPO_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def default_metrics_path(policy_path: Path) -> Path:
    return policy_path.with_name(f"{policy_path.stem}_metrics.json")


def save_metrics(path: Path, config: dict[str, Any], rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as file:
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

def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None

        return float(value.detach().float().mean().cpu())

    try:
        return float(value)
    except Exception:
        return None


def metric_name(name: str) -> str | None:
    name = name.lower().strip().replace(" ", "_")

    mapping = {
        "value": "train/value",
        "value_loss": "train/value",
        "surrogate": "train/surrogate",
        "surrogate_loss": "train/surrogate",
        "policy_loss": "train/surrogate",
        "entropy": "train/entropy",
        "entropy_loss": "train/entropy",
        "approx_kl": "train/approx_kl",
    }

    return mapping.get(name)


def find_kl_source(runner):
    alg = getattr(runner, "alg", None)

    candidates = []

    if alg is not None:
        candidates.extend(
            [
                getattr(alg, "actor", None),
                getattr(alg, "actor_critic", None),
                getattr(alg, "policy", None),
            ]
        )

    candidates.extend(
        [
            getattr(runner, "actor", None),
            getattr(runner, "actor_critic", None),
            getattr(runner, "policy", None),
        ]
    )

    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "get_kl_divergence"):
            return candidate

    return None


def patch_runner_for_metrics(runner) -> dict[str, list[dict[str, float]]]:
    storage: dict[str, list[Any]] = {
        "updates": [],
        "kl_values": [],
    }

    alg = getattr(runner, "alg", None)

    if alg is None or not hasattr(alg, "update"):
        return storage

    kl_source = find_kl_source(runner)

    if kl_source is not None:
        original_kl_fn = kl_source.get_kl_divergence

        def wrapped_kl_fn(*args, **kwargs):
            kl = original_kl_fn(*args, **kwargs)
            scalar = to_float(kl)

            if scalar is not None:
                storage["kl_values"].append(scalar)

            return kl

        kl_source.get_kl_divergence = wrapped_kl_fn

    original_update = alg.update

    def wrapped_update(*args, **kwargs):
        kl_start = len(storage["kl_values"])
        update_output = original_update(*args, **kwargs)

        record: dict[str, float] = {}

        optimizer = getattr(alg, "optimizer", None)

        if optimizer is not None and len(optimizer.param_groups) > 0:
            record["train/learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )

        if isinstance(update_output, dict):
            for raw_name, raw_value in update_output.items():
                clean_name = metric_name(str(raw_name))
                value = to_float(raw_value)

                if clean_name is not None and value is not None:
                    record[clean_name] = value

        elif isinstance(update_output, (tuple, list)):
            names = [
                "train/value",
                "train/surrogate",
                "train/entropy",
                "train/approx_kl",
            ]

            for index, raw_value in enumerate(update_output[: len(names)]):
                value = to_float(raw_value)

                if value is not None:
                    record[names[index]] = value

        new_kl_values = storage["kl_values"][kl_start:]

        if len(new_kl_values) > 0:
            record["train/approx_kl"] = float(
                sum(new_kl_values) / len(new_kl_values)
            )

        if len(record) > 0:
            storage["updates"].append(record)

        return update_output

    alg.update = wrapped_update
    return storage


def average_update_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if len(records) == 0:
        return {"train/num_updates": 0.0}

    keys = sorted({key for record in records for key in record})

    averaged = {
        "train/num_updates": float(len(records)),
    }

    for key in keys:
        values = [record[key] for record in records if key in record]
        averaged[key] = float(sum(values) / len(values))

    return averaged


# =============================================================================
# POLICY ACTION / EVALUATION
# =============================================================================

def unpack_action(action: Any) -> torch.Tensor:
    if isinstance(action, torch.Tensor):
        return action

    if isinstance(action, TensorDict):
        for key in ("action", "actions"):
            if key in action:
                return action[key]

    if isinstance(action, dict):
        for key in ("action", "actions"):
            if key in action:
                return action[key]

    if isinstance(action, tuple):
        return unpack_action(action[0])

    raise TypeError(f"Unsupported action output type: {type(action)}")


def get_policy_action(runner, obs: TensorDict, device: str) -> torch.Tensor:
    with torch.no_grad():
        if hasattr(runner, "get_inference_policy"):
            policy = runner.get_inference_policy(device=device)

            for policy_input in (obs, obs["policy"]):
                try:
                    return unpack_action(policy(policy_input))
                except Exception:
                    pass

        alg = getattr(runner, "alg", None)

        if alg is not None:
            for actor_name in ("actor", "actor_critic"):
                actor = getattr(alg, actor_name, None)

                if actor is None:
                    continue

                if hasattr(actor, "eval"):
                    actor.eval()

                for method_name in ("act_inference", "act", "forward", "__call__"):
                    method = getattr(actor, method_name, None)

                    if method is None:
                        continue

                    for policy_input in (obs, obs["policy"]):
                        try:
                            return unpack_action(method(policy_input))
                        except Exception:
                            pass

    raise RuntimeError("Could not get policy action from this RSL-RL version.")


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
    distance_sums = []
    distance_means = []
    distance_maxes = []
    solved_values = []

    for _ in range(EVAL_STEPS):
        actions = get_policy_action(runner, obs, device)
        obs, reward, _, _ = eval_env.step(actions)

        policy_obs = obs["policy"]
        target_err = policy_obs[:, TARGET_ERR_SLICE].reshape(
            eval_env.num_envs,
            NUM_FINGERTIPS,
            3,
        )

        per_finger_distance = torch.linalg.norm(target_err, dim=-1)
        distance_sum = per_finger_distance.sum(dim=-1)
        distance_mean = per_finger_distance.mean(dim=-1)
        distance_max = per_finger_distance.max(dim=-1).values

        solved = distance_max < SOLVED_DISTANCE_TH

        rewards.append(reward.detach())
        distance_sums.append(distance_sum.detach())
        distance_means.append(distance_mean.detach())
        distance_maxes.append(distance_max.detach())
        solved_values.append(solved.float().detach())

    reward_tensor = torch.stack(rewards)
    distance_sum_tensor = torch.stack(distance_sums)
    distance_mean_tensor = torch.stack(distance_means)
    distance_max_tensor = torch.stack(distance_maxes)
    solved_tensor = torch.stack(solved_values)

    return {
        "eval/mean_reward": float(reward_tensor.sum(dim=0).mean().cpu()),
        "eval/reward_per_step": float(reward_tensor.mean().cpu()),
        "eval/distance": float(distance_mean_tensor.mean().cpu()),
        "eval/distance_sum": float(distance_sum_tensor.mean().cpu()),
        "eval/distance_mean": float(distance_mean_tensor.mean().cpu()),
        "eval/distance_max": float(distance_max_tensor.mean().cpu()),
        "eval/distance_sum_last": float(distance_sum_tensor[-1].mean().cpu()),
        "eval/distance_mean_last": float(distance_mean_tensor[-1].mean().cpu()),
        "eval/distance_max_last": float(distance_max_tensor[-1].mean().cpu()),
        "eval/solved_fraction": float(solved_tensor.mean().cpu()),
    }


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
# TRAINING
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

    wrapper = HandReachRslRlWrapper(env=env, seed=seed)

    with redirect_output(quiet):
        wrapper.reset(seed=seed)

    return wrapper


def build_saved_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "env_id": args.env_id,
        "seed": args.seed,
        "device": args.device,
        "training": {
            "iterations": ITERATIONS,
            "num_envs": NUM_ENVS,
            "num_steps_per_env": NUM_STEPS_PER_ENV,
            "learning_rate": LEARNING_RATE,
            "min_adaptive_lr": MIN_ADAPTIVE_LR,
            "max_adaptive_lr": MAX_ADAPTIVE_LR,
            "schedule": SCHEDULE,
            "num_learning_epochs": NUM_LEARNING_EPOCHS,
            "num_mini_batches": NUM_MINI_BATCHES,
            "clip_param": CLIP_PARAM,
            "desired_kl": DESIRED_KL,
            "max_grad_norm": MAX_GRAD_NORM,
            "entropy_coef": ENTROPY_COEF,
        },
        "evaluation": {
            "num_evals": NUM_EVALS,
            "eval_num_envs": EVAL_NUM_ENVS,
            "eval_steps": EVAL_STEPS,
            "fixed_seed": args.seed + 20_000,
        },
        "target": {
            "randomize_target": RANDOMIZE_TARGET,
            "finger_names": FINGER_NAMES,
            "nominal_fingertip_targets": NOMINAL_FINGERTIP_TARGETS,
            "shared_target_offset_low": SHARED_TARGET_OFFSET_LOW,
            "shared_target_offset_high": SHARED_TARGET_OFFSET_HIGH,
            "per_finger_target_jitter_low": PER_FINGER_TARGET_JITTER_LOW,
            "per_finger_target_jitter_high": PER_FINGER_TARGET_JITTER_HIGH,
        },
        "reward": {
            "distance_reward_scale": DISTANCE_REWARD_SCALE,
            "robust_max_distance_penalty_scale": ROBUST_MAX_DISTANCE_PENALTY_SCALE,
            "velocity_penalty_scale": VELOCITY_PENALTY_SCALE,
            "action_penalty_scale": ACTION_PENALTY_SCALE,
            "solved_distance_th": SOLVED_DISTANCE_TH,
            "solved_bonus_value": SOLVED_BONUS_VALUE,
        },
        "observation_layout": {
            "qpos": [0, 23],
            "qvel": [23, 46],
            "tip_pos": [46, 61],
            "target_pos": [61, 76],
            "target_err": [76, 91],
            "act": [91, 130],
        },
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    policy_path = resolve_path(args.policy_path)

    if args.metrics_path is None:
        metrics_path = default_metrics_path(policy_path)
    else:
        metrics_path = resolve_path(args.metrics_path)

    make_env, load_rl_cfg = import_mjlab_tools(quiet=args.quiet)

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

    train_cfg = load_rl_cfg(args.env_id)
    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }

    train_cfg_dict = asdict(train_cfg)
    apply_training_config(train_cfg_dict)
    remove_unsupported_rsl_rl_keys(train_cfg_dict)

    with redirect_output(args.quiet):
        runner = OnPolicyRunner(
            env=train_env,
            train_cfg=train_cfg_dict,
            log_dir=None,
            device=args.device,
        )

    alg = getattr(runner, "alg", None)

    if alg is None:
        raise RuntimeError("OnPolicyRunner did not expose runner.alg.")

    install_adaptive_lr_bounds(alg)

    effective_schedule = getattr(alg, "schedule", None)
    optimizer = getattr(alg, "optimizer", None)

    if optimizer is None or len(optimizer.param_groups) == 0:
        raise RuntimeError("RSL-RL PPO optimizer was not found.")

    effective_lr = float(optimizer.param_groups[0]["lr"])

    if effective_schedule != SCHEDULE:
        raise RuntimeError(
            f"Expected PPO schedule {SCHEDULE!r}, but RSL-RL is using "
            f"{effective_schedule!r}."
        )

    lr_tolerance = max(1e-12, abs(LEARNING_RATE) * 1e-6)

    if abs(effective_lr - LEARNING_RATE) > lr_tolerance:
        raise RuntimeError(
            f"Expected optimizer LR {LEARNING_RATE}, but RSL-RL is using "
            f"{effective_lr}."
        )

    update_storage = patch_runner_for_metrics(runner)

    steps_per_iteration = NUM_ENVS * NUM_STEPS_PER_ENV
    eval_every = max(1, ITERATIONS // NUM_EVALS)

    saved_config = build_saved_config(args)
    metric_rows: list[dict[str, float]] = []

    print("=" * 80)
    print("MJLAB FIVE-FINGER HAND REACH TRAINING")
    print("=" * 80)
    print(f"env:                  {args.env_id}")
    print(f"device:               {args.device}")
    print(f"policy path:          {policy_path}")
    print(f"metrics path:         {metrics_path}")
    print(f"iterations:           {ITERATIONS}")
    print(f"num_envs:             {NUM_ENVS}")
    print(f"num_steps_per_env:    {NUM_STEPS_PER_ENV}")
    print(f"PPO schedule:         {effective_schedule}")
    print(  f"adaptive LR range:    "
        f"[{MIN_ADAPTIVE_LR}, {MAX_ADAPTIVE_LR}]"
    )
    print(f"configured LR:        {LEARNING_RATE}")
    print(f"internal PPO LR:      {getattr(alg, 'learning_rate', None)}")
    print(f"optimizer LR:         {effective_lr}")
    print(f"desired KL:           {getattr(alg, 'desired_kl', None)}")
    print(f"learning epochs:      {getattr(alg, 'num_learning_epochs', None)}")
    print(f"mini-batches:         {getattr(alg, 'num_mini_batches', None)}")
    print(f"expected steps:       {ITERATIONS * steps_per_iteration}")
    print(f"num_evals:            {NUM_EVALS}")
    print(f"obs dim:              {OBS_DIM}")
    print(f"action dim:           {ACTION_DIM}")
    print(f"fingertips:           {FINGER_NAMES}")
    print(f"nominal targets:      {NOMINAL_FINGERTIP_TARGETS}")
    print(f"shared offset low:    {SHARED_TARGET_OFFSET_LOW}")
    print(f"shared offset high:   {SHARED_TARGET_OFFSET_HIGH}")
    print(f"finger jitter low:    {PER_FINGER_TARGET_JITTER_LOW}")
    print(f"finger jitter high:   {PER_FINGER_TARGET_JITTER_HIGH}")
    print("=" * 80)

    start_time = time.time()
    completed_iterations = 0
    eval_index = 0

    initial_eval = evaluate_policy(
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
        eval_metrics=initial_eval,
        train_metrics={
            "train/num_updates": 0.0,
            "train/learning_rate": effective_lr,
        },
    )

    metric_rows.append(initial_row)
    save_metrics(metrics_path, saved_config, metric_rows)

    print(
        f"[eval {eval_index:02d}/{NUM_EVALS:02d}] "
        f"iter=0 "
        f"steps=0 "
        f"return={initial_eval['eval/mean_reward']:.3f} "
        #f"reward/step={initial_eval['eval/reward_per_step']:.4f} "
        f"dist_mean={initial_eval['eval/distance_mean']:.4f} "
        f"dist_max={initial_eval['eval/distance_max']:.4f} "
        f"solved={initial_eval['eval/solved_fraction']:.3f}"
    )

    while completed_iterations < ITERATIONS:
        remaining_iterations = ITERATIONS - completed_iterations
        chunk_iterations = min(eval_every, remaining_iterations)

        update_storage["updates"].clear()
        update_storage["kl_values"].clear()

        with redirect_output(args.quiet):
            runner.learn(
                num_learning_iterations=chunk_iterations,
                init_at_random_ep_len=(completed_iterations == 0),
            )

        train_metrics = average_update_metrics(update_storage["updates"])

        completed_iterations += chunk_iterations
        env_steps = completed_iterations * steps_per_iteration
        eval_index += 1

        eval_metrics = evaluate_policy(
            runner=runner,
            eval_env=eval_env,
            seed=args.seed + 20_000,
            device=args.device,
            quiet=args.quiet,
        )

        elapsed_sec = time.time() - start_time

        row = make_metric_row(
            eval_index=eval_index,
            iteration=completed_iterations,
            env_steps=env_steps,
            elapsed_sec=elapsed_sec,
            eval_metrics=eval_metrics,
            train_metrics=train_metrics,
        )

        metric_rows.append(row)

        runner.save(str(policy_path))
        save_metrics(metrics_path, saved_config, metric_rows)

        print(
            f"[eval {eval_index:02d}/{NUM_EVALS:02d}] "
            f"iter={completed_iterations} "
            f"steps={env_steps} "
            f"return={eval_metrics['eval/mean_reward']:.3f} "
            f"dist_mean={eval_metrics['eval/distance_mean']:.4f} "
            f"dist_max={eval_metrics['eval/distance_max']:.4f} "
            f"solved={eval_metrics['eval/solved_fraction']:.3f} "
            f"time={elapsed_sec:.1f}s"
        )

    runner.save(str(policy_path))
    save_metrics(metrics_path, saved_config, metric_rows)

    train_env.close()
    eval_env.close()

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"policy:      {policy_path}")
    print(f"metrics:     {metrics_path}")


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
        default="my_handreach_policy.pt",
    )

    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
    )

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser.parse_args()


def validate_config(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {args.device}, but CUDA is not available."
        )

    if ITERATIONS <= 0:
        raise ValueError("ITERATIONS must be greater than 0.")

    if NUM_ENVS <= 0:
        raise ValueError("NUM_ENVS must be greater than 0.")

    if NUM_STEPS_PER_ENV <= 0:
        raise ValueError("NUM_STEPS_PER_ENV must be greater than 0.")

    if NUM_EVALS <= 0:
        raise ValueError("NUM_EVALS must be greater than 0.")

    if EVAL_NUM_ENVS <= 0:
        raise ValueError("EVAL_NUM_ENVS must be greater than 0.")

    if EVAL_STEPS <= 0:
        raise ValueError("EVAL_STEPS must be greater than 0.")

    if NUM_LEARNING_EPOCHS <= 0:
        raise ValueError("NUM_LEARNING_EPOCHS must be greater than 0.")

    if SCHEDULE not in ("fixed", "adaptive"):
        raise ValueError("SCHEDULE must be either 'fixed' or 'adaptive'.")

    if NUM_MINI_BATCHES <= 0:
        raise ValueError("NUM_MINI_BATCHES must be greater than 0.")

    if NUM_FINGERTIPS != 5:
        raise ValueError("This training script expects exactly five fingertips.")

    if FINGERTIP_POS_DIM != 15:
        raise ValueError("FINGERTIP_POS_DIM must be 15 for five xyz fingertips.")

    if SOLVED_DISTANCE_TH <= 0.0:
        raise ValueError("SOLVED_DISTANCE_TH must be greater than 0.")

    #if VERY_SOLVED_DISTANCE_TH > SOLVED_DISTANCE_TH:
    #    raise ValueError(
    #        "VERY_SOLVED_DISTANCE_TH should be smaller than or equal to "
    #        "SOLVED_DISTANCE_TH."
    #    )

    #if VERY_SOLVED_BONUS_VALUE < SOLVED_BONUS_VALUE:
    #    raise ValueError(
    #        "VERY_SOLVED_BONUS_VALUE should be greater than or equal to "
    #        "SOLVED_BONUS_VALUE."
    #    )

    #if PROXIMITY_REWARD_SCALE < 0.0:
    #    raise ValueError("PROXIMITY_REWARD_SCALE should be non-negative.")

    #if PROXIMITY_REWARD_SHARPNESS <= 0.0:
    #    raise ValueError("PROXIMITY_REWARD_SHARPNESS must be greater than 0.")

    if len(NOMINAL_FINGERTIP_TARGETS) != NUM_FINGERTIPS:
        raise ValueError(
            "NOMINAL_FINGERTIP_TARGETS must contain one xyz target per fingertip."
        )

    for target in NOMINAL_FINGERTIP_TARGETS:
        if len(target) != 3:
            raise ValueError("Each nominal fingertip target must have xyz values.")

    for low_name, high_name, low_values, high_values in (
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
    ):
        if len(low_values) != 3 or len(high_values) != 3:
            raise ValueError(f"{low_name} and {high_name} must have xyz values.")

        for low, high in zip(low_values, high_values):
            if high <= low:
                raise ValueError(
                    f"Each value in {high_name} must be greater than the "
                    f"corresponding value in {low_name}."
                )


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()