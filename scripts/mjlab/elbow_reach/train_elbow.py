#!/usr/bin/env python3
"""
Train an elbow pose policy with mjlab + RSL-RL.

Run from the repo root:

    python scripts/mjlab/train_elbow.py --policy-path my_elbow_policy.pt

This saves:
    <policy_name>.pt
    <policy_name>_metrics.json

The metrics JSON is used by:

    python scripts/mjlab/plot_metrics.py --metrics-path my_elbow_policy_metrics.json
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


REPO_ROOT = Path(__file__).resolve().parents[3]


# =============================================================================
# USER CONFIG
# =============================================================================

ITERATIONS = 500
NUM_ENVS = 64
NUM_STEPS_PER_ENV = 64

NUM_EVALS = 20
EVAL_NUM_ENVS = 16
EVAL_STEPS = 200

LEARNING_RATE = 3e-4
NUM_LEARNING_EPOCHS = 8
NUM_MINI_BATCHES = 4

CLIP_PARAM = 0.2
DESIRED_KL = 0.01
MAX_GRAD_NORM = 1.0

# Set to None to keep the default value from _elbow_ppo_runner_cfg().
ENTROPY_COEF = None

RANDOMIZE_TARGET = True
TARGET_LOW = 0.35
TARGET_HIGH = 1.25

POSE_SCALE = 2.0
VELOCITY_PENALTY_SCALE = 0.05
ACTION_PENALTY_SCALE = 0.01

CLOSE_POSE_TH = 0.10
VERY_CLOSE_POSE_TH = 0.05

CLOSE_BONUS_VALUE = 2.0
VERY_CLOSE_BONUS_VALUE = 4.0


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

        from myosuite.core.registry import make_env
        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            bootstrap_myosuite_mjlab_registry,
            _elbow_ppo_runner_cfg,
        )

        bootstrap_myosuite_mjlab_registry()

    return make_env, _elbow_ppo_runner_cfg


# =============================================================================
# ENV WRAPPER
# =============================================================================

class ElbowRslRlWrapper:
    """
    Minimal adapter between mjlab and RSL-RL.

    Assumed obs["policy"] layout:
        0: qpos
        1: qvel
        2: pose_err
        3 onward: muscle activations
    """

    def __init__(self, env: Any, seed: int) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(getattr(env, "num_envs", 1))
        self.num_actions = int(env.action_space.shape[-1])
        self.cfg = getattr(env, "cfg", None)

        self.target_rng = torch.Generator(device="cpu")
        self.target_rng.manual_seed(seed)

        self.target_angles = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

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
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        self._resample_targets(torch.ones(self.num_envs, dtype=torch.bool))
        return self.get_observations()

    def get_observations(self) -> TensorDict:
        obs_dict = self.env.observation_manager.compute()
        obs_dict = self._replace_pose_error(obs_dict)
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        obs_dict, _, terminated, truncated, extras = self.env.step(actions)
        obs_dict = self._replace_pose_error(obs_dict)

        rewards, reward_info = self._reward(obs_dict, actions)

        done_mask = terminated | truncated
        dones = done_mask.to(dtype=torch.long)

        self._resample_targets(done_mask)

        obs_dict = self._replace_pose_error(obs_dict)
        obs = TensorDict(obs_dict, batch_size=[self.num_envs])

        infos = {
            "episode": {
                "reward": reward_info["reward"].mean(),
                "pose_error_abs": reward_info["pose_error_abs"].mean(),
                "close_solved": reward_info["close_solved"].mean(),
                "very_close_solved": reward_info["very_close_solved"].mean(),
            }
        }

        if isinstance(extras, dict) and "time_outs" in extras:
            infos["time_outs"] = extras["time_outs"]

        return obs, rewards, dones, infos

    def _sample_targets(self, count: int) -> torch.Tensor:
        samples = torch.rand(
            count,
            generator=self.target_rng,
            device="cpu",
            dtype=torch.float32,
        )

        targets = TARGET_LOW + (TARGET_HIGH - TARGET_LOW) * samples
        return targets.to(self.device)

    def _resample_targets(self, mask: torch.Tensor) -> None:
        if not RANDOMIZE_TARGET:
            return

        mask = mask.to(device=self.device, dtype=torch.bool)

        if not bool(mask.any()):
            return

        count = int(mask.sum().detach().cpu())
        self.target_angles[mask] = self._sample_targets(count)

    def _replace_pose_error(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not RANDOMIZE_TARGET:
            return obs_dict

        obs_dict = dict(obs_dict)
        policy_obs = obs_dict["policy"].clone()

        qpos = policy_obs[:, 0]
        policy_obs[:, 2] = self.target_angles - qpos

        obs_dict["policy"] = policy_obs
        return obs_dict

    def _reward(
        self,
        obs_dict: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_obs = obs_dict["policy"]

        qvel = policy_obs[:, 1]
        pose_error_abs = torch.abs(policy_obs[:, 2])

        pose_term = -POSE_SCALE * pose_error_abs
        velocity_penalty = VELOCITY_PENALTY_SCALE * torch.abs(qvel)
        action_penalty = ACTION_PENALTY_SCALE * torch.mean(actions.square(), dim=-1)

        close_solved = pose_error_abs < CLOSE_POSE_TH
        very_close_solved = pose_error_abs < VERY_CLOSE_POSE_TH

        close_bonus = CLOSE_BONUS_VALUE * close_solved.float()
        very_close_bonus = (
            VERY_CLOSE_BONUS_VALUE - CLOSE_BONUS_VALUE
        ) * very_close_solved.float()

        rewards = pose_term - velocity_penalty - action_penalty
        rewards = rewards + close_bonus + very_close_bonus

        info = {
            "reward": rewards.detach(),
            "pose_error_abs": pose_error_abs.detach(),
            "close_solved": close_solved.float().detach(),
            "very_close_solved": very_close_solved.float().detach(),
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
# EVALUATION
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
    eval_env: ElbowRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    with redirect_output(quiet):
        obs = eval_env.reset(seed=seed)

    rewards = []
    pose_errors = []
    close_solved = []
    very_close_solved = []

    for _ in range(EVAL_STEPS):
        actions = get_policy_action(runner, obs, device)
        obs, reward, _, infos = eval_env.step(actions)

        policy_obs = obs["policy"]
        pose_error_abs = torch.abs(policy_obs[:, 2])

        rewards.append(reward.detach())
        pose_errors.append(pose_error_abs.detach())

        episode_info = infos["episode"]
        close_solved.append(episode_info["close_solved"].detach())
        very_close_solved.append(episode_info["very_close_solved"].detach())

    reward_tensor = torch.stack(rewards)
    pose_error_tensor = torch.stack(pose_errors)

    return {
        "eval/mean_reward": float(reward_tensor.sum(dim=0).mean().cpu()),
        "eval/reward_per_step": float(reward_tensor.mean().cpu()),
        "eval/pose_error_abs": float(pose_error_tensor.mean().cpu()),
        "eval/pose_error_abs_last": float(pose_error_tensor[-1].mean().cpu()),
        "eval/close_solved_fraction": float(torch.stack(close_solved).mean().cpu()),
        "eval/very_close_solved_fraction": float(torch.stack(very_close_solved).mean().cpu()),
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
) -> ElbowRslRlWrapper:
    with redirect_output(quiet):
        env = make_env(
            env_id,
            backend="mjlab",
            device=device,
            num_envs=num_envs,
        )

    wrapper = ElbowRslRlWrapper(env=env, seed=seed)

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
        },
        "target": {
            "randomize_target": RANDOMIZE_TARGET,
            "target_low": TARGET_LOW,
            "target_high": TARGET_HIGH,
        },
        "reward": {
            "pose_scale": POSE_SCALE,
            "velocity_penalty_scale": VELOCITY_PENALTY_SCALE,
            "action_penalty_scale": ACTION_PENALTY_SCALE,
            "close_pose_th": CLOSE_POSE_TH,
            "very_close_pose_th": VERY_CLOSE_POSE_TH,
            "close_bonus_value": CLOSE_BONUS_VALUE,
            "very_close_bonus_value": VERY_CLOSE_BONUS_VALUE,
        },
    }


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    policy_path = resolve_path(args.policy_path)

    if args.metrics_path is None:
        metrics_path = default_metrics_path(policy_path)
    else:
        metrics_path = resolve_path(args.metrics_path)

    make_env, elbow_ppo_runner_cfg = import_mjlab_tools(quiet=args.quiet)

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

    train_cfg = elbow_ppo_runner_cfg()
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

    update_storage = patch_runner_for_metrics(runner)

    steps_per_iteration = NUM_ENVS * NUM_STEPS_PER_ENV
    eval_every = max(1, ITERATIONS // NUM_EVALS)

    saved_config = build_saved_config(args)
    metric_rows: list[dict[str, float]] = []

    print("=" * 80)
    print("MJLAB ELBOW TRAINING")
    print("=" * 80)
    print(f"env:                  {args.env_id}")
    print(f"device:               {args.device}")
    print(f"policy path:          {policy_path}")
    print(f"metrics path:         {metrics_path}")
    print(f"iterations:           {ITERATIONS}")
    print(f"num_envs:             {NUM_ENVS}")
    print(f"num_steps_per_env:    {NUM_STEPS_PER_ENV}")
    print(f"expected steps:       {ITERATIONS * steps_per_iteration}")
    print(f"num_evals:            {NUM_EVALS}")
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
        train_metrics={"train/num_updates": 0.0},
    )

    metric_rows.append(initial_row)
    save_metrics(metrics_path, saved_config, metric_rows)

    print(
        f"[eval {eval_index:02d}/{NUM_EVALS:02d}] "
        f"iter=0 "
        f"steps=0 "
        f"return={initial_eval['eval/mean_reward']:.3f} "
        f"reward/step={initial_eval['eval/reward_per_step']:.4f} "
        f"pose_err={initial_eval['eval/pose_error_abs']:.4f} "
        f"solved={initial_eval['eval/close_solved_fraction']:.3f} "
        f"very_solved={initial_eval['eval/very_close_solved_fraction']:.3f}"
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
            seed=args.seed + 20_000 + eval_index,
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
            f"reward/step={eval_metrics['eval/reward_per_step']:.4f} "
            f"pose_err={eval_metrics['eval/pose_error_abs']:.4f} "
            f"solved={eval_metrics['eval/close_solved_fraction']:.3f} "
            f"very_solved={eval_metrics['eval/very_close_solved_fraction']:.3f} "
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
        default="myoElbowPose1D6MFixed-v0",
    )

    parser.add_argument(
        "--policy-path",
        type=str,
        default="elbow_random_policy.pt",
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

    if TARGET_HIGH <= TARGET_LOW:
        raise ValueError("TARGET_HIGH must be greater than TARGET_LOW.")


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()