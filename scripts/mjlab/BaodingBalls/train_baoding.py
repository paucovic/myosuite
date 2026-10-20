#!/usr/bin/env python3
"""Train the survival-only MJLab Baoding hold task with RSL-RL PPO.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/train_baoding.py

Run from the repository root:

    python scripts/mjlab/BaodingBalls/train_baoding.py \
        --policy-path my_baoding_hold_policy.pt

This saves:

    my_baoding_hold_policy.pt
    my_baoding_hold_policy_metrics.json
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
BALL_SIZE_SCALE = 0.86

# Both balls receive the same random horizontal translation, in metres.
BALL_PAIR_RESET_OFFSET_LOW = (
    -0.003,
    -0.003,
    0.0,
)
BALL_PAIR_RESET_OFFSET_HIGH = (
    0.003,
    0.003,
    0.0,
)

# Each ball then receives a smaller independent horizontal displacement.
BALL_INDIVIDUAL_RESET_JITTER_LOW = (
    -0.001,
    -0.001,
    0.0,
)
BALL_INDIVIDUAL_RESET_JITTER_HIGH = (
    0.001,
    0.001,
    0.0,
)

# Apply these values before MJLab creates or registers the environment.
# Keeping this at module level also lets visualize_baoding.py obtain exactly
# the same configuration simply by importing this training module first.
baoding_registration.configure_baoding_balls(
    ball_size_scale=BALL_SIZE_SCALE,
    pair_reset_offset_low=(
        BALL_PAIR_RESET_OFFSET_LOW
    ),
    pair_reset_offset_high=(
        BALL_PAIR_RESET_OFFSET_HIGH
    ),
    individual_reset_jitter_low=(
        BALL_INDIVIDUAL_RESET_JITTER_LOW
    ),
    individual_reset_jitter_high=(
        BALL_INDIVIDUAL_RESET_JITTER_HIGH
    ),
)


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# These PPO values match the latest five-finger train_handreach.py.
ITERATIONS = 2000
NUM_ENVS = 512
NUM_STEPS_PER_ENV = 64

NUM_EVALS = 100
EVAL_NUM_ENVS = 64

# One evaluation covers one complete Baoding episode per evaluation environment.
EVAL_STEPS = MAX_EPISODE_STEPS

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
# OUTPUT REDIRECTION
# =============================================================================

@contextlib.contextmanager
def redirect_output(
    enabled: bool,
):
    if not enabled:
        yield
        return

    with open(
        os.devnull,
        "w",
        encoding="utf-8",
    ) as file:
        with contextlib.redirect_stdout(file):
            with contextlib.redirect_stderr(file):
                yield


def import_mjlab_tools(
    quiet: bool,
):
    with redirect_output(quiet):
        import myosuite  # noqa: F401
        import mjlab.tasks.registry as registry

        from myosuite.core.registry import (
            make_env,
        )

        baoding_registration.register_baoding_hold_task(
            quiet=True
        )

    return make_env, registry.load_rl_cfg


# =============================================================================
# MJLAB TO RSL-RL WRAPPER
# =============================================================================

class BaodingRslRlWrapper:
    """Adapt the native MJLab Baoding environment to RSL-RL's interface."""

    def __init__(
        self,
        env: Any,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(
            getattr(
                env,
                "num_envs",
                1,
            )
        )
        self.num_actions = int(
            env.action_space.shape[-1]
        )
        self.cfg = getattr(
            env,
            "cfg",
            None,
        )

        if (
            self.num_actions
            != EXPECTED_ACTION_DIM
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_ACTION_DIM} actions, "
                f"but the environment exposes "
                f"{self.num_actions}."
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
            try:
                self.env.reset(
                    seed=seed
                )
            except TypeError:
                self.env.reset()
        else:
            self.env.reset()

        return self.get_observations()

    def get_observations(
        self,
    ) -> TensorDict:
        obs_dict = (
            self.env
            .observation_manager
            .compute()
        )

        obs = TensorDict(
            obs_dict,
            batch_size=[
                self.num_envs
            ],
        )

        self._validate_observation(
            obs["policy"]
        )

        return obs

    def step(
        self,
        actions: torch.Tensor,
    ):
        actions = actions.to(
            self.device
        )

        if actions.ndim == 1:
            actions = actions.unsqueeze(
                0
            )

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
                f"{expected_action_shape}, "
                f"got {tuple(actions.shape)}."
            )

        (
            obs_dict,
            native_reward,
            terminated,
            truncated,
            extras,
        ) = self.env.step(actions)

        rewards = native_reward.reshape(
            self.num_envs
        )
        terminated = terminated.reshape(
            self.num_envs
        ).bool()
        truncated = truncated.reshape(
            self.num_envs
        ).bool()

        done_mask = (
            terminated
            | truncated
        )
        dones = done_mask.to(
            dtype=torch.long
        )

        obs = TensorDict(
            obs_dict,
            batch_size=[
                self.num_envs
            ],
        )

        self._validate_observation(
            obs["policy"]
        )

        if not torch.isfinite(
            rewards
        ).all():
            raise RuntimeError(
                "The Baoding reward contains "
                "NaN or infinity."
            )

        # The registered task has only one non-time-limit termination: a drop.
        # A time-limit step still counts as a held step.
        both_held = (
            ~terminated
        ).float()

        ball_dropped = (
            terminated.float()
        )

        infos: dict[str, Any] = {
            "baoding": {
                "reward": (
                    rewards.detach()
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
            },
            "episode": {
                "reward": (
                    rewards.detach()
                    .mean()
                ),
                "both_held": (
                    both_held.detach()
                    .mean()
                ),
                "ball_dropped": (
                    ball_dropped.detach()
                    .mean()
                ),
            },
        }

        if isinstance(
            extras,
            dict,
        ):
            if "time_outs" in extras:
                infos["time_outs"] = (
                    extras["time_outs"]
                )
            else:
                infos["time_outs"] = (
                    truncated
                )

            if "log" in extras:
                infos["log"] = extras[
                    "log"
                ]
        else:
            infos["time_outs"] = (
                truncated
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
                f"shape {expected_shape}, "
                f"got "
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
        if isinstance(
            child,
            dict,
        ):
            child_found = (
                set_key_recursively(
                    child,
                    key,
                    value,
                )
            )
            found = (
                child_found
                or found
            )

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

    train_cfg["logger"] = (
        "tensorboard"
    )

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
        (
            "num_learning_epochs",
        ),
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
        (
            "desired_kl",
        ),
        DESIRED_KL,
    )

    set_training_value(
        train_cfg,
        (
            "max_grad_norm",
        ),
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

    algorithm_cfg["schedule"] = (
        SCHEDULE
    )


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
        ) == 0
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
            "This PyTorch version does "
            "not support optimizer "
            "step pre-hooks."
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

        algorithm.learning_rate = (
            bounded_lr
        )

        for parameter_group in (
            optimizer.param_groups
        ):
            parameter_group["lr"] = (
                bounded_lr
            )

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
        path = (
            REPO_ROOT
            / path
        )

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
    rows: list[
        dict[str, float]
    ],
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
        or isinstance(
            value,
            bool,
        )
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
        .replace(
            " ",
            "_",
        )
    )

    mapping = {
        "value": "train/value",
        "value_loss": "train/value",
        "surrogate": (
            "train/surrogate"
        ),
        "surrogate_loss": (
            "train/surrogate"
        ),
        "policy_loss": (
            "train/surrogate"
        ),
        "entropy": "train/entropy",
        "entropy_loss": (
            "train/entropy"
        ),
        "kl": "train/approx_kl",
        "kl_mean": (
            "train/approx_kl"
        ),
        "mean_kl": (
            "train/approx_kl"
        ),
        "approx_kl": (
            "train/approx_kl"
        ),
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
        ) > 0
    ):
        fixed_lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        def force_fixed_lr() -> None:
            algorithm.learning_rate = (
                fixed_lr
            )

            for parameter_group in (
                optimizer.param_groups
            ):
                parameter_group["lr"] = (
                    fixed_lr
                )

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
            kl_value = (
                original_kl_function(
                    *args,
                    **kwargs,
                )
            )

            scalar = to_float(
                kl_value
            )

            if scalar is not None:
                storage[
                    "kl_values"
                ].append(scalar)

            return kl_value

        kl_source.get_kl_divergence = (
            wrapped_kl_function
        )

    original_update = (
        algorithm.update
    )

    def wrapped_update(
        *args,
        **kwargs,
    ):
        kl_start = len(
            storage[
                "kl_values"
            ]
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
            algorithm.schedule = (
                "adaptive"
            )

        try:
            update_output = (
                original_update(
                    *args,
                    **kwargs,
                )
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

        record: dict[
            str,
            float,
        ] = {}

        if (
            optimizer is not None
            and len(
                optimizer.param_groups
            ) > 0
        ):
            record[
                "train/learning_rate"
            ] = float(
                optimizer
                .param_groups[0]["lr"]
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

        if len(record) > 0:
            storage[
                "updates"
            ].append(record)

        return update_output

    algorithm.update = (
        wrapped_update
    )

    return storage


def average_update_metrics(
    records: list[
        dict[str, float]
    ],
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
            sum(values)
            / len(values)
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

    if isinstance(
        action,
        dict,
    ):
        for key in (
            "action",
            "actions",
        ):
            if key in action:
                return action[key]

    if isinstance(
        action,
        tuple,
    ):
        return unpack_action(
            action[0]
        )

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
                runner
                .get_inference_policy(
                    device=device
                )
            )

            for policy_input in (
                obs,
                obs["policy"],
            ):
                try:
                    return unpack_action(
                        policy(
                            policy_input
                        )
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

                if hasattr(
                    actor,
                    "eval",
                ):
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
    eval_env: BaodingRslRlWrapper,
    seed: int,
    device: str,
    quiet: bool,
) -> dict[str, float]:
    """Evaluate exactly the first episode produced by each evaluation env."""
    with redirect_output(quiet):
        obs = eval_env.reset(
            seed=seed
        )

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

    active = torch.ones(
        EVAL_NUM_ENVS,
        device=eval_env.device,
        dtype=torch.bool,
    )

    dropped = torch.zeros_like(
        active
    )
    timed_out = torch.zeros_like(
        active
    )

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
        ) = eval_env.step(
            actions
        )

        baoding_info = infos[
            "baoding"
        ]

        terminated = baoding_info[
            "terminated"
        ].bool()

        truncated = baoding_info[
            "truncated"
        ].bool()

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

    completed = ~active

    full_hold = (
        held_steps >= EVAL_STEPS
    ) & (~dropped)

    episode_lengths_float = (
        episode_lengths.float()
    )

    held_steps_float = (
        held_steps.float()
    )

    return_per_executed_step = (
        episode_returns
        / episode_lengths_float
        .clamp_min(1.0)
    )

    metrics = {
        "eval/mean_reward": float(
            episode_returns
            .mean()
            .cpu()
        ),
        "eval/reward_per_step": float(
            return_per_executed_step
            .mean()
            .cpu()
        ),
        "eval/reward_std": float(
            episode_returns
            .std(
                unbiased=False
            )
            .cpu()
        ),
        "eval/mean_episode_length": float(
            episode_lengths_float
            .mean()
            .cpu()
        ),
        "eval/mean_hold_steps": float(
            held_steps_float
            .mean()
            .cpu()
        ),
        "eval/mean_hold_time_sec": float(
            (
                held_steps_float
                * CTRL_DT
            )
            .mean()
            .cpu()
        ),
        "eval/mean_hold_fraction": float(
            (
                held_steps_float
                / EVAL_STEPS
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

    return metrics


def make_metric_row(
    *,
    eval_index: int,
    iteration: int,
    env_steps: int,
    elapsed_sec: float,
    eval_metrics: dict[
        str,
        float,
    ],
    train_metrics: dict[
        str,
        float,
    ],
) -> dict[str, float]:
    row = {
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

    row.update(
        eval_metrics
    )
    row.update(
        train_metrics
    )

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
) -> BaodingRslRlWrapper:
    with redirect_output(quiet):
        env = make_env(
            env_id,
            backend="mjlab",
            device=device,
            num_envs=num_envs,
        )

    wrapper = (
        BaodingRslRlWrapper(
            env=env
        )
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
        exposed_length = (
            wrapper.max_episode_length
        )

        wrapper.close()

        raise RuntimeError(
            f"Expected an episode length "
            f"of {MAX_EPISODE_STEPS}, "
            f"but the environment exposes "
            f"{exposed_length}."
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
            "clip_param": (
                CLIP_PARAM
            ),
            "desired_kl": (
                DESIRED_KL
            ),
            "max_grad_norm": (
                MAX_GRAD_NORM
            ),
            "entropy_coef": (
                ENTROPY_COEF
            ),
        },
        "evaluation": {
            "num_evals": NUM_EVALS,
            "eval_num_envs": (
                EVAL_NUM_ENVS
            ),
            "eval_steps": (
                EVAL_STEPS
            ),
            "fixed_seed": (
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
            "drop_height": (
                DROP_HEIGHT
            ),
        },
        "ball_configuration": {
            "size_scale": (
                BALL_SIZE_SCALE
            ),
            "mass_scale": (
                BALL_SIZE_SCALE ** 3
            ),
            "pair_reset_offset_low": (
                list(
                    BALL_PAIR_RESET_OFFSET_LOW
                )
            ),
            "pair_reset_offset_high": (
                list(
                    BALL_PAIR_RESET_OFFSET_HIGH
                )
            ),
            "individual_reset_jitter_low": (
                list(
                    BALL_INDIVIDUAL_RESET_JITTER_LOW
                )
            ),
            "individual_reset_jitter_high": (
                list(
                    BALL_INDIVIDUAL_RESET_JITTER_HIGH
                )
            ),
            "vertical_randomization": False,
        },
        "reward": {
            "hold_reward_per_step": (
                HOLD_REWARD_PER_STEP
            ),
            "drop_penalty": (
                DROP_PENALTY
            ),
            "maximum_return": (
                MAX_EPISODE_STEPS
                * HOLD_REWARD_PER_STEP
            ),
        },
        "observation_layout": {
            "hand_qpos": [
                0,
                23,
            ],
            "hand_qvel": [
                23,
                46,
            ],
            "ball1_pos": [
                46,
                49,
            ],
            "ball1_lin_vel": [
                49,
                52,
            ],
            "ball2_pos": [
                52,
                55,
            ],
            "ball2_lin_vel": [
                55,
                58,
            ],
            "activation": [
                58,
                97,
            ],
            "total": OBS_DIM,
        },
        "action_dimension": (
            EXPECTED_ACTION_DIM
        ),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    args: argparse.Namespace,
) -> None:
    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

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

    from rsl_rl.runners import (
        OnPolicyRunner,
    )

    train_env: (
        BaodingRslRlWrapper
        | None
    ) = None

    eval_env: (
        BaodingRslRlWrapper
        | None
    ) = None

    try:
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
            seed=(
                args.seed
                + 10_000
            ),
            quiet=args.quiet,
        )

        train_cfg = load_rl_cfg(
            args.env_id
        )

        train_cfg.obs_groups = {
            "policy": (
                "policy",
            ),
            "critic": (
                "policy",
            ),
            "actor": (
                "policy",
            ),
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

        with redirect_output(
            args.quiet
        ):
            runner = OnPolicyRunner(
                env=train_env,
                train_cfg=(
                    train_cfg_dict
                ),
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
            ) == 0
        ):
            raise RuntimeError(
                "RSL-RL PPO optimizer "
                "was not found."
            )

        effective_lr = float(
            optimizer
            .param_groups[0]["lr"]
        )

        if (
            effective_schedule
            != SCHEDULE
        ):
            raise RuntimeError(
                f"Expected PPO schedule "
                f"{SCHEDULE!r}, but "
                f"RSL-RL is using "
                f"{effective_schedule!r}."
            )

        learning_rate_tolerance = max(
            1e-12,
            abs(
                LEARNING_RATE
            ) * 1e-6,
        )

        if (
            abs(
                effective_lr
                - LEARNING_RATE
            )
            > learning_rate_tolerance
        ):
            raise RuntimeError(
                f"Expected optimizer "
                f"learning rate "
                f"{LEARNING_RATE}, but "
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

        eval_every = (
            ITERATIONS
            // NUM_EVALS
        )

        saved_config = (
            build_saved_config(
                args
            )
        )

        metric_rows: list[
            dict[str, float]
        ] = []

        print("=" * 80)
        print(
            "MJLAB BAODING HOLD TRAINING"
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
            "adaptive LR range:     "
            f"[{MIN_ADAPTIVE_LR}, "
            f"{MAX_ADAPTIVE_LR}]"
        )
        print(
            f"configured LR:         "
            f"{LEARNING_RATE}"
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
            f"maximum return:        "
            f"{MAX_EPISODE_STEPS * HOLD_REWARD_PER_STEP}"
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
                    args.seed
                    + 20_000
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

        metric_rows.append(
            initial_row
        )

        save_metrics(
            path=metrics_path,
            config=saved_config,
            rows=metric_rows,
        )

        print(
            f"[eval {eval_index:03d}/"
            f"{NUM_EVALS:03d}] "
            "iter=0 "
            "steps=0 "
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
            f"{initial_evaluation['eval/drop_fraction']:.3f}"
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
                    update_storage[
                        "updates"
                    ]
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

            evaluation = (
                evaluate_policy(
                    runner=runner,
                    eval_env=eval_env,
                    seed=(
                        args.seed
                        + 20_000
                    ),
                    device=args.device,
                    quiet=args.quiet,
                )
            )

            elapsed_seconds = (
                time.time()
                - start_time
            )

            row = make_metric_row(
                eval_index=eval_index,
                iteration=(
                    completed_iterations
                ),
                env_steps=(
                    env_steps
                ),
                elapsed_sec=(
                    elapsed_seconds
                ),
                eval_metrics=(
                    evaluation
                ),
                train_metrics=(
                    train_metrics
                ),
            )

            metric_rows.append(row)

            # This deliberately stores the current policy, not only the best.
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
                        optimizer
                        .param_groups[0][
                            "lr"
                        ]
                    ),
                )
            )

            approx_kl = (
                train_metrics.get(
                    "train/approx_kl"
                )
            )

            if approx_kl is None:
                kl_text = "n/a"
            else:
                kl_text = (
                    f"{approx_kl:.3e}"
                )

            print(
                f"[eval {eval_index:03d}/"
                f"{NUM_EVALS:03d}] "
                f"iter="
                f"{completed_iterations} "
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
                f"lr="
                f"{current_learning_rate:.3e} "
                f"kl={kl_text} "
                f"time="
                f"{elapsed_seconds:.1f}s"
            )

        runner.save(
            str(policy_path)
        )

        save_metrics(
            path=metrics_path,
            config=saved_config,
            rows=metric_rows,
        )

        print("=" * 80)
        print(
            "TRAINING COMPLETE"
        )
        print("=" * 80)
        print(
            f"policy:  {policy_path}"
        )
        print(
            f"metrics: {metrics_path}"
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
            "my_baoding_hold_policy.pt"
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
            "is not available."
        )

    if not (
        0.0
        < BALL_SIZE_SCALE
        <= 1.0
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
            (
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            (
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
        ),
    )

    for (
        name,
        low,
        high,
    ) in reset_ranges:
        if (
            len(low) != 3
            or len(high) != 3
        ):
            raise ValueError(
                f"{name} bounds must "
                "contain exactly three "
                "values."
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
                f"Every lower {name} "
                "bound must be less than "
                "or equal to its upper "
                "bound."
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
        "EVAL_NUM_ENVS": (
            EVAL_NUM_ENVS
        ),
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

    if (
        ITERATIONS
        % NUM_EVALS
        != 0
    ):
        raise ValueError(
            "ITERATIONS must be divisible "
            "by NUM_EVALS so the requested "
            "number of evenly spaced "
            "evaluations is exact."
        )

    if (
        EVAL_STEPS
        != MAX_EPISODE_STEPS
    ):
        raise ValueError(
            "EVAL_STEPS must equal "
            "MAX_EPISODE_STEPS so "
            "full-hold success is measured "
            "over one complete episode."
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
            "LEARNING_RATE must be "
            "positive."
        )


def main() -> None:
    args = parse_args()
    validate_config(args)
    train(args)


if __name__ == "__main__":
    main()