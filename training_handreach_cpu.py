#!/usr/bin/env python3
"""Train PPO on the classic, non-MJX MyoSuite HandReach environment.

Outputs:
    <name>_all_metrics.jsonl
    <name>_BEST.zip
    <name>_BEST_vecnormalize.pkl
    <name>_FINAL.zip
    <name>_FINAL_vecnormalize.pkl

This does not create TensorBoard logs, CSV files, or evaluations.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import myosuite  # Registers the MyoSuite environments.
from myosuite.utils import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    JSONOutputFormat,
    Logger,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# POLICY SAVING
# ============================================================

class SaveBestPolicyCallback(BaseCallback):
    """Save the best policy and its observation-normalization statistics."""

    def __init__(
        self,
        policy_path: Path,
        vecnormalize_path: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        # Do not include ".zip" here.
        # Stable-Baselines3 adds it automatically.
        self.policy_path = policy_path
        self.vecnormalize_path = vecnormalize_path

    def _on_step(self) -> bool:
        self.policy_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save(str(self.policy_path))

        vec_normalize = self.model.get_vec_normalize_env()

        if vec_normalize is not None:
            vec_normalize.save(str(self.vecnormalize_path))

        if self.verbose:
            print()
            print("Saved new best classic policy:")
            print(f"  policy: {self.policy_path}.zip")
            print(
                "  normalization: "
                f"{self.vecnormalize_path}"
            )

        return True


# ============================================================
# EVENLY SPACED EVALUATIONS
# ============================================================

class EvenlySpacedEvalCallback(EvalCallback):
    """Run a fixed number of evaluations distributed over training."""

    def __init__(
        self,
        *,
        total_timesteps: int,
        num_evals: int,
        **kwargs,
    ):
        if total_timesteps <= 0:
            raise ValueError(
                "total_timesteps must be greater than zero."
            )

        if num_evals <= 0:
            raise ValueError(
                "num_evals must be greater than zero."
            )

        if num_evals > total_timesteps:
            raise ValueError(
                "num_evals cannot be greater than total_timesteps."
            )

        # The parent checks eval_freq whenever _on_step() is called.
        # We call it manually only at our requested evaluation points.
        super().__init__(
            eval_freq=1,
            **kwargs,
        )

        self.total_timesteps_target = int(total_timesteps)
        self.num_evals_target = int(num_evals)

        # Example for 5,000,000 steps and 32 evaluations:
        #
        # 156250, 312500, ..., 5000000
        self.eval_targets = np.ceil(
            np.arange(1, num_evals + 1)
            * total_timesteps
            / num_evals
        ).astype(np.int64)

        self.next_eval_index = 0

    def _on_training_start(self) -> None:
        super()._on_training_start()

        print()
        print("Evaluation schedule")
        print("=" * 70)
        print("Number of evaluations:", self.num_evals_target)
        print(
            "Approximate spacing:",
            self.total_timesteps_target
            / self.num_evals_target,
        )
        print(
            "First evaluation:",
            int(self.eval_targets[0]),
        )
        print(
            "Final evaluation:",
            int(self.eval_targets[-1]),
        )
        print("=" * 70)
        print()

    def _on_step(self) -> bool:
        if self.next_eval_index >= len(self.eval_targets):
            return True

        target_step = int(
            self.eval_targets[self.next_eval_index]
        )

        if self.num_timesteps < target_step:
            return True

        evaluation_number = self.next_eval_index + 1

        print()
        print(
            f"Starting evaluation "
            f"{evaluation_number}/{self.num_evals_target}"
        )
        print(
            f"Requested step: {target_step}; "
            f"actual step: {self.num_timesteps}"
        )

        # EvalCallback will:
        #   1. synchronize VecNormalize statistics
        #   2. run deterministic evaluation episodes
        #   3. log eval/mean_reward and eval/mean_ep_length
        #   4. trigger SaveBestPolicyCallback when appropriate
        continue_training = super()._on_step()

        self.next_eval_index += 1

        return continue_training


# ============================================================
# ENVIRONMENT CREATION
# ============================================================

def make_monitored_env(env_id: str):
    """Create one classic MyoSuite environment."""

    def _make():
        env = gym.make(env_id)
        return Monitor(env)

    return _make


def build_vec_env(
    env_id: str,
    num_envs: int,
    seed: int,
    training: bool,
) -> VecNormalize:
    """Create classic CPU environments with observation normalization."""

    vec_env = DummyVecEnv(
        [
            make_monitored_env(env_id)
            for _ in range(num_envs)
        ]
    )

    vec_env.seed(seed)

    vec_env = VecNormalize(
        vec_env,
        training=training,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    return vec_env


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Stable-Baselines3 PPO on classic "
            "MyoSuite HandReach."
        )
    )

    parser.add_argument(
        "--env",
        type=str,
        default="myoHandReachRandom-v0",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5_000_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help=(
            "Number of transitions collected per environment "
            "before each PPO update."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--num-evals",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="ClassicHandReach_PPO",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    return parser.parse_args()


# ============================================================
# TRAINING
# ============================================================

def main() -> None:
    args = parse_args()

    rollout_buffer_size = args.n_steps * args.num_envs

    if rollout_buffer_size < args.batch_size:
        raise ValueError(
            "n_steps * num_envs must be at least batch_size. "
            f"Current values: {args.n_steps} * {args.num_envs} "
            f"= {rollout_buffer_size}; "
            f"batch_size={args.batch_size}."
        )

    if rollout_buffer_size % args.batch_size != 0:
        print(
            "Warning: n_steps * num_envs is not divisible by "
            "batch_size. The final minibatch will be smaller."
        )

    evaluation_spacing = args.steps / args.num_evals

    if evaluation_spacing < args.num_envs:
        raise ValueError(
            "The evaluation spacing is smaller than one "
            "vectorized environment step. Reduce --num-evals "
            "or --num-envs."
        )

    output_dir = (
        Path("classic_handreach_runs")
        / args.name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        output_dir
        / f"{args.name}_all_metrics.jsonl"
    )

    best_policy_path = (
        output_dir
        / f"{args.name}_BEST"
    )

    best_vecnormalize_path = (
        output_dir
        / f"{args.name}_BEST_vecnormalize.pkl"
    )

    final_policy_path = (
        output_dir
        / f"{args.name}_FINAL"
    )

    final_vecnormalize_path = (
        output_dir
        / f"{args.name}_FINAL_vecnormalize.pkl"
    )

    # Remove an old metrics file when reusing the same run name.
    if metrics_path.exists():
        metrics_path.unlink()

    train_env = build_vec_env(
        env_id=args.env,
        num_envs=args.num_envs,
        seed=args.seed,
        training=True,
    )

    eval_env = build_vec_env(
        env_id=args.env,
        num_envs=1,
        seed=args.seed + 10_000,
        training=False,
    )

    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": {
            "pi": [64, 64, 64],
            "vf": [64, 64, 64],
        },
    }

    print()
    print("Classic HandReach PPO")
    print("=" * 70)
    print("Environment:", args.env)
    print("Total steps:", args.steps)
    print("Parallel environments:", args.num_envs)
    print("Rollout length:", args.n_steps)
    print("Rollout buffer size:", rollout_buffer_size)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)
    print("Number of evaluations:", args.num_evals)
    print("Episodes per evaluation:", args.eval_episodes)
    print("Metrics file:", metrics_path)
    print("=" * 70)
    print()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,

        learning_rate=args.learning_rate,

        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=8,

        gamma=0.97,
        gae_lambda=0.95,

        clip_range=0.3,
        ent_coef=0.001,
        vf_coef=0.5,

        max_grad_norm=1.0,
        normalize_advantage=True,

        policy_kwargs=policy_kwargs,

        verbose=0,
        seed=args.seed,
        device=args.device,
    )

    # Write both:
    #   - readable terminal tables
    #   - newline-delimited JSON metrics
    #
    # The JSON file contains normal SB3 names such as:
    #   time/total_timesteps
    #   rollout/ep_rew_mean
    #   eval/mean_reward
    #   train/policy_gradient_loss
    #   train/value_loss
    metrics_logger = Logger(
        folder=str(output_dir),
        output_formats=[
            HumanOutputFormat(sys.stdout),
            JSONOutputFormat(str(metrics_path)),
        ],
    )

    model.set_logger(metrics_logger)

    save_best_callback = SaveBestPolicyCallback(
        policy_path=best_policy_path,
        vecnormalize_path=best_vecnormalize_path,
        verbose=1,
    )

    eval_callback = EvenlySpacedEvalCallback(
        total_timesteps=args.steps,
        num_evals=args.num_evals,

        eval_env=eval_env,

        callback_on_new_best=save_best_callback,

        # No best_model_save_path:
        # SaveBestPolicyCallback handles it with our filenames.
        best_model_save_path=None,

        # No log_path:
        # this prevents evaluations.npz from being created.
        log_path=None,

        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=eval_callback,

            # Log standard PPO metrics after every rollout/update.
            log_interval=1,

            reset_num_timesteps=True,
            progress_bar=False,
        )

        model.save(str(final_policy_path))
        train_env.save(str(final_vecnormalize_path))

        print()
        print("Training finished.")
        print("=" * 70)
        print("Metrics:")
        print(f"  {metrics_path}")
        print()
        print("Best policy:")
        print(f"  {best_policy_path}.zip")
        print(f"  {best_vecnormalize_path}")
        print()
        print("Final policy:")
        print(f"  {final_policy_path}.zip")
        print(f"  {final_vecnormalize_path}")
        print("=" * 70)

    finally:
        train_env.close()
        eval_env.close()
        metrics_logger.close()


if __name__ == "__main__":
    main()