"""Train a PPO agent using brax."""

import functools
import time
import os
import jax
import csv

print(f"Current backend: {jax.default_backend()}. "
      f"If you expect gpu but see cpu, you may need to reinstall jax with a suitable version of cuda.")
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from myosuite.envs.myo.mjx import ppo_config

from myosuite.envs.myo.mjx import make, get_default_config
from mujoco_playground import wrapper
import pickle
import wandb
import argparse
from brax.io import model

def main(env_name, impl, log_to_wandb, save_policy, policy_name):
    """Run training and evaluation for the specified environment."""

    env, ppo_params, network_factory = load_env_and_network_factory(env_name, impl)

    global log_csv_path
    log_csv_path = policy_name.replace(".pickle", "_learning_curve.csv")

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_steps",
            "wall_time",
            "eval_episode_reward",
        ])

    print(f"Saving learning curve to: {log_csv_path}")

    if log_to_wandb:
        wandb_run = wandb.init(project=env_name, config=ppo_params)

    # Train the model
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_envs=env._config.num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=functools.partial(
            progress,
            log_to_wandb=log_to_wandb,
            policy_name=policy_name,
        ),
        policy_params_fn=functools.partial(
            save_policy_checkpoint,
            policy_name=policy_name,
        ),
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.pop("num_eval_envs"),
        **ppo_params
    )

    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

    if save_policy:
        model.save_params(policy_name, params)
        print(f"Saved policy to: {policy_name}")

def load_env_and_network_factory(env_name, impl, config_overrides=None):
    env_overrides = {"impl": impl}

    if config_overrides is not None:
        env_overrides.update(config_overrides)

    env = make(env_name, config_overrides=env_overrides)
    config = env._config
    ppo_params = dict(ppo_config)

    print(f"Training on environment:\n{env_name}")
    print(f"Using backend:\n{impl}")
    print(f"Environment Config:\n{config}")
    print(f"PPO Training Parameters:\n{ppo_config}")

    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.pop("network_factory")
        )
    else:
        network_factory = ppo_networks.make_ppo_networks

    return env, ppo_params, network_factory

times = [time.monotonic()]
total_steps = [0]
log_csv_path = None

eval_rewards_by_step = {}
params_by_step = {}
best_eval_reward = -float("inf")


def _policy_path(policy_name, suffix):
    base, ext = os.path.splitext(policy_name)
    if ext == "":
        ext = ".pickle"
    return f"{base}_{suffix}{ext}"


def save_policy_checkpoint(num_steps, make_policy, params, policy_name):
    """Called by Brax PPO when policy params are available."""
    params_by_step[num_steps] = params
    maybe_save_best_policy(num_steps, policy_name)


def progress(num_steps, metrics, log_to_wandb, policy_name):
    """Log eval reward and save BEST policy if this eval improved."""
    times.append(time.monotonic())
    total_steps.append(num_steps)

    eval_reward = float(metrics["eval/episode_reward"])
    eval_rewards_by_step[num_steps] = eval_reward

    maybe_save_best_policy(num_steps, policy_name)

    print(f"Step {num_steps} at {times[-1]}: reward={eval_reward:.3f}")

    if log_csv_path is not None:
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                num_steps,
                times[-1] - times[0],
                eval_reward,
            ])

    if log_to_wandb:
        wandb.log(metrics, step=num_steps)


def maybe_save_best_policy(num_steps, policy_name):
    """Overwrite only <policy_name>_BEST.pickle when eval reward improves."""
    global best_eval_reward

    if num_steps not in params_by_step or num_steps not in eval_rewards_by_step:
        return

    params = params_by_step.pop(num_steps)
    eval_reward = eval_rewards_by_step.pop(num_steps)

    if eval_reward > best_eval_reward:
        best_eval_reward = eval_reward
        best_path = _policy_path(policy_name, "BEST")

        model.save_params(best_path, params)

        print(
            f"Saved new best policy to: {best_path} "
            f"(step={num_steps}, reward={eval_reward:.3f})"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PPO agent with Brax")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxFingerPoseRandom-v0",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="jax",
        help='Implementation to use: "jax" (MJX) or "warp" (MJWarp)',
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--save_policy",
        action="store_true",
    )
    parser.add_argument(
        "--policy_name",
        type=str,
        default="playground_params.pickle",
        help="Filename where the trained policy parameters will be saved.",
    )

    args = parser.parse_args()

    main(args.env_name, args.impl, args.log_to_wandb, args.save_policy, args.policy_name)
