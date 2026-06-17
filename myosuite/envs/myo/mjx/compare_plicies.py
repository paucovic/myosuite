import functools
import jax
import jax.numpy as jp

import pickle

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

from myosuite.envs.myo.mjx import make, ppo_config

def run_policy(params_path, max_steps=400,seed=0):
    env = make(
        "MjxBaodingDebug-v0",
        config_overrides={
            "impl": "warp",
            "num_envs": 64,
        },
    )
    #print("Compare env config:")
    #print("  impl:", env._config.impl)
    #print("  num_envs:", env._config.num_envs)
    #print("  hold_dist_th:", env._config.hold_dist_th)
    #print("  goal_time_period:", env._config.goal_time_period)
    #print("  max_episode_steps:", env._config.max_episode_steps)
    
    ppo_params = dict(ppo_config)

    network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.pop("network_factory"),
    )

    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
    )

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    @jax.jit
    def policy_fn(obs, rng):
        action, _ = inference_fn(obs, rng)
        return action

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    total_reward = 0.0
    solved_steps = 0
    drop_step = max_steps

    for i in range(max_steps):
        rng, policy_rng = jax.random.split(rng)
        action = policy_fn(state.obs, policy_rng)
        state = jit_step(state, action)

        total_reward += float(state.reward)
        solved_steps += float(state.metrics["solved_bonus"] > 0.0)

        if float(state.done) == 1.0:
            drop_step = i + 1
            break

    print(f"Policy: {params_path}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Survived steps: {drop_step}/{max_steps}")
    print(f"Survived seconds: {drop_step * env._config.ctrl_dt:.3f}")
    print(f"Solved steps: {solved_steps}")
    print(f"Final pos_dist_1: {float(state.metrics['pos_dist_1']):.4f}")
    print(f"Final pos_dist_2: {float(state.metrics['pos_dist_2']):.4f}")

    #print(f"is_fall_1: {float(state.metrics['is_fall_1'])}")
    #print(f"is_fall_2: {float(state.metrics['is_fall_2'])}")
    #print(f"too_far_1: {float(state.metrics['too_far_1'])}")
    #print(f"too_far_2: {float(state.metrics['too_far_2'])}")

    print()

def run_policy_with_resets(params_path, max_steps=400):
    env = make(
        "MjxBaodingDebug-v0",
        config_overrides={
            "impl": "warp",
            "num_envs": 1,
        },
    )

    ppo_params = dict(ppo_config)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.pop("network_factory"),
    )

    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
    )

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    @jax.jit
    def policy_fn(obs, rng):
        action, _ = inference_fn(obs, rng)
        return action

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)

    total_reward = 0.0
    num_failures = 0
    first_failure_step = None

    for i in range(max_steps):
        rng, policy_rng = jax.random.split(rng)
        action = policy_fn(state.obs, policy_rng)
        state = jit_step(state, action)

        total_reward += float(state.reward)

        if float(state.done) == 1.0:
            num_failures += 1

            if first_failure_step is None:
                first_failure_step = i + 1

            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)

    if first_failure_step is None:
        first_failure_step = max_steps

    print(f"Policy with resets: {params_path}")
    print(f"Total reward over {max_steps} steps: {total_reward:.3f}")
    print(f"First failure step: {first_failure_step}/{max_steps}")
    print(f"Number of failures: {num_failures}")
    print()


if __name__ == "__main__":
    #Change here to compare different policies. Make sure to update the paths to the policy files as needed.

    #run_policy("BEST_BaodingHold_alive08_cons.pickle")
    #run_policy_with_resets("BEST_BaodingHold_alive08_cons.pickle")
    
    #run_policy("policies/BaodingHold_64N_LR2_run2_BEST.pickle")
    #run_policy_with_resets("BaodingHold_64N_run2_BEST.pickle")

    #run_policy("BaodingHold_64N_20M_LR2_run2.pickle")
    #run_policy_with_resets("BaodingHold_64N_run1_BEST.pickle")

    run_policy("BaodingHold_64N_20M_LR2_run2_BEST.pickle")
    #run_policy_with_resets("BaodingHold_64N_run3_BEST.pickle")

    #run_policy("BaodingHold_alive08_cons_run2.pickle")
    #run_policy_with_resets("BaodingHold_alive08_cons_run2.pickle")
