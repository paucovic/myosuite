import functools
from brax.io import model

import jax
import jax.numpy as jp
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from myosuite.envs.myo.mjx import make, ppo_config


# ============================================================
# CHANGE THESE
# ============================================================

POLICIES_TO_COMPARE = [
    "BaodingHold_64N_20M_LR2_run2_BEST.pickle",
    "BaodingHold_64N_20M_LR2_run2.pickle",
]

ENV_NAME = "MjxBaodingDebug-v0"
IMPL = "warp"
CONTACT_BUFFER_NUM_ENVS = 32

NUM_ENVS = 64
MAX_STEPS = 400
SEED = 0


# ============================================================
# CODE
# ============================================================

def build_env_and_policy(params_path):


    env = make(
        ENV_NAME,
        config_overrides={
            "impl": IMPL,
            # This is used by your env to size MJX/Warp contact buffers.
            # It is NOT the number of rollouts in this manual evaluator.
            "num_envs": CONTACT_BUFFER_NUM_ENVS,
        },
    )

    print("Compare env config:")
    print("  env:", ENV_NAME)
    print("  impl:", env._config.impl)
    print("  env config num_envs:", env._config.num_envs)
    print("  manual parallel rollouts:", NUM_ENVS)
    print("  hold_dist_th:", env._config.hold_dist_th)
    print("  goal_time_period:", env._config.goal_time_period)
    print("  max_episode_steps:", env._config.max_episode_steps)
    print("  ctrl_dt:", env._config.ctrl_dt)
    print("  observation_size:", env.observation_size)
    print("  action_size:", env.action_size)

    ppo_params = dict(ppo_config)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.pop("network_factory"),
    )

    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
    )

    params = model.load_params(params_path)

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    return env, inference_fn


def evaluate_policy_parallel(params_path):
    env, inference_fn = build_env_and_policy(params_path)

    reset_many = jax.jit(jax.vmap(env.reset))
    step_many = jax.jit(jax.vmap(env.step))

    @jax.jit
    def policy_many(obs, rngs):
        def one_policy(one_obs, one_rng):
            action, _ = inference_fn(one_obs, one_rng)
            return action

        return jax.vmap(one_policy)(obs, rngs)

    rng = jax.random.PRNGKey(SEED)

    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, NUM_ENVS)

    state = reset_many(reset_keys)

    total_rewards = jp.zeros((NUM_ENVS,))
    survived_steps = jp.ones((NUM_ENVS,), dtype=jp.int32) * MAX_STEPS
    solved_steps = jp.zeros((NUM_ENVS,))

    alive = jp.ones((NUM_ENVS,), dtype=jp.bool_)

    # Store distances at the last counted step for each env.
    last_pos_dist_1 = jp.zeros((NUM_ENVS,))
    last_pos_dist_2 = jp.zeros((NUM_ENVS,))

    for step_idx in range(MAX_STEPS):
        rng, action_rng = jax.random.split(rng)
        action_keys = jax.random.split(action_rng, NUM_ENVS)

        actions = policy_many(state.obs, action_keys)
        state = step_many(state, actions)

        reward_vec = jp.asarray(state.reward)
        done_vec = jp.asarray(state.done) > 0.5

        pos_dist_1 = jp.asarray(state.metrics["pos_dist_1"])
        pos_dist_2 = jp.asarray(state.metrics["pos_dist_2"])

        # Count reward only while the env is alive.
        total_rewards = total_rewards + reward_vec * alive.astype(jp.float32)

        # Store the last relevant distances before/at failure.
        last_pos_dist_1 = jp.where(alive, pos_dist_1, last_pos_dist_1)
        last_pos_dist_2 = jp.where(alive, pos_dist_2, last_pos_dist_2)

        solved_vec = (
            (pos_dist_1 < env._config.proximity_th)
            & (pos_dist_2 < env._config.proximity_th)
            & (~done_vec)
            & alive
        )
        solved_steps = solved_steps + solved_vec.astype(jp.float32)

        newly_done = alive & done_vec
        survived_steps = jp.where(newly_done, step_idx + 1, survived_steps)

        alive = alive & (~done_vec)

    total_rewards = np.asarray(total_rewards)
    survived_steps = np.asarray(survived_steps)
    solved_steps = np.asarray(solved_steps)
    last_pos_dist_1 = np.asarray(last_pos_dist_1)
    last_pos_dist_2 = np.asarray(last_pos_dist_2)

    print()
    print("=" * 80)
    print(f"Policy: {params_path}")
    print(f"Parallel evaluation over {NUM_ENVS} independent rollouts")
    print("=" * 80)

    print(f"Mean total reward: {np.mean(total_rewards):.3f}")
    print(f"Min total reward:  {np.min(total_rewards):.3f}")
    print(f"Max total reward:  {np.max(total_rewards):.3f}")
    print()

    print(f"Mean survived steps: {np.mean(survived_steps):.1f}/{MAX_STEPS}")
    print(f"Min survived steps:  {np.min(survived_steps)}/{MAX_STEPS}")
    print(f"Max survived steps:  {np.max(survived_steps)}/{MAX_STEPS}")
    print(f"Mean survived seconds: {np.mean(survived_steps) * env._config.ctrl_dt:.3f}")
    print()

    print(f"Failures: {np.sum(survived_steps < MAX_STEPS)}/{NUM_ENVS}")
    print(f"Mean solved steps: {np.mean(solved_steps):.2f}")
    print()

    print(f"Mean final/terminal pos_dist_1: {np.mean(last_pos_dist_1):.4f}")
    print(f"Mean final/terminal pos_dist_2: {np.mean(last_pos_dist_2):.4f}")
    print(f"Max final/terminal pos_dist_1:  {np.max(last_pos_dist_1):.4f}")
    print(f"Max final/terminal pos_dist_2:  {np.max(last_pos_dist_2):.4f}")
    print()


if __name__ == "__main__":
    for policy_path in POLICIES_TO_COMPARE:
        evaluate_policy_parallel(policy_path)