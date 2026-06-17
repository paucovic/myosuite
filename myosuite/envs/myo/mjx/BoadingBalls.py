import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx import make

#CODE FOR DEBUGGING THE BAODING ENVIRONMENT.
def main():
    env = make("MjxBaodingDebug-v0", config_overrides={"impl": "warp","num_envs": 1})

    print("Environment created.")
    print("Action size:", env.action_size)
    print("nq:", env.mjx_model.nq)
    print("nv:", env.mjx_model.nv)
    print("nu:", env.mjx_model.nu)

    rng = jax.random.PRNGKey(0)

    state = env.reset(rng)
    
    print("Reset worked.")
    print("Initial reward:", state.reward)
    print("Initial done:", state.done)
    print("Observation keys:", state.obs.keys())
    print("Observation shape:", state.obs["state"].shape)

    action = jp.zeros(env.action_size)

    for i in range(250):
        state = env.step(state, action)

        b = env._get_baoding_state(state.data, state.info)

        print(f"Step {i+1}:")
        print(f"reward={state.reward}")
        print(f"done={state.done}")
        print(f"step_count={state.info['step_count']}")
        print(f"object1={b['object1_pos']}")
        print(f"object2={b['object2_pos']}")
        print(f"target1={state.info['target1_pos']}")
        print(f"target2={state.info['target2_pos']}")
        print(f"pos_dist_1={state.metrics['pos_dist_1']}")
        print(f"pos_dist_2={state.metrics['pos_dist_2']}")
        print(f"pos_reward_1={state.metrics['pos_reward_1']}")
        print(f"pos_reward_2={state.metrics['pos_reward_2']}")
        print(f"act_reg_reward={state.metrics['act_reg_reward']}")
        print(f"alive_reward={state.metrics['alive_reward']}")
        print(f"drop_penalty={state.metrics['drop_penalty']}")
        print(f"solved_bonus={state.metrics['solved_bonus']}")

        if float(state.done) == 1.0:
            rng, rng_reset = jax.random.split(rng)
            state = env.reset(rng_reset)


if __name__ == "__main__":
    main()