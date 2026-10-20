#!/usr/bin/env python3
"""Small simulation smoke test for the MJLab Baoding hold task.

Run from the repository root:

    python scripts/mjlab/BaodingBalls/test_baoding.py
"""

from __future__ import annotations

import torch

from register_baoding import (
    BALL1_ENTITY_NAME,
    BALL2_ENTITY_NAME,
    DROP_HEIGHT,
    EXPECTED_ACTION_DIM,
    OBS_DIM,
    TASK_ID,
    register_baoding_hold_task,
)


SEED = 0
NUM_TEST_STEPS = 50
PRINT_EVERY = 1

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


def ball_position(
    env,
    entity_name: str,
) -> torch.Tensor:
    """Return one ball's xyz position for environment zero."""
    return (
        env.scene[entity_name]
        .data.root_link_pos_w[0]
        .detach()
        .cpu()
    )


def main() -> None:
    torch.manual_seed(SEED)

    print("=" * 80)
    print("MJLAB BAODING HOLD SMOKE TEST")
    print("=" * 80)
    print(f"Task:   {TASK_ID}")
    print(f"Device: {DEVICE}")

    # Register the custom task before calling make_env().
    register_baoding_hold_task(
        quiet=True,
    )

    from myosuite.core.registry import (
        make_env,
    )

    env = None

    try:
        env = make_env(
            TASK_ID,
            backend="mjlab",
            device=DEVICE,
            num_envs=1,
        )

        # Ignore the direct reset return because the current observation
        # can be obtained consistently through the observation manager.
        env.reset(seed=SEED)

        observations = (
            env.observation_manager.compute()
        )

        if "policy" not in observations:
            raise RuntimeError(
                "The environment did not produce "
                "a 'policy' observation group."
            )

        policy_obs = observations["policy"]

        action_dim = int(
            env.action_space.shape[-1]
        )

        print()
        print("Environment created successfully")
        print(
            "Policy observation shape:",
            tuple(policy_obs.shape),
        )
        print(
            "Action dimension:",
            action_dim,
        )

        expected_obs_shape = (
            1,
            OBS_DIM,
        )

        if (
            tuple(policy_obs.shape)
            != expected_obs_shape
        ):
            raise RuntimeError(
                "Unexpected observation shape. "
                f"Expected {expected_obs_shape}, "
                f"but received "
                f"{tuple(policy_obs.shape)}."
            )

        if (
            action_dim
            != EXPECTED_ACTION_DIM
        ):
            raise RuntimeError(
                "Unexpected action dimension. "
                f"Expected "
                f"{EXPECTED_ACTION_DIM}, "
                f"but received {action_dim}."
            )

        if not torch.isfinite(
            policy_obs
        ).all():
            raise RuntimeError(
                "The initial observation "
                "contains NaN or infinite values."
            )

        ball1_initial = ball_position(
            env,
            BALL1_ENTITY_NAME,
        )
        ball2_initial = ball_position(
            env,
            BALL2_ENTITY_NAME,
        )

        print()
        print("Initial state")
        print(
            "Ball 1 position:",
            ball1_initial.tolist(),
        )
        print(
            "Ball 2 position:",
            ball2_initial.tolist(),
        )
        print(
            "Drop threshold:",
            DROP_HEIGHT,
        )

        if (
            float(ball1_initial[2])
            <= DROP_HEIGHT
        ):
            raise RuntimeError(
                "Ball 1 begins below the "
                "drop threshold."
            )

        if (
            float(ball2_initial[2])
            <= DROP_HEIGHT
        ):
            raise RuntimeError(
                "Ball 2 begins below the "
                "drop threshold."
            )

        zero_actions = torch.zeros(
            (
                1,
                action_dim,
            ),
            device=env.device,
            dtype=torch.float32,
        )

        print()
        print(
            "Stepping with zero muscle actions"
        )
        print("-" * 80)

        episode_ended = False

        for step in range(
            1,
            NUM_TEST_STEPS + 1,
        ):
            (
                observations,
                reward,
                terminated,
                truncated,
                _extras,
            ) = env.step(zero_actions)

            policy_obs = observations[
                "policy"
            ]

            if not torch.isfinite(
                policy_obs
            ).all():
                raise RuntimeError(
                    f"Observation became "
                    f"non-finite at step "
                    f"{step}."
                )

            if not torch.isfinite(
                reward
            ).all():
                raise RuntimeError(
                    f"Reward became "
                    f"non-finite at step "
                    f"{step}."
                )

            ball1_pos = ball_position(
                env,
                BALL1_ENTITY_NAME,
            )
            ball2_pos = ball_position(
                env,
                BALL2_ENTITY_NAME,
            )

            reward_value = float(
                reward[0].detach().cpu()
            )
            terminated_value = bool(
                terminated[0]
                .detach()
                .cpu()
                .item()
            )
            truncated_value = bool(
                truncated[0]
                .detach()
                .cpu()
                .item()
            )

            ball1_z = float(
                ball1_pos[2]
            )
            ball2_z = float(
                ball2_pos[2]
            )

            below_drop_height = (
                ball1_z < DROP_HEIGHT
                or ball2_z < DROP_HEIGHT
            )

            done = (
                terminated_value
                or truncated_value
            )

            if (
                step % PRINT_EVERY == 0
                or done
            ):
                print(
                    f"step={step:03d} "
                    f"reward={reward_value:+7.3f} "
                    f"ball1_z={ball1_z:.4f} "
                    f"ball2_z={ball2_z:.4f} "
                    f"below_threshold="
                    f"{below_drop_height} "
                    f"terminated="
                    f"{terminated_value} "
                    f"truncated="
                    f"{truncated_value}"
                )

            if done:
                episode_ended = True

                print("-" * 80)
                print(
                    "The episode ended. With "
                    "zero muscle actions, a "
                    "ball drop is expected."
                )
                print(
                    "MJLab may already have "
                    "reset the scene, so the "
                    "displayed positions can "
                    "be the reset positions."
                )
                break

        print()
        print("=" * 80)
        print("SMOKE TEST PASSED")
        print("=" * 80)
        print(
            "The environment reset and "
            "completed simulation steps "
            "without invalid observations "
            "or rewards."
        )

        if not episode_ended:
            print(
                f"No termination occurred "
                f"during the first "
                f"{NUM_TEST_STEPS} steps."
            )

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()