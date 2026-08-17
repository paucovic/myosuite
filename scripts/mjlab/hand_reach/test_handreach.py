from __future__ import annotations

import argparse
import traceback

import torch

from register_handreach import TASK_ID, register_hand_reach_task


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def describe_tensor(name: str, value) -> None:
    if hasattr(value, "shape"):
        print(
            f"{name}: shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, device={value.device}"
        )
    else:
        print(f"{name}: type={type(value)}")


def describe_obs(obs) -> None:
    print_header("OBSERVATION")

    if isinstance(obs, dict):
        for key, value in obs.items():
            describe_tensor(f"obs[{key!r}]", value)
    else:
        describe_tensor("obs", obs)


def describe_step_output(step_output) -> None:
    print_header("STEP OUTPUT")

    print("type:", type(step_output))

    if not isinstance(step_output, tuple):
        print(step_output)
        return

    print("tuple length:", len(step_output))

    for index, value in enumerate(step_output):
        if isinstance(value, dict):
            print(f"out[{index}]: dict keys={list(value.keys())}")

            for key, dict_value in value.items():
                describe_tensor(f"  {key}", dict_value)
        else:
            describe_tensor(f"out[{index}]", value)


def print_env_attributes(env) -> None:
    print_header("ENV ATTRIBUTES")

    attribute_names = [
        "num_envs",
        "num_actions",
        "action_dim",
        "num_acts",
        "num_observations",
        "observation_space",
        "action_space",
        "device",
        "cfg",
    ]

    for name in attribute_names:
        if not hasattr(env, name):
            continue

        try:
            value = getattr(env, name)
            print(f"{name}: {value}")
        except Exception:
            print(f"{name}: <could not read>")


def infer_num_actions(env) -> int:
    for name in ["num_actions", "action_dim", "num_acts"]:
        value = getattr(env, name, None)

        if isinstance(value, int):
            return value

    action_space = getattr(env, "action_space", None)

    if action_space is not None and hasattr(action_space, "shape"):
        shape = action_space.shape

        if len(shape) > 0:
            return int(shape[-1])

    action_manager = getattr(env, "action_manager", None)

    if action_manager is not None:
        for name in ["action_dim", "total_action_dim", "num_actions"]:
            value = getattr(action_manager, name, None)

            if isinstance(value, int):
                return value

    raise RuntimeError("Could not infer action dimension from the mjlab env.")


def main(args: argparse.Namespace) -> None:
    register_hand_reach_task(quiet=True)

    from myosuite.core.registry import make_env

    print_header("HAND REACH MJLAB LOAD TEST")
    print("task id: ", TASK_ID)
    print("device:  ", args.device)
    print("num envs:", args.num_envs)

    env = make_env(
        TASK_ID,
        backend="mjlab",
        device=args.device,
        num_envs=args.num_envs,
    )

    print_env_attributes(env)

    print_header("RESET")
    obs = env.reset()
    describe_obs(obs)

    num_actions = infer_num_actions(env)

    print_header("ACTION")
    print("inferred action dimension:", num_actions)

    action = torch.zeros(
        (args.num_envs, num_actions),
        device=args.device,
        dtype=torch.float32,
    )

    describe_tensor("zero action", action)

    print_header("STEP")
    step_output = env.step(action)
    describe_step_output(step_output)

    print_header("SUCCESS")
    print("HandReach was created, reset, and stepped successfully.")

    try:
        env.close()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()

    try:
        main(parsed_args)
    except Exception:
        traceback.print_exc()
        raise