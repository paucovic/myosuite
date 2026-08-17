#!/usr/bin/env python3
"""
Visualize a trained two-fingertip mjlab HandReach policy.

Run from the repository root:

    python scripts/mjlab/hand_reach/visualize_handreach_2F.py \
        --policy-path my_handreach_2F_policy.pt

Optional fixed target, using THtip xyz followed by IFtip xyz:

    python scripts/mjlab/hand_reach/visualize_handreach_2F.py \
        --policy-path my_handreach_2F_policy.pt \
        --target-pos \
        -0.174 -0.522 1.482 \
        -0.160 -0.574 1.463
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import torch

from register_handreach_2F import (
    FINGERTIP_SITE_NAMES,
    NUM_FINGERTIPS,
    TASK_ID,
)
from train_handreach_2F import (
    OBS_DIM,
    QPOS_SLICE,
    REPO_ROOT,
    TARGET_ERR_SLICE,
    TARGET_POS_SLICE,
    TIP_POS_SLICE,
    apply_training_config,
    get_policy_action,
    import_mjlab_tools,
    make_wrapped_env,
    redirect_output,
    remove_unsupported_rsl_rl_keys,
)


DEFAULT_POLICY_PATH = "my_handreach_2F_policy.pt"
DEFAULT_ENV_ID = TASK_ID

NUM_VIEWER_ENVS = 1

# Viewer delay only. This does not change the simulation timestep.
DEFAULT_VIEWER_DT = 0.06

TARGET_MARKER_RADIUS = 0.015
ACTUAL_MARKER_RADIUS = 0.012
REFERENCE_LINE_RADIUS = 0.003

PRINT_EVERY_N_STEPS = 20

# THtip target, IFtip target.
TARGET_COLORS = (
    (0.10, 1.00, 0.10, 0.75),
    (0.10, 0.75, 1.00, 0.75),
)

# THtip actual, IFtip actual.
ACTUAL_COLORS = (
    (1.00, 0.10, 0.10, 0.85),
    (1.00, 0.10, 0.90, 0.85),
)

LINE_COLORS = (
    (1.00, 1.00, 0.10, 0.70),
    (1.00, 0.60, 0.10, 0.70),
)


# =============================================================================
# PATH AND RUNNER HELPERS
# =============================================================================

def resolve_repo_path(path_arg: str) -> Path:
    path = Path(path_arg)

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path


def build_runner(
    env_id: str,
    device: str,
    seed: int,
    quiet: bool,
):
    make_env, load_rl_cfg = import_mjlab_tools(
        quiet=quiet
    )

    from rsl_rl.runners import OnPolicyRunner

    env = make_wrapped_env(
        make_env=make_env,
        env_id=env_id,
        device=device,
        num_envs=NUM_VIEWER_ENVS,
        seed=seed,
        quiet=quiet,
    )

    train_cfg = load_rl_cfg(env_id)

    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }

    train_cfg_dict = asdict(train_cfg)

    apply_training_config(train_cfg_dict)
    remove_unsupported_rsl_rl_keys(train_cfg_dict)

    with redirect_output(quiet):
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=None,
            device=device,
        )

    return env, runner


def load_policy_into_runner(
    runner,
    policy_path: Path,
) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy file not found: {policy_path}"
        )

    try:
        runner.load(
            str(policy_path),
            load_optimizer=False,
        )
    except TypeError:
        runner.load(str(policy_path))


# =============================================================================
# MUJOCO MODEL EXTRACTION
# =============================================================================

def get_nested_attr(
    obj: Any,
    path: str,
) -> Any | None:
    current = obj

    for part in path.split("."):
        if current is None:
            return None

        if not hasattr(current, part):
            return None

        try:
            current = getattr(current, part)
        except Exception:
            return None

    return current


def find_mujoco_model_from_object(
    root: Any,
    max_depth: int = 5,
) -> mujoco.MjModel | None:
    known_paths = (
        "mj_model",
        "_mj_model",
        "model",
        "_model",
        "mujoco_model",
        "_mujoco_model",
        "scene.mj_model",
        "scene._mj_model",
        "scene.model",
        "scene._model",
        "sim.model",
        "physics.model",
    )

    for path in known_paths:
        value = get_nested_attr(root, path)

        if isinstance(value, mujoco.MjModel):
            return value

    visited: set[int] = set()

    def search(
        obj: Any,
        depth: int,
    ) -> mujoco.MjModel | None:
        if obj is None:
            return None

        if isinstance(obj, mujoco.MjModel):
            return obj

        if depth <= 0:
            return None

        object_id = id(obj)

        if object_id in visited:
            return None

        visited.add(object_id)

        if isinstance(obj, dict):
            children = list(obj.values())

        elif isinstance(obj, (list, tuple)):
            children = list(obj)

        elif hasattr(obj, "__dict__"):
            children = list(vars(obj).values())

        else:
            return None

        for child in children:
            result = search(
                child,
                depth - 1,
            )

            if result is not None:
                return result

        return None

    return search(
        root,
        max_depth,
    )


def get_mujoco_model(
    env_wrapper,
) -> mujoco.MjModel:
    model = find_mujoco_model_from_object(
        env_wrapper.env
    )

    if model is None:
        raise RuntimeError(
            "The visualizer could not find a "
            "mujoco.MjModel inside the mjlab environment."
        )

    return model


# =============================================================================
# TARGET AND OBSERVATION HELPERS
# =============================================================================

def target_tensor(
    values: list[float],
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(
        values,
        device=device,
        dtype=torch.float32,
    ).reshape(
        NUM_FINGERTIPS,
        3,
    )


def apply_fixed_target(
    env,
    values: list[float],
) -> None:
    target = target_tensor(
        values,
        env.device,
    )

    env.target_positions[:] = target.unsqueeze(0)


def reset_episode(
    env,
    seed: int,
    fixed_target_pos: list[float] | None,
):
    obs = env.reset(seed=seed)

    if fixed_target_pos is not None:
        apply_fixed_target(
            env,
            fixed_target_pos,
        )

        obs = env.get_observations()

    return obs


def obs_to_values(obs):
    policy_obs = (
        obs["policy"][0]
        .detach()
        .cpu()
        .numpy()
    )

    if policy_obs.shape[0] != OBS_DIM:
        raise RuntimeError(
            f"Expected observation dimension {OBS_DIM}, "
            f"got {policy_obs.shape[0]}."
        )

    qpos = np.asarray(
        policy_obs[QPOS_SLICE],
        dtype=np.float64,
    )

    tip_pos = np.asarray(
        policy_obs[TIP_POS_SLICE],
        dtype=np.float64,
    ).reshape(
        NUM_FINGERTIPS,
        3,
    )

    target_pos = np.asarray(
        policy_obs[TARGET_POS_SLICE],
        dtype=np.float64,
    ).reshape(
        NUM_FINGERTIPS,
        3,
    )

    target_err = np.asarray(
        policy_obs[TARGET_ERR_SLICE],
        dtype=np.float64,
    ).reshape(
        NUM_FINGERTIPS,
        3,
    )

    expected_error = target_pos - tip_pos

    if not np.allclose(
        target_err,
        expected_error,
        atol=1e-6,
        rtol=1e-5,
    ):
        maximum_error = np.max(
            np.abs(target_err - expected_error)
        )

        raise RuntimeError(
            "The observation is inconsistent: "
            "target_err is not target_pos - tip_pos. "
            f"Maximum discrepancy: {maximum_error}"
        )

    distances = np.linalg.norm(
        target_err,
        axis=-1,
    )

    return (
        qpos,
        tip_pos,
        target_pos,
        target_err,
        distances,
    )


# =============================================================================
# MUJOCO VISUAL STATE
# =============================================================================

def set_hand_pose_from_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    count = min(
        model.nq,
        qpos.shape[0],
    )

    data.qpos[:count] = qpos[:count]

    mujoco.mj_forward(
        model,
        data,
    )


def clear_geom_label(geom) -> None:
    try:
        geom.label = ""
        return
    except Exception:
        pass

    try:
        geom.label = b""
        return
    except Exception:
        pass

    try:
        geom.label[:] = 0
    except Exception:
        pass


def add_sphere_marker(
    viewer,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    scene = viewer.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array(
            [radius, 0.0, 0.0],
            dtype=np.float64,
        ),
        np.asarray(
            position,
            dtype=np.float64,
        ),
        np.eye(
            3,
            dtype=np.float64,
        ).reshape(-1),
        np.asarray(
            rgba,
            dtype=np.float32,
        ),
    )

    clear_geom_label(geom)

    scene.ngeom += 1


def add_reference_line(
    viewer,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    scene = viewer.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    distance = float(
        np.linalg.norm(
            end_pos - start_pos
        )
    )

    if distance < 1e-8:
        return

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(
            start_pos,
            dtype=np.float64,
        ),
        np.asarray(
            end_pos,
            dtype=np.float64,
        ),
    )

    geom.rgba[:] = np.asarray(
        rgba,
        dtype=np.float32,
    )

    clear_geom_label(geom)

    scene.ngeom += 1


def configure_viewer(viewer) -> None:
    try:
        viewer.opt.label = (
            mujoco.mjtLabel.mjLABEL_NONE
        )
    except Exception:
        pass


def draw_markers(
    viewer,
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
) -> None:
    viewer.user_scn.ngeom = 0

    for finger_index in range(NUM_FINGERTIPS):
        add_sphere_marker(
            viewer=viewer,
            position=target_pos[finger_index],
            radius=TARGET_MARKER_RADIUS,
            rgba=TARGET_COLORS[finger_index],
        )

        add_sphere_marker(
            viewer=viewer,
            position=tip_pos[finger_index],
            radius=ACTUAL_MARKER_RADIUS,
            rgba=ACTUAL_COLORS[finger_index],
        )

        add_reference_line(
            viewer=viewer,
            start_pos=tip_pos[finger_index],
            end_pos=target_pos[finger_index],
            radius=REFERENCE_LINE_RADIUS,
            rgba=LINE_COLORS[finger_index],
        )


# =============================================================================
# TERMINAL STATUS
# =============================================================================

def print_status(
    step: int,
    reward: float,
    distances: np.ndarray,
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
) -> None:
    print(
        f"step={step:05d} "
        f"reward={reward:+.4f} "
        f"mean={float(distances.mean()):.5f} m "
        f"max={float(distances.max()):.5f} m"
    )

    for finger_index, finger_name in enumerate(
        FINGERTIP_SITE_NAMES
    ):
        tip = tip_pos[finger_index]
        target = target_pos[finger_index]

        print(
            f"  {finger_name}: "
            f"distance={distances[finger_index]:.5f} m "
            f"tip=["
            f"{tip[0]:+.3f}, "
            f"{tip[1]:+.3f}, "
            f"{tip[2]:+.3f}] "
            f"target=["
            f"{target[0]:+.3f}, "
            f"{target[1]:+.3f}, "
            f"{target[2]:+.3f}]"
        )


# =============================================================================
# VISUALIZATION
# =============================================================================

def main(
    args: argparse.Namespace,
) -> None:
    policy_path = resolve_repo_path(
        args.policy_path
    )

    env, runner = build_runner(
        env_id=args.env_id,
        device=args.device,
        seed=args.seed,
        quiet=args.quiet,
    )

    load_policy_into_runner(
        runner=runner,
        policy_path=policy_path,
    )

    model = get_mujoco_model(env)
    data = mujoco.MjData(model)

    obs = reset_episode(
        env=env,
        seed=args.seed,
        fixed_target_pos=args.target_pos,
    )

    (
        qpos,
        tip_pos,
        target_pos,
        _target_err,
        distances,
    ) = obs_to_values(obs)

    set_hand_pose_from_obs(
        model=model,
        data=data,
        qpos=qpos,
    )

    print("=" * 80)
    print("MJLAB TWO-FINGER HANDREACH POLICY VISUALIZER")
    print("=" * 80)
    print(f"policy:       {policy_path}")
    print(f"environment:  {args.env_id}")
    print(f"device:       {args.device}")
    print(f"fingertips:   {FINGERTIP_SITE_NAMES}")
    print(f"viewer dt:    {args.dt}")
    print("TH target/actual: green/red")
    print("IF target/actual: cyan/magenta")
    print("Lines connect each fingertip to its target")
    print("=" * 80)

    viewer_step = 0
    episode_index = 0

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:
            configure_viewer(viewer)

            while viewer.is_running():
                step_start = time.time()

                action = get_policy_action(
                    runner=runner,
                    obs=obs,
                    device=args.device,
                )

                (
                    obs,
                    reward,
                    done,
                    _infos,
                ) = env.step(action)

                if args.target_pos is not None:
                    apply_fixed_target(
                        env,
                        args.target_pos,
                    )

                    obs = env.get_observations()

                (
                    qpos,
                    tip_pos,
                    target_pos,
                    _target_err,
                    distances,
                ) = obs_to_values(obs)

                set_hand_pose_from_obs(
                    model=model,
                    data=data,
                    qpos=qpos,
                )

                draw_markers(
                    viewer=viewer,
                    tip_pos=tip_pos,
                    target_pos=target_pos,
                )

                viewer.sync()

                reward_scalar = float(
                    reward.detach().mean().cpu()
                )

                if (
                    args.print_every is not None
                    and viewer_step % args.print_every == 0
                ):
                    print_status(
                        step=viewer_step,
                        reward=reward_scalar,
                        distances=distances,
                        tip_pos=tip_pos,
                        target_pos=target_pos,
                    )

                done_now = bool(
                    done.detach()
                    .cpu()
                    .reshape(-1)[0]
                    .item()
                )

                if (
                    args.reset_on_done
                    and done_now
                ):
                    episode_index += 1

                    obs = reset_episode(
                        env=env,
                        seed=args.seed + episode_index,
                        fixed_target_pos=args.target_pos,
                    )

                viewer_step += 1

                elapsed = time.time() - step_start
                sleep_time = args.dt - elapsed

                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    finally:
        env.close()


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy-path",
        type=str,
        default=DEFAULT_POLICY_PATH,
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default=DEFAULT_ENV_ID,
    )

    parser.add_argument(
        "--target-pos",
        type=float,
        nargs=6,
        default=None,
        metavar=(
            "TH_X",
            "TH_Y",
            "TH_Z",
            "IF_X",
            "IF_Y",
            "IF_Z",
        ),
        help=(
            "Optional fixed THtip xyz followed "
            "by IFtip xyz."
        ),
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
        "--dt",
        type=float,
        default=DEFAULT_VIEWER_DT,
        help=(
            "Seconds between viewer frames. "
            "This does not change the physics timestep."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY_N_STEPS,
    )

    parser.add_argument(
        "--reset-on-done",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if (
        args.print_every is not None
        and args.print_every <= 0
    ):
        args.print_every = 1

    return args


def validate_args(
    args: argparse.Namespace,
) -> None:
    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested device {args.device}, "
            "but CUDA is not available."
        )

    if args.dt <= 0.0:
        raise ValueError(
            "--dt must be greater than zero."
        )

    if NUM_FINGERTIPS != 2:
        raise RuntimeError(
            "This visualizer expects exactly "
            "two fingertips."
        )

    if tuple(FINGERTIP_SITE_NAMES) != (
        "THtip",
        "IFtip",
    ):
        raise RuntimeError(
            "Expected fingertip order "
            "('THtip', 'IFtip')."
        )

    if (
        args.target_pos is not None
        and len(args.target_pos) != 6
    ):
        raise ValueError(
            "--target-pos must contain six values."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    validate_args(parsed_args)
    main(parsed_args)