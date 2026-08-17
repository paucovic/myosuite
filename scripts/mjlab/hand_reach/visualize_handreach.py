#!/usr/bin/env python3
"""
Visualize a trained five-fingertip mjlab HandReach policy.

Run from the repo root:

    python scripts/mjlab/hand_reach/visualize_handreach.py \
        --policy-path my_handreach_5fingers.pt

Optional fixed target with 15 values:

    python scripts/mjlab/hand_reach/visualize_handreach.py \
        --policy-path my_handreach_5fingers.pt \
        --target-pos \
        -0.165 -0.537 1.495 \
        -0.151 -0.547 1.455 \
        -0.146 -0.547 1.447 \
        -0.148 -0.543 1.445 \
        -0.148 -0.528 1.434

The target order is:
    THtip xyz,
    IFtip xyz,
    MFtip xyz,
    RFtip xyz,
    LFtip xyz
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

from train_handreach import (
    REPO_ROOT,
    TASK_ID,
    QPOS_SLICE,
    QVEL_SLICE,
    TIP_POS_SLICE,
    TARGET_POS_SLICE,
    TARGET_ERR_SLICE,
    NUM_FINGERTIPS,
    FINGERTIP_POS_DIM,
    FINGER_NAMES,
    NOMINAL_FINGERTIP_TARGETS,
    apply_training_config,
    get_policy_action,
    import_mjlab_tools,
    make_wrapped_env,
    redirect_output,
    remove_unsupported_rsl_rl_keys,
)


DEFAULT_POLICY_PATH = "my_handreach_5fingers.pt"
DEFAULT_ENV_ID = TASK_ID

NUM_VIEWER_ENVS = 1

# This is only the viewer playback delay.
# It does NOT change the simulation timestep.
DEFAULT_VIEWER_DT = 0.06

TARGET_MARKER_RADIUS = 0.008
ACTUAL_MARKER_RADIUS = 0.007
REFERENCE_LINE_RADIUS = 0.0025

# None disables terminal status printing.
PRINT_EVERY_N_STEPS = 20


# =============================================================================
# PATH / RUNNER HELPERS
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
    make_env, load_rl_cfg = import_mjlab_tools(quiet=quiet)

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


def load_policy_into_runner(runner, policy_path: Path) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    try:
        runner.load(str(policy_path), load_optimizer=False)
    except TypeError:
        runner.load(str(policy_path))


# =============================================================================
# MUJOCO MODEL EXTRACTION
# =============================================================================

def get_nested_attr(obj: Any, path: str) -> Any | None:
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
    known_paths = [
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
    ]

    for path in known_paths:
        value = get_nested_attr(root, path)

        if isinstance(value, mujoco.MjModel):
            return value

    visited = set()

    def search(obj: Any, depth: int) -> mujoco.MjModel | None:
        if obj is None:
            return None

        if isinstance(obj, mujoco.MjModel):
            return obj

        if depth <= 0:
            return None

        obj_id = id(obj)

        if obj_id in visited:
            return None

        visited.add(obj_id)

        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple)):
            children = list(obj)
        elif hasattr(obj, "__dict__"):
            children = list(vars(obj).values())
        else:
            return None

        for child in children:
            result = search(child, depth - 1)

            if result is not None:
                return result

        return None

    return search(root, max_depth)


def print_possible_model_attributes(obj: Any) -> None:
    print()
    print("Could not automatically find a mujoco.MjModel inside the mjlab env.")
    print("Useful attributes containing model/mj/sim/scene/xml:")

    try:
        names = sorted(dir(obj))
    except Exception:
        names = []

    for name in names:
        lower = name.lower()

        if (
            "model" not in lower
            and "mj" not in lower
            and "sim" not in lower
            and "scene" not in lower
            and "xml" not in lower
        ):
            continue

        try:
            value = getattr(obj, name)
            print(f"  {name}: {type(value)}")
        except Exception:
            print(f"  {name}: <could not read>")


def get_mujoco_model(env_wrapper) -> mujoco.MjModel:
    raw_env = env_wrapper.env

    model = find_mujoco_model_from_object(raw_env)

    if model is not None:
        return model

    print_possible_model_attributes(raw_env)

    raise RuntimeError(
        "The visualizer could not find the MuJoCo model inside the mjlab env. "
        "Send me the printed attribute list and I will adapt the extractor."
    )


# =============================================================================
# OBSERVATION / TARGET HELPERS
# =============================================================================

def target_values_to_array(values: list[float] | None) -> np.ndarray | None:
    if values is None:
        return None

    arr = np.array(values, dtype=np.float32)

    if arr.shape[0] != FINGERTIP_POS_DIM:
        raise ValueError(
            f"--target-pos must contain exactly {FINGERTIP_POS_DIM} values: "
            f"{NUM_FINGERTIPS} fingertips * xyz."
        )

    return arr.reshape(NUM_FINGERTIPS, 3)


def nominal_targets_array() -> np.ndarray:
    return np.array(
        NOMINAL_FINGERTIP_TARGETS,
        dtype=np.float32,
    ).reshape(NUM_FINGERTIPS, 3)


def build_fixed_target(
    target_pos_values: list[float] | None,
    use_nominal_target: bool,
) -> np.ndarray | None:
    if target_pos_values is not None and use_nominal_target:
        raise ValueError(
            "Use either --target-pos or --use-nominal-target, not both."
        )

    if target_pos_values is not None:
        return target_values_to_array(target_pos_values)

    if use_nominal_target:
        return nominal_targets_array()

    return None


def reset_episode(
    env,
    seed: int,
    fixed_target_pos: np.ndarray | None,
):
    obs = env.reset(seed=seed)

    if fixed_target_pos is not None:
        target = torch.tensor(
            fixed_target_pos,
            device=env.device,
            dtype=torch.float32,
        ).reshape(NUM_FINGERTIPS, 3)

        env.target_positions[:] = target.unsqueeze(0)
        obs = env.get_observations()

    return obs


def obs_to_values(
    obs,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    policy_obs = obs["policy"][0].detach().cpu().numpy()

    qpos = np.array(policy_obs[QPOS_SLICE], dtype=np.float64)
    qvel = np.array(policy_obs[QVEL_SLICE], dtype=np.float64)

    tip_pos = np.array(policy_obs[TIP_POS_SLICE], dtype=np.float64).reshape(
        NUM_FINGERTIPS,
        3,
    )

    target_pos = np.array(policy_obs[TARGET_POS_SLICE], dtype=np.float64).reshape(
        NUM_FINGERTIPS,
        3,
    )

    target_err = np.array(policy_obs[TARGET_ERR_SLICE], dtype=np.float64).reshape(
        NUM_FINGERTIPS,
        3,
    )

    per_finger_distance = np.linalg.norm(target_err, axis=-1)

    distance_sum = float(np.sum(per_finger_distance))
    distance_mean = float(np.mean(per_finger_distance))
    distance_max = float(np.max(per_finger_distance))

    return (
        qpos,
        qvel,
        tip_pos,
        target_pos,
        target_err,
        per_finger_distance,
        distance_sum,
        distance_mean,
        distance_max,
    )


# =============================================================================
# MUJOCO VISUAL STATE HELPERS
# =============================================================================

def name_or_empty(model: mujoco.MjModel, obj_type, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)

    if name is None:
        return ""

    return name


def find_tip_site_ids(
    model: mujoco.MjModel,
    requested_site_names: tuple[str, ...],
) -> list[int | None]:
    site_ids: list[int | None] = []

    for site_name in requested_site_names:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id >= 0:
            site_ids.append(int(site_id))
        else:
            site_ids.append(None)

    return site_ids


def set_hand_pose_from_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    nq = min(model.nq, qpos.shape[0])
    data.qpos[:nq] = qpos[:nq]

    mujoco.mj_forward(model, data)


def clear_geom_label(geom) -> None:
    """
    Prevent MuJoCo viewer from displaying garbage text labels on custom geoms.
    """
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
    rgba: list[float],
) -> None:
    scene = viewer.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.array(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.array(rgba, dtype=np.float32),
    )

    clear_geom_label(geom)

    scene.ngeom += 1


def add_reference_line(
    viewer,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    radius: float,
    rgba: list[float],
) -> None:
    scene = viewer.user_scn

    if scene.ngeom >= scene.maxgeom:
        return

    if float(np.linalg.norm(end_pos - start_pos)) < 1e-8:
        return

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.array(start_pos, dtype=np.float64),
        np.array(end_pos, dtype=np.float64),
    )

    geom.rgba[:] = np.array(rgba, dtype=np.float32)
    clear_geom_label(geom)

    scene.ngeom += 1


def configure_viewer(viewer) -> None:
    """
    Hide MuJoCo labels that can show as garbage text on custom user geoms.
    """
    try:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
    except Exception:
        pass


def print_status(
    step: int,
    reward: float,
    distance_sum: float,
    distance_mean: float,
    distance_max: float,
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    per_finger_distance: np.ndarray,
) -> None:
    print(
        f"step={step:05d} "
        f"reward={reward:+.4f} "
        f"dist_sum={distance_sum:.5f} m "
        f"dist_mean={distance_mean:.5f} m "
        f"dist_max={distance_max:.5f} m"
    )

    for finger_name, tip, target, dist in zip(
        FINGER_NAMES,
        tip_pos,
        target_pos,
        per_finger_distance,
    ):
        print(
            f"  {finger_name}: "
            f"dist={dist:.5f} "
            f"tip=[{tip[0]:+.3f}, {tip[1]:+.3f}, {tip[2]:+.3f}] "
            f"target=[{target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f}]"
        )


# =============================================================================
# MAIN VISUALIZER
# =============================================================================

def main(args: argparse.Namespace) -> None:
    policy_path = resolve_repo_path(args.policy_path)

    fixed_target_pos = build_fixed_target(
        target_pos_values=args.target_pos,
        use_nominal_target=args.use_nominal_target,
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

    tip_site_ids = find_tip_site_ids(
        model=model,
        requested_site_names=tuple(FINGER_NAMES),
    )

    obs = reset_episode(
        env=env,
        seed=args.seed,
        fixed_target_pos=fixed_target_pos,
    )

    (
        qpos,
        qvel,
        tip_pos,
        target_pos,
        target_err,
        per_finger_distance,
        distance_sum,
        distance_mean,
        distance_max,
    ) = obs_to_values(obs)

    set_hand_pose_from_obs(
        model=model,
        data=data,
        qpos=qpos,
    )

    print("=" * 80)
    print("MJLAB FIVE-FINGER HANDREACH POLICY VISUALIZER")
    print("=" * 80)
    print(f"policy:       {policy_path}")
    print(f"env:          {args.env_id}")
    print(f"device:       {args.device}")
    print(f"viewer dt:    {args.dt}")
    print(f"fingers:      {FINGER_NAMES}")
    print(f"obs dim:      {obs['policy'].shape[-1]}")
    print("green marker  = target position")
    print("red marker    = fingertip position")
    print("yellow line   = target/fingertip distance")
    print("=" * 80)

    print("Tip site lookup:")
    for finger_name, site_id in zip(FINGER_NAMES, tip_site_ids):
        if site_id is None:
            print(f"  {finger_name}: not found, using obs position only")
        else:
            site_name = name_or_empty(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
            )
            print(f"  {finger_name}: site id {site_id}, name '{site_name}'")

    if fixed_target_pos is not None:
        print("=" * 80)
        print("Using fixed target positions:")
        for finger_name, target in zip(FINGER_NAMES, fixed_target_pos):
            print(
                f"  {finger_name}: "
                f"[{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}]"
            )
    else:
        print("=" * 80)
        print("Using randomized targets from train_handreach.py wrapper.")

    print("=" * 80)

    viewer_step = 0
    episode_index = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer(viewer)

        while viewer.is_running():
            step_start = time.time()

            action = get_policy_action(
                runner=runner,
                obs=obs,
                device=args.device,
            )

            obs, reward, done, _ = env.step(action)

            if fixed_target_pos is not None:
                target = torch.tensor(
                    fixed_target_pos,
                    device=env.device,
                    dtype=torch.float32,
                ).reshape(NUM_FINGERTIPS, 3)

                env.target_positions[:] = target.unsqueeze(0)
                obs = env.get_observations()

            (
                qpos,
                qvel,
                tip_pos,
                target_pos,
                target_err,
                per_finger_distance,
                distance_sum,
                distance_mean,
                distance_max,
            ) = obs_to_values(obs)

            set_hand_pose_from_obs(
                model=model,
                data=data,
                qpos=qpos,
            )

            viewer.user_scn.ngeom = 0

            for finger_index in range(NUM_FINGERTIPS):
                current_tip_pos = tip_pos[finger_index]
                current_target_pos = target_pos[finger_index]

                add_sphere_marker(
                    viewer=viewer,
                    position=current_target_pos,
                    radius=TARGET_MARKER_RADIUS,
                    rgba=[0.0, 1.0, 0.0, 0.70],
                )

                add_sphere_marker(
                    viewer=viewer,
                    position=current_tip_pos,
                    radius=ACTUAL_MARKER_RADIUS,
                    rgba=[1.0, 0.0, 0.0, 0.80],
                )

                add_reference_line(
                    viewer=viewer,
                    start_pos=current_tip_pos,
                    end_pos=current_target_pos,
                    radius=REFERENCE_LINE_RADIUS,
                    rgba=[1.0, 1.0, 0.0, 0.65],
                )

            viewer.sync()

            reward_scalar = float(reward.detach().mean().cpu())

            if args.print_every is not None and viewer_step % args.print_every == 0:
                print_status(
                    step=viewer_step,
                    reward=reward_scalar,
                    distance_sum=distance_sum,
                    distance_mean=distance_mean,
                    distance_max=distance_max,
                    tip_pos=tip_pos,
                    target_pos=target_pos,
                    per_finger_distance=per_finger_distance,
                )

            done_now = bool(done.detach().cpu().reshape(-1)[0].item())

            if args.reset_on_done and done_now:
                episode_index += 1

                obs = reset_episode(
                    env=env,
                    seed=args.seed + episode_index,
                    fixed_target_pos=fixed_target_pos,
                )

            viewer_step += 1

            sleep_time = args.dt - (time.time() - step_start)

            if sleep_time > 0.0:
                time.sleep(sleep_time)

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
        nargs=FINGERTIP_POS_DIM,
        default=None,
        metavar="V",
        help=(
            "Optional fixed target positions. Must provide exactly 15 values: "
            "THtip xyz, IFtip xyz, MFtip xyz, RFtip xyz, LFtip xyz."
        ),
    )

    parser.add_argument(
        "--use-nominal-target",
        action="store_true",
        help="Use the nominal fixed five-fingertip target positions from train_handreach.py.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_VIEWER_DT,
        help="Seconds between viewer updates. Larger is slower. Does not change physics dt.",
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY_N_STEPS,
        help="Print status every N viewer steps. Use 0 for every step.",
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

    if args.print_every is not None and args.print_every <= 0:
        args.print_every = 1

    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {args.device}, but CUDA is not available."
        )

    if args.dt <= 0.0:
        raise ValueError("--dt must be greater than 0.")

    if args.target_pos is not None and len(args.target_pos) != FINGERTIP_POS_DIM:
        raise ValueError(
            f"--target-pos must have exactly {FINGERTIP_POS_DIM} values."
        )

    if args.target_pos is not None and args.use_nominal_target:
        raise ValueError(
            "Use either --target-pos or --use-nominal-target, not both."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    validate_args(parsed_args)
    main(parsed_args)