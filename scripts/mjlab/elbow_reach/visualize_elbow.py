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

from train_elbow import (
    REPO_ROOT,
    apply_training_config,
    get_policy_action,
    import_mjlab_tools,
    make_wrapped_env,
    redirect_output,
    remove_unsupported_rsl_rl_keys,
)


DEFAULT_POLICY_PATH = "my_elbow_policy.pt"
DEFAULT_ENV_ID = "myoElbowPose1D6MFixed-v0"

NUM_VIEWER_ENVS = 1

# Larger = slower. 0.08 is around 12.5 viewer steps/second.
DEFAULT_VIEWER_DT = 0.08

TARGET_MARKER_RADIUS = 0.035
ACTUAL_MARKER_RADIUS = 0.025
REFERENCE_LINE_RADIUS = 0.006

# None disables terminal status printing.
PRINT_EVERY_N_STEPS = None


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
    make_env, elbow_ppo_runner_cfg = import_mjlab_tools(quiet=quiet)

    from rsl_rl.runners import OnPolicyRunner

    env = make_wrapped_env(
        make_env=make_env,
        env_id=env_id,
        device=device,
        num_envs=NUM_VIEWER_ENVS,
        seed=seed,
        quiet=quiet,
    )

    train_cfg = elbow_ppo_runner_cfg()

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


def reset_episode(
    env,
    seed: int,
    fixed_target_angle: float | None,
):
    obs = env.reset(seed=seed)

    if fixed_target_angle is not None:
        env.target_angles[:] = float(fixed_target_angle)
        obs = env.get_observations()

    return obs


def obs_to_values(obs) -> tuple[float, float, float, float]:
    policy_obs = obs["policy"][0].detach().cpu().numpy()

    current_angle = float(policy_obs[0])
    current_velocity = float(policy_obs[1])
    pose_error = float(policy_obs[2])
    target_angle = current_angle + pose_error

    return current_angle, current_velocity, target_angle, pose_error


def name_or_empty(model: mujoco.MjModel, obj_type, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)

    if name is None:
        return ""

    return name


def choose_elbow_qpos_index(
    model: mujoco.MjModel,
    requested_joint_name: str | None,
) -> int:
    if requested_joint_name is not None:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            requested_joint_name,
        )

        if joint_id < 0:
            raise ValueError(f"Joint not found: {requested_joint_name}")

        return int(model.jnt_qposadr[joint_id])

    preferred_words = [
        "elbow",
        "flex",
    ]

    for joint_id in range(model.njnt):
        name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        lower = name.lower()

        if any(word in lower for word in preferred_words):
            return int(model.jnt_qposadr[joint_id])

    return 0


def choose_endpoint_site_id(model: mujoco.MjModel) -> int | None:
    preferred_words = [
        "hand",
        "wrist",
        "distal",
        "end",
        "tip",
        "marker",
    ]

    forbidden_words = [
        "target",
        "goal",
    ]

    for site_id in range(model.nsite):
        name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )

        lower = name.lower()

        if any(word in lower for word in forbidden_words):
            continue

        if any(word in lower for word in preferred_words):
            return site_id

    return None


def choose_endpoint_geom_id(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> int | None:
    root_pos = np.array(data.xpos[0], dtype=np.float64)

    best_geom_id = None
    best_distance = -np.inf

    forbidden_words = [
        "floor",
        "ground",
        "target",
        "goal",
    ]

    for geom_id in range(model.ngeom):
        name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        lower = name.lower()

        if any(word in lower for word in forbidden_words):
            continue

        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        geom_pos = np.array(data.geom_xpos[geom_id], dtype=np.float64)
        distance = float(np.linalg.norm(geom_pos - root_pos))

        if distance > best_distance:
            best_distance = distance
            best_geom_id = geom_id

    return best_geom_id


def get_endpoint_position(
    data: mujoco.MjData,
    site_id: int | None,
    geom_id: int | None,
) -> np.ndarray:
    if site_id is not None:
        return np.array(data.site_xpos[site_id], dtype=np.float64)

    if geom_id is not None:
        return np.array(data.geom_xpos[geom_id], dtype=np.float64)

    return np.array(data.xpos[0], dtype=np.float64)


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_index: int,
    elbow_angle: float,
    elbow_velocity: float,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    data.qpos[qpos_index] = elbow_angle

    if qpos_index < model.nv:
        data.qvel[qpos_index] = elbow_velocity

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


def update_viewer_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_data: mujoco.MjData,
    qpos_index: int,
    current_angle: float,
    current_velocity: float,
    target_angle: float,
    site_id: int | None,
    geom_id: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    set_pose(
        model=model,
        data=data,
        qpos_index=qpos_index,
        elbow_angle=current_angle,
        elbow_velocity=current_velocity,
    )

    set_pose(
        model=model,
        data=target_data,
        qpos_index=qpos_index,
        elbow_angle=target_angle,
        elbow_velocity=0.0,
    )

    actual_pos = get_endpoint_position(
        data=data,
        site_id=site_id,
        geom_id=geom_id,
    )

    target_pos = get_endpoint_position(
        data=target_data,
        site_id=site_id,
        geom_id=geom_id,
    )

    distance = float(np.linalg.norm(target_pos - actual_pos))

    return actual_pos, target_pos, distance


def print_status(
    step: int,
    current_angle: float,
    target_angle: float,
    pose_error: float,
    endpoint_distance: float,
    reward: float,
) -> None:
    print(
        f"step={step:05d} "
        f"current={current_angle:+.4f} rad "
        f"target={target_angle:+.4f} rad "
        f"angle_error={pose_error:+.4f} rad "
        f"endpoint_dist={endpoint_distance:.5f} m "
        f"reward={reward:+.4f}"
    )


def configure_viewer(viewer) -> None:
    """
    Hide MuJoCo labels that can show as garbage text on custom user geoms.
    """
    try:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
    except Exception:
        pass


def main(args: argparse.Namespace) -> None:
    policy_path = resolve_repo_path(args.policy_path)

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
    target_data = mujoco.MjData(model)

    qpos_index = choose_elbow_qpos_index(
        model=model,
        requested_joint_name=args.joint_name,
    )

    obs = reset_episode(
        env=env,
        seed=args.seed,
        fixed_target_angle=args.target_angle,
    )

    current_angle, current_velocity, target_angle, pose_error = obs_to_values(obs)

    set_pose(
        model=model,
        data=data,
        qpos_index=qpos_index,
        elbow_angle=current_angle,
        elbow_velocity=current_velocity,
    )

    site_id = choose_endpoint_site_id(model)
    geom_id = None

    if site_id is None:
        geom_id = choose_endpoint_geom_id(model, data)

    if site_id is not None:
        marker_name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        )
        print(f"Endpoint marker source: site '{marker_name}'")
    elif geom_id is not None:
        marker_name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        print(f"Endpoint marker source: geom '{marker_name}'")
    else:
        print("Endpoint marker source: root body position fallback")

    print("=" * 80)
    print("MJLAB ELBOW POLICY VISUALIZER")
    print("=" * 80)
    print(f"policy:      {policy_path}")
    print(f"env:         {args.env_id}")
    print(f"device:      {args.device}")
    print(f"qpos index:  {qpos_index}")
    print(f"viewer dt:   {args.dt}")
    print("green marker = endpoint position at target elbow angle")
    print("red marker   = current endpoint position")
    print("yellow line  = distance between target and current endpoint")
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

            current_angle, current_velocity, target_angle, pose_error = obs_to_values(obs)

            actual_pos, target_pos, endpoint_distance = update_viewer_state(
                model=model,
                data=data,
                target_data=target_data,
                qpos_index=qpos_index,
                current_angle=current_angle,
                current_velocity=current_velocity,
                target_angle=target_angle,
                site_id=site_id,
                geom_id=geom_id,
            )

            viewer.user_scn.ngeom = 0

            add_sphere_marker(
                viewer=viewer,
                position=target_pos,
                radius=TARGET_MARKER_RADIUS,
                rgba=[0.0, 1.0, 0.0, 0.70],
            )

            add_sphere_marker(
                viewer=viewer,
                position=actual_pos,
                radius=ACTUAL_MARKER_RADIUS,
                rgba=[1.0, 0.0, 0.0, 0.80],
            )

            add_reference_line(
                viewer=viewer,
                start_pos=actual_pos,
                end_pos=target_pos,
                radius=REFERENCE_LINE_RADIUS,
                rgba=[1.0, 1.0, 0.0, 0.65],
            )

            viewer.sync()

            reward_scalar = float(reward.detach().mean().cpu())

            if args.print_every is not None and viewer_step % args.print_every == 0:
                print_status(
                    step=viewer_step,
                    current_angle=current_angle,
                    target_angle=target_angle,
                    pose_error=pose_error,
                    endpoint_distance=endpoint_distance,
                    reward=reward_scalar,
                )

            done_now = bool(done.detach().cpu().reshape(-1)[0].item())

            if args.reset_on_done and done_now:
                episode_index += 1

                obs = reset_episode(
                    env=env,
                    seed=args.seed + episode_index,
                    fixed_target_angle=args.target_angle,
                )

            viewer_step += 1

            sleep_time = args.dt - (time.time() - step_start)

            if sleep_time > 0.0:
                time.sleep(sleep_time)

    env.close()


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
        "--target-angle",
        type=float,
        default=None,
        help="Optional fixed target angle in radians.",
    )

    parser.add_argument(
        "--joint-name",
        type=str,
        default=None,
        help="Optional MuJoCo joint name. If omitted, the script searches for elbow/flex.",
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
        help="Seconds between viewer updates. Larger is slower.",
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


if __name__ == "__main__":
    parsed_args = parse_args()
    validate_args(parsed_args)
    main(parsed_args)