#!/usr/bin/env python3
"""Visualize a trained static-target MJLab Baoding PPO-LSTM policy.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/visualize_baoding_LSTM.py

Run from the repository root:

    python scripts/mjlab/BaodingBalls/visualize_baoding_LSTM.py \
        --policy-path my_baoding_static_lstm_policy.pt

The MJLab environment performs the physics and policy rollout. A separate
classic MuJoCo model is used only for interactive rendering. Its hand joints
and both ball positions are updated from the exact 97-value policy observation.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Import train_baoding_LSTM first. It applies the experiment-facing ball
# settings before the derived registration constants below are read.
from train_baoding_LSTM import (
    BALL_SIZE_SCALE,
    REPO_ROOT,
    TASK_ID,
    apply_training_config,
    configure_lstm_models,
    get_deterministic_action,
    import_mjlab_tools,
    make_wrapped_env,
    redirect_output,
    validate_lstm_runner,
)
from register_baoding import (
    BALL1_ENTITY_NAME,
    BALL1_INITIAL_POS,
    BALL2_ENTITY_NAME,
    BALL2_INITIAL_POS,
    CTRL_DT,
    DROP_HEIGHT,
    EXPECTED_ACTION_DIM,
    EXPECTED_HAND_QPOS_DIM,
    EXPECTED_HAND_QVEL_DIM,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    hand_joint_names,
    reference_baoding_xml_path,
)


DEFAULT_POLICY_PATH = "my_baoding_static_lstm_policy.pt"
DEFAULT_ENV_ID = TASK_ID

NUM_VIEWER_ENVS = 1

# Match real simulated time by default.
DEFAULT_VIEWER_DT = CTRL_DT

# None disables periodic status printing.
PRINT_EVERY_N_STEPS = 20

# Visual-only marker for the drop threshold.
DROP_PLANE_HALF_SIZE = 0.13
DROP_PLANE_THICKNESS = 0.002


# Observation layout defined by register_baoding.py.
HAND_QPOS_SLICE = slice(0, 23)
HAND_QVEL_SLICE = slice(23, 46)
BALL1_POS_SLICE = slice(46, 49)
BALL1_LIN_VEL_SLICE = slice(49, 52)
BALL2_POS_SLICE = slice(52, 55)
BALL2_LIN_VEL_SLICE = slice(55, 58)
ACTIVATION_SLICE = slice(58, 97)


# =============================================================================
# POLICY / RUNNER HELPERS
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
        normalize_rewards=False,
    )

    train_cfg = load_rl_cfg(env_id)

    train_cfg.obs_groups = {
        "policy": ("policy",),
        "critic": ("policy",),
        "actor": ("policy",),
    }

    train_cfg_dict = asdict(train_cfg)

    apply_training_config(
        train_cfg_dict
    )
    configure_lstm_models(
        train_cfg_dict
    )

    with redirect_output(quiet):
        runner = OnPolicyRunner(
            env=env,
            train_cfg=train_cfg_dict,
            log_dir=None,
            device=device,
        )

    validate_lstm_runner(runner)

    return env, runner


def load_policy_into_runner(
    runner,
    policy_path: Path,
) -> None:
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Policy file not found: "
            f"{policy_path}"
        )

    try:
        runner.load(
            str(policy_path),
            load_optimizer=False,
        )
    except TypeError:
        # RSL-RL 5.x versions without
        # the load_optimizer argument.
        runner.load(
            str(policy_path)
        )


# =============================================================================
# OBSERVATION HELPERS
# =============================================================================

@dataclass(frozen=True)
class BaodingFrame:
    hand_qpos: np.ndarray
    hand_qvel: np.ndarray
    ball1_pos: np.ndarray
    ball1_lin_vel: np.ndarray
    ball2_pos: np.ndarray
    ball2_lin_vel: np.ndarray
    activation: np.ndarray

    @property
    def minimum_ball_height(
        self,
    ) -> float:
        return min(
            float(self.ball1_pos[2]),
            float(self.ball2_pos[2]),
        )

    @property
    def both_balls_above_drop_height(
        self,
    ) -> bool:
        return (
            self.minimum_ball_height
            >= DROP_HEIGHT
        )


def observation_to_frame(
    obs,
) -> BaodingFrame:
    policy_obs = obs["policy"]

    expected_shape = (
        NUM_VIEWER_ENVS,
        OBS_DIM,
    )

    if (
        tuple(policy_obs.shape)
        != expected_shape
    ):
        raise RuntimeError(
            f"Expected policy observation "
            f"shape {expected_shape}, got "
            f"{tuple(policy_obs.shape)}."
        )

    values = (
        policy_obs[0]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=False,
        )
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "The Baoding policy observation "
            "contains NaN or infinity."
        )

    return BaodingFrame(
        hand_qpos=np.array(
            values[HAND_QPOS_SLICE],
            dtype=np.float64,
            copy=True,
        ),
        hand_qvel=np.array(
            values[HAND_QVEL_SLICE],
            dtype=np.float64,
            copy=True,
        ),
        ball1_pos=np.array(
            values[BALL1_POS_SLICE],
            dtype=np.float64,
            copy=True,
        ),
        ball1_lin_vel=np.array(
            values[
                BALL1_LIN_VEL_SLICE
            ],
            dtype=np.float64,
            copy=True,
        ),
        ball2_pos=np.array(
            values[BALL2_POS_SLICE],
            dtype=np.float64,
            copy=True,
        ),
        ball2_lin_vel=np.array(
            values[
                BALL2_LIN_VEL_SLICE
            ],
            dtype=np.float64,
            copy=True,
        ),
        activation=np.array(
            values[ACTIVATION_SLICE],
            dtype=np.float64,
            copy=True,
        ),
    )


def tensor_first_bool(
    value,
) -> bool:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return bool(
            value.detach()
            .reshape(-1)[0]
            .cpu()
            .item()
        )

    return bool(value)


def tensor_mean_float(
    value,
) -> float:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return float(
            value.detach()
            .float()
            .mean()
            .cpu()
            .item()
        )

    return float(value)


# =============================================================================
# CLASSIC MUJOCO RENDER MODEL
# =============================================================================

@dataclass(frozen=True)
class ViewerStateMap:
    hand_qpos_addresses: tuple[
        int,
        ...,
    ]
    hand_dof_addresses: tuple[
        int,
        ...,
    ]
    ball1_qpos_address: int
    ball1_dof_address: int
    ball2_qpos_address: int
    ball2_dof_address: int


def name_or_empty(
    model: mujoco.MjModel,
    obj_type,
    obj_id: int,
) -> str:
    name = mujoco.mj_id2name(
        model,
        obj_type,
        obj_id,
    )

    if name is None:
        return ""

    return name


def find_scalar_joint_id(
    model: mujoco.MjModel,
    joint_name: str,
) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Could not find hand joint "
            f"{joint_name!r} in the "
            "Baoding render model."
        )

    joint_type = int(
        model.jnt_type[joint_id]
    )

    scalar_types = {
        int(
            mujoco.mjtJoint.mjJNT_HINGE
        ),
        int(
            mujoco.mjtJoint.mjJNT_SLIDE
        ),
    }

    if joint_type not in scalar_types:
        raise RuntimeError(
            f"Hand joint {joint_name!r} "
            "is not a scalar hinge or "
            "slide joint."
        )

    return int(joint_id)


def find_ball_free_joint_id(
    model: mujoco.MjModel,
    ball_body_name: str,
) -> int:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ball_body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Could not find ball body "
            f"{ball_body_name!r} in the "
            "Baoding render model."
        )

    matches: list[int] = []

    for joint_id in range(
        model.njnt
    ):
        if (
            int(
                model.jnt_bodyid[
                    joint_id
                ]
            )
            != int(body_id)
        ):
            continue

        if (
            int(
                model.jnt_type[
                    joint_id
                ]
            )
            == int(
                mujoco.mjtJoint
                .mjJNT_FREE
            )
        ):
            matches.append(
                joint_id
            )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one free "
            f"joint on body "
            f"{ball_body_name!r}, but "
            f"found {len(matches)}."
        )

    return int(matches[0])


def build_viewer_state_map(
    model: mujoco.MjModel,
) -> ViewerStateMap:
    source_joint_names = (
        hand_joint_names()
    )

    if (
        len(source_joint_names)
        != EXPECTED_HAND_QPOS_DIM
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_HAND_QPOS_DIM} "
            "hand joints, but "
            "register_baoding returned "
            f"{len(source_joint_names)}."
        )

    hand_joint_ids = tuple(
        find_scalar_joint_id(
            model,
            name,
        )
        for name in source_joint_names
    )

    ball1_joint_id = (
        find_ball_free_joint_id(
            model,
            BALL1_ENTITY_NAME,
        )
    )
    ball2_joint_id = (
        find_ball_free_joint_id(
            model,
            BALL2_ENTITY_NAME,
        )
    )

    return ViewerStateMap(
        hand_qpos_addresses=tuple(
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
            for joint_id
            in hand_joint_ids
        ),
        hand_dof_addresses=tuple(
            int(
                model.jnt_dofadr[
                    joint_id
                ]
            )
            for joint_id
            in hand_joint_ids
        ),
        ball1_qpos_address=int(
            model.jnt_qposadr[
                ball1_joint_id
            ]
        ),
        ball1_dof_address=int(
            model.jnt_dofadr[
                ball1_joint_id
            ]
        ),
        ball2_qpos_address=int(
            model.jnt_qposadr[
                ball2_joint_id
            ]
        ),
        ball2_dof_address=int(
            model.jnt_dofadr[
                ball2_joint_id
            ]
        ),
    )


def hide_unused_target_markers(
    model: mujoco.MjModel,
) -> None:
    """Hide classic rotating targets."""
    for site_id in range(
        model.nsite
    ):
        name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
        ).lower()

        if (
            "target" in name
            or "goal" in name
        ):
            model.site_rgba[
                site_id,
                3,
            ] = 0.0

    for geom_id in range(
        model.ngeom
    ):
        name = name_or_empty(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        ).lower()

        if (
            "target" in name
            or "goal" in name
        ):
            model.geom_rgba[
                geom_id,
                3,
            ] = 0.0


def resize_render_balls(
    model: mujoco.MjModel,
) -> None:
    """Match the rendered balls to MJLab."""
    for ball_name in (
        BALL1_ENTITY_NAME,
        BALL2_ENTITY_NAME,
    ):
        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            ball_name,
        )
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"{ball_name}_site",
        )

        if (
            geom_id < 0
            or site_id < 0
        ):
            raise RuntimeError(
                "Could not resize render "
                f"geometry for "
                f"{ball_name!r}."
            )

        model.geom_size[
            geom_id,
            0,
        ] *= BALL_SIZE_SCALE

        model.site_size[
            site_id,
            0,
        ] *= BALL_SIZE_SCALE


def set_free_joint_state(
    data: mujoco.MjData,
    qpos_address: int,
    dof_address: int,
    position: np.ndarray,
    scaled_linear_velocity: (
        np.ndarray
    ),
) -> None:
    data.qpos[
        qpos_address:
        qpos_address + 3
    ] = position

    # Ball orientation is absent from
    # the policy observation.
    quaternion = data.qpos[
        qpos_address + 3:
        qpos_address + 7
    ]

    quaternion_norm = float(
        np.linalg.norm(quaternion)
    )

    if quaternion_norm < 1e-12:
        data.qpos[
            qpos_address + 3:
            qpos_address + 7
        ] = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
    else:
        data.qpos[
            qpos_address + 3:
            qpos_address + 7
        ] = (
            quaternion
            / quaternion_norm
        )

    # Velocity observations are scaled
    # by CTRL_DT in register_baoding.
    data.qvel[
        dof_address:
        dof_address + 3
    ] = (
        scaled_linear_velocity
        / CTRL_DT
    )
    data.qvel[
        dof_address + 3:
        dof_address + 6
    ] = 0.0


def set_render_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state_map: ViewerStateMap,
    frame: BaodingFrame,
    action: (
        torch.Tensor
        | None
    ),
    episode_time: float,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    for (
        index,
        qpos_address,
    ) in enumerate(
        state_map
        .hand_qpos_addresses
    ):
        data.qpos[
            qpos_address
        ] = frame.hand_qpos[
            index
        ]

    for (
        index,
        dof_address,
    ) in enumerate(
        state_map
        .hand_dof_addresses
    ):
        data.qvel[
            dof_address
        ] = (
            frame.hand_qvel[index]
            / CTRL_DT
        )

    set_free_joint_state(
        data=data,
        qpos_address=(
            state_map
            .ball1_qpos_address
        ),
        dof_address=(
            state_map
            .ball1_dof_address
        ),
        position=frame.ball1_pos,
        scaled_linear_velocity=(
            frame.ball1_lin_vel
        ),
    )

    set_free_joint_state(
        data=data,
        qpos_address=(
            state_map
            .ball2_qpos_address
        ),
        dof_address=(
            state_map
            .ball2_dof_address
        ),
        position=frame.ball2_pos,
        scaled_linear_velocity=(
            frame.ball2_lin_vel
        ),
    )

    if model.na > 0:
        data.act[:] = 0.0

        activation_count = min(
            model.na,
            frame.activation.shape[0],
        )

        data.act[
            :activation_count
        ] = frame.activation[
            :activation_count
        ]

    if model.nu > 0:
        data.ctrl[:] = 0.0

        if action is not None:
            action_values = (
                action.detach()
                .reshape(
                    NUM_VIEWER_ENVS,
                    -1,
                )[0]
                .float()
                .cpu()
                .numpy()
            )

            control_count = min(
                model.nu,
                action_values.shape[0],
            )

            data.ctrl[
                :control_count
            ] = action_values[
                :control_count
            ]

    data.time = episode_time
    mujoco.mj_forward(
        model,
        data,
    )


# =============================================================================
# VIEWER HELPERS
# =============================================================================

def clear_geom_label(
    geom,
) -> None:
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


def add_drop_plane(
    viewer,
) -> None:
    scene = viewer.user_scn

    if (
        scene.ngeom
        >= scene.maxgeom
    ):
        return

    center = np.array(
        [
            0.5
            * (
                BALL1_INITIAL_POS[0]
                + BALL2_INITIAL_POS[0]
            ),
            0.5
            * (
                BALL1_INITIAL_POS[1]
                + BALL2_INITIAL_POS[1]
            ),
            DROP_HEIGHT,
        ],
        dtype=np.float64,
    )

    geom = scene.geoms[
        scene.ngeom
    ]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array(
            [
                DROP_PLANE_HALF_SIZE,
                DROP_PLANE_HALF_SIZE,
                DROP_PLANE_THICKNESS,
            ],
            dtype=np.float64,
        ),
        center,
        np.eye(
            3,
            dtype=np.float64,
        ).reshape(-1),
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.12,
            ],
            dtype=np.float32,
        ),
    )

    clear_geom_label(geom)
    scene.ngeom += 1


def configure_viewer(
    viewer,
) -> None:
    try:
        viewer.opt.label = (
            mujoco.mjtLabel
            .mjLABEL_NONE
        )
    except Exception:
        pass

    try:
        viewer.cam.lookat[:] = (
            np.array(
                [
                    0.5
                    * (
                        BALL1_INITIAL_POS[
                            0
                        ]
                        + BALL2_INITIAL_POS[
                            0
                        ]
                    ),
                    0.5
                    * (
                        BALL1_INITIAL_POS[
                            1
                        ]
                        + BALL2_INITIAL_POS[
                            1
                        ]
                    ),
                    0.5
                    * (
                        max(
                            BALL1_INITIAL_POS[
                                2
                            ],
                            BALL2_INITIAL_POS[
                                2
                            ],
                        )
                        + DROP_HEIGHT
                    ),
                ],
                dtype=np.float64,
            )
        )
        viewer.cam.distance = 0.55
    except Exception:
        pass


def get_done_flags(
    done,
    infos,
) -> tuple[
    bool,
    bool,
    bool,
]:
    done_now = tensor_first_bool(
        done
    )

    try:
        terminated = (
            tensor_first_bool(
                infos[
                    "baoding"
                ][
                    "terminated"
                ]
            )
        )
        truncated = (
            tensor_first_bool(
                infos[
                    "baoding"
                ][
                    "truncated"
                ]
            )
        )
    except Exception:
        terminated = done_now
        truncated = False

    return (
        done_now,
        terminated,
        truncated,
    )


def print_step_status(
    viewer_step: int,
    episode_index: int,
    episode_step: int,
    reward: float,
    episode_return: float,
    frame: BaodingFrame,
    terminated: bool,
    truncated: bool,
) -> None:
    if terminated:
        state_text = "DROP"
    elif truncated:
        state_text = "TIME_LIMIT"
    elif (
        frame
        .both_balls_above_drop_height
    ):
        state_text = "HOLD"
    else:
        state_text = (
            "BELOW_THRESHOLD"
        )

    ball1_speed = float(
        np.linalg.norm(
            frame.ball1_lin_vel
        )
        / CTRL_DT
    )
    ball2_speed = float(
        np.linalg.norm(
            frame.ball2_lin_vel
        )
        / CTRL_DT
    )

    print(
        f"step={viewer_step:06d} "
        f"episode={episode_index:04d} "
        f"ep_step={episode_step:03d}/"
        f"{MAX_EPISODE_STEPS} "
        f"reward={reward:+.2f} "
        f"return={episode_return:+.2f} "
        f"z1={frame.ball1_pos[2]:.3f} "
        f"z2={frame.ball2_pos[2]:.3f} "
        f"speed1={ball1_speed:.3f} "
        f"speed2={ball2_speed:.3f} "
        f"state={state_text}"
    )


def print_episode_summary(
    episode_index: int,
    episode_steps: int,
    episode_return: float,
    terminated: bool,
    truncated: bool,
) -> None:
    if terminated:
        result = "BALL DROPPED"
    elif truncated:
        result = (
            f"FULL "
            f"{MAX_EPISODE_STEPS}-"
            "STEP HOLD"
        )
    else:
        result = "DONE"

    print(
        f"Episode {episode_index} "
        f"finished: {result}; "
        f"steps={episode_steps}; "
        f"sim_time="
        f"{episode_steps * CTRL_DT:.2f}s; "
        f"return={episode_return:.2f}"
    )


# =============================================================================
# MAIN VISUALIZER
# =============================================================================

def main(
    args: argparse.Namespace,
) -> None:
    policy_path = (
        resolve_repo_path(
            args.policy_path
        )
    )

    env = None

    try:
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

        policy = (
            runner
            .get_inference_policy(
                device=args.device
            )
        )

        model = (
            mujoco.MjModel
            .from_xml_path(
                str(
                    reference_baoding_xml_path()
                )
            )
        )

        resize_render_balls(model)
        data = mujoco.MjData(model)

        hide_unused_target_markers(
            model
        )
        state_map = (
            build_viewer_state_map(
                model
            )
        )

        obs = env.reset(
            seed=args.seed
        )
        policy.reset()

        frame = observation_to_frame(
            obs
        )

        if (
            frame.hand_qpos.shape[0]
            != EXPECTED_HAND_QPOS_DIM
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_HAND_QPOS_DIM} "
                "hand qpos values, got "
                f"{frame.hand_qpos.shape[0]}."
            )

        if (
            frame.hand_qvel.shape[0]
            != EXPECTED_HAND_QVEL_DIM
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_HAND_QVEL_DIM} "
                "hand qvel values, got "
                f"{frame.hand_qvel.shape[0]}."
            )

        if (
            frame.activation.shape[0]
            != EXPECTED_ACTION_DIM
        ):
            raise RuntimeError(
                f"Expected "
                f"{EXPECTED_ACTION_DIM} "
                "muscle activations, got "
                f"{frame.activation.shape[0]}."
            )

        set_render_state(
            model=model,
            data=data,
            state_map=state_map,
            frame=frame,
            action=None,
            episode_time=0.0,
        )

        print("=" * 80)
        print(
            "MJLAB BAODING STATIC "
            "PPO-LSTM POLICY VISUALIZER"
        )
        print("=" * 80)
        print(
            f"policy:              "
            f"{policy_path}"
        )
        print(
            f"environment:         "
            f"{args.env_id}"
        )
        print(
            f"device:              "
            f"{args.device}"
        )
        print(
            f"viewer delay:        "
            f"{args.dt:.4f} s/step"
        )
        print(
            f"control timestep:    "
            f"{CTRL_DT:.4f} s/step"
        )
        print(
            f"episode horizon:     "
            f"{MAX_EPISODE_STEPS} steps"
        )
        print(
            f"drop height:         "
            f"{DROP_HEIGHT:.3f} m"
        )
        print(
            f"ball size scale:     "
            f"{BALL_SIZE_SCALE:.2f}"
        )
        print(
            f"show drop plane:     "
            f"{args.show_drop_plane}"
        )
        print(
            "red translucent box = "
            "drop-height threshold"
        )
        print(
            "MJLab automatically resets "
            "after a drop or time limit. "
            "The corresponding LSTM state "
            "is also reset."
        )
        print("=" * 80)

        viewer_step = 0
        episode_index = 1
        episode_step = 0
        episode_return = 0.0
        completed_episodes = 0

        with (
            mujoco.viewer
            .launch_passive(
                model,
                data,
            )
        ) as viewer:
            configure_viewer(viewer)

            while viewer.is_running():
                step_start = time.time()

                action = (
                    get_deterministic_action(
                        policy=policy,
                        obs=obs,
                    )
                )

                (
                    obs,
                    reward,
                    done,
                    infos,
                ) = env.step(action)

                # The environment automatically
                # resets finished episodes.
                policy.reset(done)

                reward_scalar = (
                    tensor_mean_float(
                        reward
                    )
                )

                (
                    done_now,
                    terminated,
                    truncated,
                ) = get_done_flags(
                    done=done,
                    infos=infos,
                )

                episode_step += 1
                episode_return += (
                    reward_scalar
                )
                viewer_step += 1

                # On terminal steps, obs is
                # already the next reset state.
                frame = (
                    observation_to_frame(
                        obs
                    )
                )

                render_episode_time = (
                    0.0
                    if done_now
                    else (
                        episode_step
                        * CTRL_DT
                    )
                )

                set_render_state(
                    model=model,
                    data=data,
                    state_map=state_map,
                    frame=frame,
                    action=action,
                    episode_time=(
                        render_episode_time
                    ),
                )

                viewer.user_scn.ngeom = 0

                if args.show_drop_plane:
                    add_drop_plane(viewer)

                viewer.sync()

                should_print = (
                    done_now
                    or (
                        args.print_every
                        is not None
                        and viewer_step
                        % args.print_every
                        == 0
                    )
                )

                if should_print:
                    print_step_status(
                        viewer_step=(
                            viewer_step
                        ),
                        episode_index=(
                            episode_index
                        ),
                        episode_step=(
                            episode_step
                        ),
                        reward=reward_scalar,
                        episode_return=(
                            episode_return
                        ),
                        frame=frame,
                        terminated=(
                            terminated
                        ),
                        truncated=(
                            truncated
                        ),
                    )

                if done_now:
                    print_episode_summary(
                        episode_index=(
                            episode_index
                        ),
                        episode_steps=(
                            episode_step
                        ),
                        episode_return=(
                            episode_return
                        ),
                        terminated=(
                            terminated
                        ),
                        truncated=(
                            truncated
                        ),
                    )

                    completed_episodes += 1

                    if (
                        args.max_episodes
                        is not None
                        and completed_episodes
                        >= args.max_episodes
                    ):
                        break

                    episode_index += 1
                    episode_step = 0
                    episode_return = 0.0

                sleep_time = (
                    args.dt
                    - (
                        time.time()
                        - step_start
                    )
                )

                if sleep_time > 0.0:
                    time.sleep(
                        sleep_time
                    )

    finally:
        if env is not None:
            env.close()


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> (
    argparse.Namespace
):
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
            "Wall-clock seconds between "
            "viewer updates. This does "
            "not change physics dt."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=(
            PRINT_EVERY_N_STEPS
        ),
        help=(
            "Print status every N viewer "
            "steps. Use 0 to print every "
            "step."
        ),
    )

    parser.add_argument(
        "--show-drop-plane",
        action=(
            argparse
            .BooleanOptionalAction
        ),
        default=True,
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Optional number of completed "
            "episodes before closing."
        ),
    )

    parser.add_argument(
        "--quiet",
        action=(
            argparse
            .BooleanOptionalAction
        ),
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
        args.device.startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested device "
            f"{args.device}, but CUDA "
            "is not available."
        )

    if args.dt <= 0.0:
        raise ValueError(
            "--dt must be greater "
            "than zero."
        )

    if (
        args.max_episodes is not None
        and args.max_episodes <= 0
    ):
        raise ValueError(
            "--max-episodes must be "
            "greater than zero."
        )

    if NUM_VIEWER_ENVS != 1:
        raise ValueError(
            "The interactive viewer "
            "requires exactly one "
            "environment."
        )

    if OBS_DIM != 97:
        raise ValueError(
            "OBS_DIM must be 97 for "
            "this Baoding observation "
            "layout."
        )

    if CTRL_DT <= 0.0:
        raise ValueError(
            "CTRL_DT must be positive."
        )


if __name__ == "__main__":
    parsed_args = parse_args()
    validate_args(parsed_args)
    main(parsed_args)