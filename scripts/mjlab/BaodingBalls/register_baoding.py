"""Register the survival-only MJLab BaodingBalls task.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/register_baoding.py

Run the registration check from the repository root with:

    python scripts/mjlab/BaodingBalls/register_baoding.py

Training and visualization scripts placed in the same BaodingBalls directory
can import this module directly, for example:

    from register_baoding import TASK_ID, register_baoding_hold_task
"""

from __future__ import annotations

import contextlib
import copy
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


# =============================================================================
# TASK CONFIGURATION
# =============================================================================

TASK_ID = "myoHandBaodingHold-v0"

NUM_ENVS = 64
PLAY_NUM_ENVS = 1

MAX_EPISODE_STEPS = 400

SIM_DT = 0.002
CTRL_DT = 0.02

# The original MyoSuite Baoding task uses 1.25 m as its drop threshold.
DROP_HEIGHT = 1.25

# Survival-only reward:
#
#   +1 for every step during which both balls remain above DROP_HEIGHT
#   -10 on the step on which either ball drops
#
# There is deliberately no target-distance, rotation, velocity, activation,
# action, or energy term in this first-stage task.
HOLD_REWARD_PER_STEP = 1.0
DROP_PENALTY = 10.0

BALL1_CLASSIC_INITIAL_POS = (
    -0.227,
    -0.511,
    1.452,
)
BALL2_CLASSIC_INITIAL_POS = (
    -0.256,
    -0.552,
    1.442,
)

# The original balls have a nominal radius of 0.022 m (44 mm diameter).
# Scaling them to 86% produces a diameter of approximately 38 mm.
#
# train_baoding.py overrides this value before task registration.
BALL_SIZE_SCALE = 0.86

# Lower each center by the removed radius so that the smaller sphere keeps
# approximately the same bottom contact height against the upward-facing palm.
ORIGINAL_BALL_NOMINAL_RADIUS = 0.022

BALL_CONTACT_HEIGHT_COMPENSATION = (
    ORIGINAL_BALL_NOMINAL_RADIUS
    * (1.0 - BALL_SIZE_SCALE)
)

BALL1_INITIAL_POS = (
    BALL1_CLASSIC_INITIAL_POS[0],
    BALL1_CLASSIC_INITIAL_POS[1],
    BALL1_CLASSIC_INITIAL_POS[2]
    - BALL_CONTACT_HEIGHT_COMPENSATION,
)

BALL2_INITIAL_POS = (
    BALL2_CLASSIC_INITIAL_POS[0],
    BALL2_CLASSIC_INITIAL_POS[1],
    BALL2_CLASSIC_INITIAL_POS[2]
    - BALL_CONTACT_HEIGHT_COMPENSATION,
)

# Preserve the original material density when the sphere size changes.
# Sphere mass scales with the cube of its radius.
BALL_MASS_SCALE = BALL_SIZE_SCALE ** 3

# Reset randomization is deliberately horizontal. Randomizing height can place
# a ball above the palm with an air gap or partially inside the hand, producing
# a fall or a large contact impulse before the policy can react.
#
# Both balls first receive the same pair translation, after which each ball
# receives a smaller independent jitter.
#
# train_baoding.py overrides these values before task registration.
BALL_PAIR_RESET_OFFSET_LOW = (
    -0.003,
    -0.003,
    0.0,
)
BALL_PAIR_RESET_OFFSET_HIGH = (
    0.003,
    0.003,
    0.0,
)

BALL_INDIVIDUAL_RESET_JITTER_LOW = (
    -0.001,
    -0.001,
    0.0,
)
BALL_INDIVIDUAL_RESET_JITTER_HIGH = (
    0.001,
    0.001,
    0.0,
)

# Enforce a small free-space margin between the two sphere surfaces.
BALL_SEPARATION_MARGIN = 0.002

# Enforce this center-height clearance above the termination threshold.
BALL_RESET_DROP_CLEARANCE = 0.10

PALM_UP_JOINT_POSITION = -1.57

HAND_ENTITY_NAME = "hand"
BALL1_ENTITY_NAME = "ball1"
BALL2_ENTITY_NAME = "ball2"

EXPECTED_HAND_QPOS_DIM = 23
EXPECTED_HAND_QVEL_DIM = 23
EXPECTED_ACTION_DIM = 39

# Observation order:
#
#   hand_qpos       0:23    23 values
#   hand_qvel      23:46    23 values
#   ball1_pos      46:49     3 values
#   ball1_lin_vel  49:52     3 values
#   ball2_pos      52:55     3 values
#   ball2_lin_vel  55:58     3 values
#   activation     58:97    39 values
OBS_DIM = 97

# Keep this object alive for as long as MuJoCo may need the temporary assets.
_HAND_ASSET_VIEW_TMPDIR: (
    tempfile.TemporaryDirectory[str] | None
) = None

# Once an environment configuration has been placed in MJLab's process-local
# registry, changing these values would not update that registered object.
_TASK_CONFIGURATION_LOCKED = False


@dataclass(frozen=True)
class BaodingHoldTaskSettings:
    max_episode_steps: int
    sim_dt: float
    ctrl_dt: float
    drop_height: float
    num_envs: int


def make_baoding_hold_task_settings(
    num_envs: int,
) -> BaodingHoldTaskSettings:
    return BaodingHoldTaskSettings(
        max_episode_steps=MAX_EPISODE_STEPS,
        sim_dt=SIM_DT,
        ctrl_dt=CTRL_DT,
        drop_height=DROP_HEIGHT,
        num_envs=num_envs,
    )


# =============================================================================
# EXTERNAL CONFIGURATION
# =============================================================================

def configure_baoding_balls(
    *,
    ball_size_scale: float,
    pair_reset_offset_low: tuple[
        float,
        float,
        float,
    ],
    pair_reset_offset_high: tuple[
        float,
        float,
        float,
    ],
    individual_reset_jitter_low: tuple[
        float,
        float,
        float,
    ],
    individual_reset_jitter_high: tuple[
        float,
        float,
        float,
    ],
) -> None:
    """Apply ball settings before the MJLab task is registered.

    train_baoding.py owns the experiment-facing values. This registration
    module calculates the resulting radius, mass, contact height and reset
    states.
    """
    global BALL_SIZE_SCALE
    global BALL_CONTACT_HEIGHT_COMPENSATION
    global BALL1_INITIAL_POS
    global BALL2_INITIAL_POS
    global BALL_MASS_SCALE
    global BALL_PAIR_RESET_OFFSET_LOW
    global BALL_PAIR_RESET_OFFSET_HIGH
    global BALL_INDIVIDUAL_RESET_JITTER_LOW
    global BALL_INDIVIDUAL_RESET_JITTER_HIGH

    requested_size_scale = float(
        ball_size_scale
    )

    requested_pair_low = tuple(
        float(value)
        for value in pair_reset_offset_low
    )
    requested_pair_high = tuple(
        float(value)
        for value in pair_reset_offset_high
    )

    requested_jitter_low = tuple(
        float(value)
        for value in individual_reset_jitter_low
    )
    requested_jitter_high = tuple(
        float(value)
        for value in individual_reset_jitter_high
    )

    requested_configuration = (
        requested_size_scale,
        requested_pair_low,
        requested_pair_high,
        requested_jitter_low,
        requested_jitter_high,
    )

    current_configuration = (
        BALL_SIZE_SCALE,
        BALL_PAIR_RESET_OFFSET_LOW,
        BALL_PAIR_RESET_OFFSET_HIGH,
        BALL_INDIVIDUAL_RESET_JITTER_LOW,
        BALL_INDIVIDUAL_RESET_JITTER_HIGH,
    )

    if _TASK_CONFIGURATION_LOCKED:
        if (
            requested_configuration
            == current_configuration
        ):
            return

        raise RuntimeError(
            "The Baoding task is already registered "
            "in this process. Configure the ball "
            "settings before calling "
            "register_baoding_hold_task()."
        )

    if not (
        0.0
        < requested_size_scale
        <= 1.0
    ):
        raise ValueError(
            "ball_size_scale must be in (0, 1]."
        )

    ranges = (
        (
            "pair_reset_offset",
            requested_pair_low,
            requested_pair_high,
        ),
        (
            "individual_reset_jitter",
            requested_jitter_low,
            requested_jitter_high,
        ),
    )

    for (
        name,
        low,
        high,
    ) in ranges:
        if (
            len(low) != 3
            or len(high) != 3
        ):
            raise ValueError(
                f"{name} bounds must contain "
                "exactly three values."
            )

        if any(
            low_value > high_value
            for (
                low_value,
                high_value,
            ) in zip(
                low,
                high,
                strict=True,
            )
        ):
            raise ValueError(
                f"Every lower {name} bound "
                "must be less than or equal "
                "to its upper bound."
            )

    vertical_bounds = (
        requested_pair_low[2],
        requested_pair_high[2],
        requested_jitter_low[2],
        requested_jitter_high[2],
    )

    if any(
        abs(value) > 1e-12
        for value in vertical_bounds
    ):
        raise ValueError(
            "The Baoding reset z offsets must "
            "remain 0.0. Randomizing height can "
            "create an air gap or an initial "
            "contact impulse."
        )

    BALL_SIZE_SCALE = (
        requested_size_scale
    )

    BALL_CONTACT_HEIGHT_COMPENSATION = (
        ORIGINAL_BALL_NOMINAL_RADIUS
        * (1.0 - BALL_SIZE_SCALE)
    )

    BALL1_INITIAL_POS = (
        BALL1_CLASSIC_INITIAL_POS[0],
        BALL1_CLASSIC_INITIAL_POS[1],
        (
            BALL1_CLASSIC_INITIAL_POS[2]
            - BALL_CONTACT_HEIGHT_COMPENSATION
        ),
    )

    BALL2_INITIAL_POS = (
        BALL2_CLASSIC_INITIAL_POS[0],
        BALL2_CLASSIC_INITIAL_POS[1],
        (
            BALL2_CLASSIC_INITIAL_POS[2]
            - BALL_CONTACT_HEIGHT_COMPENSATION
        ),
    )

    BALL_MASS_SCALE = (
        BALL_SIZE_SCALE ** 3
    )

    BALL_PAIR_RESET_OFFSET_LOW = (
        requested_pair_low
    )
    BALL_PAIR_RESET_OFFSET_HIGH = (
        requested_pair_high
    )

    BALL_INDIVIDUAL_RESET_JITTER_LOW = (
        requested_jitter_low
    )
    BALL_INDIVIDUAL_RESET_JITTER_HIGH = (
        requested_jitter_high
    )


# =============================================================================
# OUTPUT REDIRECTION
# =============================================================================

@contextlib.contextmanager
def silence_output(
    enabled: bool = True,
):
    if not enabled:
        yield
        return

    with open(
        os.devnull,
        "w",
        encoding="utf-8",
    ) as file:
        with contextlib.redirect_stdout(
            file
        ):
            with contextlib.redirect_stderr(
                file
            ):
                yield


# =============================================================================
# MYOSUITE HAND ASSETS
# =============================================================================

@lru_cache(maxsize=1)
def hand_xml_path() -> Path:
    """Return a usable fixed-base MyoHand pose XML path."""
    global _HAND_ASSET_VIEW_TMPDIR

    from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import (
        HandReachCfg,
    )
    from myosuite.utils.asset_path_resolver import (
        get_sim_asset_root,
        resolve_model_xml_path,
    )

    locator_cfg = HandReachCfg(
        num_envs=1
    )

    original_xml = Path(
        locator_cfg.model_path
    )

    if not original_xml.exists():
        original_xml = Path(
            resolve_model_xml_path(
                original_xml
            )
        )

    if not original_xml.exists():
        raise FileNotFoundError(
            "Could not find the MyoHand XML: "
            f"{original_xml}"
        )

    original_hand_dir = (
        original_xml.parent
    )
    original_assets_dir = (
        original_hand_dir.parent
    )

    sibling_myo_sim_dir = (
        original_assets_dir
        / "myo_sim"
    )
    source_checkout_myo_sim_dir = (
        original_hand_dir.parents[3]
        / "simhive"
        / "myo_sim"
    )

    if (
        sibling_myo_sim_dir.exists()
        or source_checkout_myo_sim_dir.exists()
    ):
        return original_xml

    real_myo_sim_dir = Path(
        get_sim_asset_root(
            "myo_sim"
        )
    )

    if not real_myo_sim_dir.exists():
        raise FileNotFoundError(
            "Could not find the myo_sim "
            f"asset root: {real_myo_sim_dir}"
        )

    _HAND_ASSET_VIEW_TMPDIR = (
        tempfile.TemporaryDirectory(
            prefix=(
                "myosuite_baoding_hold_assets_"
            )
        )
    )

    temporary_package_dir = (
        Path(
            _HAND_ASSET_VIEW_TMPDIR.name
        )
        / "myosuite"
    )

    temporary_assets_dir = (
        temporary_package_dir
        / "envs"
        / "myo"
        / "assets"
    )

    temporary_hand_dir = (
        temporary_assets_dir
        / "hand"
    )

    temporary_sibling_myo_sim_dir = (
        temporary_assets_dir
        / "myo_sim"
    )

    temporary_source_myo_sim_dir = (
        temporary_package_dir
        / "simhive"
        / "myo_sim"
    )

    temporary_hand_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        original_hand_dir,
        temporary_hand_dir,
        dirs_exist_ok=True,
    )

    try:
        temporary_source_myo_sim_dir.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        temporary_source_myo_sim_dir.symlink_to(
            real_myo_sim_dir,
            target_is_directory=True,
        )
    except OSError:
        shutil.copytree(
            real_myo_sim_dir,
            temporary_source_myo_sim_dir,
            dirs_exist_ok=True,
        )

    try:
        temporary_sibling_myo_sim_dir.symlink_to(
            temporary_source_myo_sim_dir,
            target_is_directory=True,
        )
    except OSError:
        shutil.copytree(
            temporary_source_myo_sim_dir,
            temporary_sibling_myo_sim_dir,
            dirs_exist_ok=True,
        )

    patched_xml = (
        temporary_hand_dir
        / original_xml.name
    )

    if not patched_xml.exists():
        raise FileNotFoundError(
            "The temporary MyoHand XML "
            "was not created: "
            f"{patched_xml}"
        )

    return patched_xml


@lru_cache(maxsize=1)
def reference_baoding_xml_path() -> Path:
    """Return the classic MyoSuite Baoding XML."""
    path = hand_xml_path().with_name(
        "myohand_baoding.xml"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Could not find "
            "myohand_baoding.xml beside "
            "the MyoHand pose XML: "
            f"{path}"
        )

    return path


@lru_cache(maxsize=1)
def hand_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(
        str(
            hand_xml_path()
        )
    )


@lru_cache(maxsize=1)
def reference_baoding_model():
    """Load the classic model to copy the original ball properties."""
    import mujoco

    return mujoco.MjModel.from_xml_path(
        str(
            reference_baoding_xml_path()
        )
    )


def hand_spec_fn():
    import mujoco

    return mujoco.MjSpec.from_file(
        str(
            hand_xml_path()
        )
    )


@lru_cache(maxsize=1)
def hand_joint_names() -> tuple[
    str,
    ...,
]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for joint_id in range(
        model.njnt
    ):
        joint_type = int(
            model.jnt_type[
                joint_id
            ]
        )

        if joint_type in (
            int(
                mujoco.mjtJoint.mjJNT_FREE
            ),
            int(
                mujoco.mjtJoint.mjJNT_BALL
            ),
        ):
            raise RuntimeError(
                "The fixed-base MyoHand "
                "pose model unexpectedly "
                "contains a free or ball joint."
            )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if name is None:
            raise RuntimeError(
                f"MyoHand joint {joint_id} "
                "has no name."
            )

        names.append(name)

    return tuple(names)


@lru_cache(maxsize=1)
def hand_initial_joint_positions() -> (
    dict[str, float]
):
    """Return the classic palm-up Baoding hand pose."""
    names = hand_joint_names()

    if (
        len(names)
        != EXPECTED_HAND_QPOS_DIM
    ):
        raise RuntimeError(
            "Expected 23 MyoHand joints, "
            f"but found {len(names)}: "
            f"{names}"
        )

    positions = {
        name: 0.0
        for name in names
    }

    positions[names[0]] = (
        PALM_UP_JOINT_POSITION
    )

    return positions


@lru_cache(maxsize=1)
def hand_tendon_names() -> tuple[
    str,
    ...,
]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for actuator_id in range(
        model.nu
    ):
        tendon_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if tendon_id < 0:
            continue

        tendon_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_id,
        )

        if tendon_name is not None:
            names.append(
                tendon_name
            )

    if len(names) == 0:
        raise RuntimeError(
            "Could not find tendon names "
            "for the MyoHand actuators."
        )

    return tuple(names)


@lru_cache(maxsize=1)
def hand_muscle_names() -> tuple[
    str,
    ...,
]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for actuator_id in range(
        model.nu
    ):
        actuator_name = (
            mujoco.mj_id2name(
                model,
                (
                    mujoco.mjtObj
                    .mjOBJ_ACTUATOR
                ),
                actuator_id,
            )
        )

        if actuator_name is not None:
            names.append(
                actuator_name
            )
            continue

        tendon_id = int(
            model.actuator_trnid[
                actuator_id,
                0,
            ]
        )

        if tendon_id < 0:
            continue

        tendon_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_id,
        )

        if tendon_name is not None:
            names.append(
                tendon_name.replace(
                    "_tendon",
                    "",
                )
            )

    if len(names) == 0:
        raise RuntimeError(
            "Could not find actuator or "
            "muscle names for the MyoHand."
        )

    return tuple(names)


# =============================================================================
# BALL ENTITIES
# =============================================================================

@lru_cache(maxsize=None)
def original_ball_properties(
    ball_name: str,
) -> dict[str, Any]:
    """Read one ball's physical and visual properties from MyoSuite."""
    import mujoco

    if ball_name not in (
        BALL1_ENTITY_NAME,
        BALL2_ENTITY_NAME,
    ):
        raise ValueError(
            "Unknown ball name: "
            f"{ball_name!r}"
        )

    model = (
        reference_baoding_model()
    )

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        ball_name,
    )

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

    if min(
        body_id,
        geom_id,
        site_id,
    ) < 0:
        raise RuntimeError(
            "Could not resolve the original "
            "body, geom and site for "
            f"{ball_name!r}."
        )

    return {
        "radius": float(
            model.geom_size[
                geom_id,
                0,
            ]
        ),
        "mass": float(
            model.body_mass[
                body_id
            ]
        ),
        "friction": tuple(
            float(value)
            for value in (
                model.geom_friction[
                    geom_id
                ]
            )
        ),
        "solref": tuple(
            float(value)
            for value in (
                model.geom_solref[
                    geom_id
                ]
            )
        ),
        "solimp": tuple(
            float(value)
            for value in (
                model.geom_solimp[
                    geom_id
                ]
            )
        ),
        "margin": float(
            model.geom_margin[
                geom_id
            ]
        ),
        "gap": float(
            model.geom_gap[
                geom_id
            ]
        ),
        "condim": int(
            model.geom_condim[
                geom_id
            ]
        ),
        "contype": int(
            model.geom_contype[
                geom_id
            ]
        ),
        "conaffinity": int(
            model.geom_conaffinity[
                geom_id
            ]
        ),
        "priority": int(
            model.geom_priority[
                geom_id
            ]
        ),
        "geom_group": int(
            model.geom_group[
                geom_id
            ]
        ),
        "geom_rgba": tuple(
            float(value)
            for value in (
                model.geom_rgba[
                    geom_id
                ]
            )
        ),
        "site_size": float(
            model.site_size[
                site_id,
                0,
            ]
        ),
        "site_group": int(
            model.site_group[
                site_id
            ]
        ),
        "site_rgba": tuple(
            float(value)
            for value in (
                model.site_rgba[
                    site_id
                ]
            )
        ),
    }


def scaled_ball_radius(
    ball_name: str,
) -> float:
    return (
        original_ball_properties(
            ball_name
        )["radius"]
        * BALL_SIZE_SCALE
    )


def scaled_ball_mass(
    ball_name: str,
) -> float:
    return (
        original_ball_properties(
            ball_name
        )["mass"]
        * BALL_MASS_SCALE
    )


def make_ball_spec(
    ball_name: str,
):
    """Create one smaller, density-preserving MyoSuite ball entity."""
    import mujoco

    properties = (
        original_ball_properties(
            ball_name
        )
    )

    spec = mujoco.MjSpec()
    spec.modelname = (
        f"MyoSuite {ball_name}"
    )
    spec.option.timestep = SIM_DT

    body = spec.worldbody.add_body(
        name=ball_name,
    )

    body.add_joint(
        name=(
            f"{ball_name}_freejoint"
        ),
        type=(
            mujoco.mjtJoint.mjJNT_FREE
        ),
    )

    geom = body.add_geom(
        name=ball_name,
        type=(
            mujoco.mjtGeom
            .mjGEOM_SPHERE
        ),
        size=[
            scaled_ball_radius(
                ball_name
            ),
            0.0,
            0.0,
        ],
        mass=scaled_ball_mass(
            ball_name
        ),
        condim=(
            properties["condim"]
        ),
        contype=(
            properties["contype"]
        ),
        conaffinity=(
            properties["conaffinity"]
        ),
        priority=(
            properties["priority"]
        ),
        group=(
            properties["geom_group"]
        ),
        friction=(
            properties["friction"]
        ),
        solref=(
            properties["solref"]
        ),
        solimp=(
            properties["solimp"]
        ),
        margin=(
            properties["margin"]
        ),
        gap=properties["gap"],
        rgba=(
            properties["geom_rgba"]
        ),
    )

    body.add_site(
        name=(
            f"{ball_name}_site"
        ),
        type=(
            mujoco.mjtGeom
            .mjGEOM_SPHERE
        ),
        size=[
            (
                properties["site_size"]
                * BALL_SIZE_SCALE
            ),
            0.0,
            0.0,
        ],
        group=(
            properties["site_group"]
        ),
        rgba=(
            properties["site_rgba"]
        ),
    )

    if geom is None:
        raise RuntimeError(
            f"Could not create the "
            f"{ball_name} geom."
        )

    return spec


def ball1_spec_fn():
    return make_ball_spec(
        BALL1_ENTITY_NAME
    )


def ball2_spec_fn():
    return make_ball_spec(
        BALL2_ENTITY_NAME
    )


# =============================================================================
# RANDOM BALL RESET
# =============================================================================

def _reset_env_ids_tensor(
    env,
    env_ids,
):
    """Convert all supported reset-index forms to a device tensor."""
    import torch

    all_env_ids = torch.arange(
        env.num_envs,
        device=env.device,
        dtype=torch.long,
    )

    if env_ids is None:
        return all_env_ids

    if isinstance(
        env_ids,
        slice,
    ):
        return all_env_ids[
            env_ids
        ]

    if isinstance(
        env_ids,
        torch.Tensor,
    ):
        return env_ids.to(
            device=env.device,
            dtype=torch.long,
        )

    return torch.as_tensor(
        env_ids,
        device=env.device,
        dtype=torch.long,
    )


def _sample_uniform_xyz(
    *,
    count: int,
    low: tuple[
        float,
        float,
        float,
    ],
    high: tuple[
        float,
        float,
        float,
    ],
    device,
    dtype,
):
    import torch

    low_tensor = torch.tensor(
        low,
        device=device,
        dtype=dtype,
    )

    high_tensor = torch.tensor(
        high,
        device=device,
        dtype=dtype,
    )

    return low_tensor + (
        high_tensor
        - low_tensor
    ) * torch.rand(
        (
            count,
            3,
        ),
        device=device,
        dtype=dtype,
    )


def reset_baoding_balls_randomly(
    env,
    env_ids,
) -> None:
    """Reset both balls near their known-safe palm positions.

    The default scene reset runs first. This event replaces only the two ball
    root states. A common pair offset retains the valid two-ball layout, while
    small independent offsets give the policy distinct initial states.

    Heights remain at their nominal values, initial velocities are zero and a
    horizontal separation constraint prevents ball-ball overlap.
    """
    import torch

    env_ids = (
        _reset_env_ids_tensor(
            env,
            env_ids,
        )
    )

    if env_ids.numel() == 0:
        return

    ball1 = env.scene[
        BALL1_ENTITY_NAME
    ]
    ball2 = env.scene[
        BALL2_ENTITY_NAME
    ]

    ball1_state = (
        ball1.data
        .default_root_state[
            env_ids
        ]
        .clone()
    )

    ball2_state = (
        ball2.data
        .default_root_state[
            env_ids
        ]
        .clone()
    )

    dtype = ball1_state.dtype
    count = int(
        env_ids.numel()
    )

    pair_offset = (
        _sample_uniform_xyz(
            count=count,
            low=(
                BALL_PAIR_RESET_OFFSET_LOW
            ),
            high=(
                BALL_PAIR_RESET_OFFSET_HIGH
            ),
            device=env.device,
            dtype=dtype,
        )
    )

    ball1_jitter = (
        _sample_uniform_xyz(
            count=count,
            low=(
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            high=(
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
            device=env.device,
            dtype=dtype,
        )
    )

    ball2_jitter = (
        _sample_uniform_xyz(
            count=count,
            low=(
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            high=(
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
            device=env.device,
            dtype=dtype,
        )
    )

    env_origins = (
        env.scene.env_origins[
            env_ids
        ].to(
            dtype=dtype
        )
    )

    ball1_position = (
        ball1_state[:, 0:3]
        + env_origins
        + pair_offset
        + ball1_jitter
    )

    ball2_position = (
        ball2_state[:, 0:3]
        + env_origins
        + pair_offset
        + ball2_jitter
    )

    minimum_center_height = (
        DROP_HEIGHT
        + BALL_RESET_DROP_CLEARANCE
    )

    ball1_position[
        :,
        2,
    ].clamp_(
        min=minimum_center_height
    )

    ball2_position[
        :,
        2,
    ].clamp_(
        min=minimum_center_height
    )

    # Enforce separation in the horizontal plane. Horizontal separation
    # guarantees at least the same full 3-D center distance without disturbing
    # the carefully chosen vertical contact pose.
    difference_xy = (
        ball2_position[:, 0:2]
        - ball1_position[:, 0:2]
    )

    distance_xy = (
        torch.linalg.vector_norm(
            difference_xy,
            dim=1,
            keepdim=True,
        )
    )

    nominal_difference_xy = (
        ball2_state[:, 0:2]
        - ball1_state[:, 0:2]
    )

    nominal_distance_xy = (
        torch.linalg.vector_norm(
            nominal_difference_xy,
            dim=1,
            keepdim=True,
        ).clamp_min(
            1e-9
        )
    )

    fallback_direction_xy = (
        nominal_difference_xy
        / nominal_distance_xy
    )

    sampled_direction_xy = (
        difference_xy
        / distance_xy.clamp_min(
            1e-9
        )
    )

    direction_xy = torch.where(
        distance_xy > 1e-9,
        sampled_direction_xy,
        fallback_direction_xy,
    )

    minimum_center_separation = (
        scaled_ball_radius(
            BALL1_ENTITY_NAME
        )
        + scaled_ball_radius(
            BALL2_ENTITY_NAME
        )
        + BALL_SEPARATION_MARGIN
    )

    correction = 0.5 * (
        (
            minimum_center_separation
            - distance_xy
        ).clamp_min(
            0.0
        )
    )

    ball1_position[:, 0:2] -= (
        direction_xy
        * correction
    )

    ball2_position[:, 0:2] += (
        direction_xy
        * correction
    )

    ball1_state[:, 0:3] = (
        ball1_position
    )
    ball2_state[:, 0:3] = (
        ball2_position
    )

    # Preserve the valid default quaternions and remove all initial motion.
    ball1_state[:, 7:13] = 0.0
    ball2_state[:, 7:13] = 0.0

    ball1.write_root_state_to_sim(
        ball1_state,
        env_ids=env_ids,
    )

    ball2.write_root_state_to_sim(
        ball2_state,
        env_ids=env_ids,
    )


# =============================================================================
# OBSERVATIONS
# =============================================================================

def baoding_obs_hand_qpos(
    env,
):
    return env.scene[
        HAND_ENTITY_NAME
    ].data.joint_pos


def baoding_obs_hand_qvel(
    env,
):
    ctrl_dt = (
        env.physics_dt
        * env.cfg.decimation
    )

    return (
        env.scene[
            HAND_ENTITY_NAME
        ].data.joint_vel
        * ctrl_dt
    )


def _ball_position(
    env,
    entity_name: str,
):
    return env.scene[
        entity_name
    ].data.root_link_pos_w


def _ball_linear_velocity(
    env,
    entity_name: str,
):
    ctrl_dt = (
        env.physics_dt
        * env.cfg.decimation
    )

    return (
        env.scene[
            entity_name
        ].data.root_link_lin_vel_w
        * ctrl_dt
    )


def baoding_obs_ball1_pos(
    env,
):
    return _ball_position(
        env,
        BALL1_ENTITY_NAME,
    )


def baoding_obs_ball1_lin_vel(
    env,
):
    return _ball_linear_velocity(
        env,
        BALL1_ENTITY_NAME,
    )


def baoding_obs_ball2_pos(
    env,
):
    return _ball_position(
        env,
        BALL2_ENTITY_NAME,
    )


def baoding_obs_ball2_lin_vel(
    env,
):
    return _ball_linear_velocity(
        env,
        BALL2_ENTITY_NAME,
    )


def baoding_obs_activation(
    env,
):
    data = env.scene[
        HAND_ENTITY_NAME
    ].data.data

    return data.act


# =============================================================================
# REWARD AND TERMINATION
# =============================================================================

def baoding_ball1_dropped(
    env,
):
    return (
        baoding_obs_ball1_pos(
            env
        )[:, 2]
        < DROP_HEIGHT
    )


def baoding_ball2_dropped(
    env,
):
    return (
        baoding_obs_ball2_pos(
            env
        )[:, 2]
        < DROP_HEIGHT
    )


def baoding_any_ball_dropped(
    env,
):
    """Return true when either ball crosses the drop threshold."""
    return (
        baoding_ball1_dropped(
            env
        )
        | baoding_ball2_dropped(
            env
        )
    )


def baoding_both_balls_held_reward(
    env,
):
    """Return one while both balls are retained."""
    return (
        ~baoding_any_ball_dropped(
            env
        )
    ).float()


def baoding_drop_indicator(
    env,
):
    """Return one on a dropped state."""
    return baoding_any_ball_dropped(
        env
    ).float()


# =============================================================================
# MJLAB ENVIRONMENT CONFIGURATION
# =============================================================================

def make_baoding_hold_env_cfg(
    num_envs: int,
    play: bool = False,
):
    from mjlab.actuator import (
        XmlActuatorCfg
        as _XmlWrappedActuatorCfg,
    )
    from mjlab.actuator.actuator import (
        TransmissionType,
    )
    from mjlab.entity import EntityCfg
    from mjlab.managers.event_manager import (
        EventTermCfg,
    )
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import (
        RewardTermCfg,
    )
    from mjlab.managers.termination_manager import (
        TerminationTermCfg,
    )
    from mjlab.sim import (
        MujocoCfg,
        SimulationCfg,
    )

    from myosuite.core.config import (
        TaskConfig,
    )
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
        MyoMuscleActivationActionCfg,
        mjlab_env_cfg_from_task_config,
    )

    del play

    task_settings = (
        make_baoding_hold_task_settings(
            num_envs=num_envs
        )
    )

    decimation = int(
        round(
            task_settings.ctrl_dt
            / task_settings.sim_dt
        )
    )

    if decimation <= 0:
        raise ValueError(
            "Invalid Baoding decimation "
            f"from ctrl_dt="
            f"{task_settings.ctrl_dt} "
            f"and sim_dt="
            f"{task_settings.sim_dt}."
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "hand_qpos": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_hand_qpos
                        )
                    )
                ),
                "hand_qvel": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_hand_qvel
                        )
                    )
                ),
                "ball1_pos": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_ball1_pos
                        )
                    )
                ),
                "ball1_lin_vel": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_ball1_lin_vel
                        )
                    )
                ),
                "ball2_pos": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_ball2_pos
                        )
                    )
                ),
                "ball2_lin_vel": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_ball2_lin_vel
                        )
                    )
                ),
                "activation": (
                    ObservationTermCfg(
                        func=(
                            baoding_obs_activation
                        )
                    )
                ),
            },
        ),
    }

    actions = {
        "muscles": (
            MyoMuscleActivationActionCfg(
                entity_name=(
                    HAND_ENTITY_NAME
                ),
                actuator_names=(
                    hand_muscle_names()
                ),
            )
        ),
    }

    env_cfg = (
        mjlab_env_cfg_from_task_config(
            cfg=TaskConfig(
                max_episode_steps=(
                    task_settings
                    .max_episode_steps
                )
            ),
            spec_fn=hand_spec_fn,
            entity_name=(
                HAND_ENTITY_NAME
            ),
            actuators=(
                _XmlWrappedActuatorCfg(
                    target_names_expr=(
                        hand_tendon_names()
                    ),
                    transmission_type=(
                        TransmissionType.TENDON
                    ),
                ),
            ),
            observations=observations,
            actions=actions,
            num_envs=num_envs,
            decimation=decimation,
            sim_cfg=SimulationCfg(
                njmax=512,
                nconmax=256,
                mujoco=MujocoCfg(
                    timestep=(
                        task_settings.sim_dt
                    )
                ),
            ),
        )
    )

    # Reproduce the classic Baoding reset: open hand with palm facing upward.
    hand_entity_cfg = (
        env_cfg.scene.entities[
            HAND_ENTITY_NAME
        ]
    )

    hand_entity_cfg.init_state.joint_pos = (
        hand_initial_joint_positions()
    )

    hand_entity_cfg.init_state.joint_vel = {
        name: 0.0
        for name in hand_joint_names()
    }

    # Each ball is a separate entity because an MJLab entity may contain at
    # most one free joint.
    env_cfg.scene.entities[
        BALL1_ENTITY_NAME
    ] = EntityCfg(
        spec_fn=ball1_spec_fn,
        init_state=(
            EntityCfg.InitialStateCfg(
                pos=BALL1_INITIAL_POS,
                rot=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                lin_vel=(
                    0.0,
                    0.0,
                    0.0,
                ),
                ang_vel=(
                    0.0,
                    0.0,
                    0.0,
                ),
            )
        ),
    )

    env_cfg.scene.entities[
        BALL2_ENTITY_NAME
    ] = EntityCfg(
        spec_fn=ball2_spec_fn,
        init_state=(
            EntityCfg.InitialStateCfg(
                pos=BALL2_INITIAL_POS,
                rot=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                lin_vel=(
                    0.0,
                    0.0,
                    0.0,
                ),
                ang_vel=(
                    0.0,
                    0.0,
                    0.0,
                ),
            )
        ),
    )

    # The default reset_scene_to_default event remains first. This event runs
    # afterwards and replaces the two fixed ball states with constrained random
    # states at every episode reset.
    env_cfg.events[
        "randomize_baoding_balls"
    ] = EventTermCfg(
        func=(
            reset_baoding_balls_randomly
        ),
        mode="reset",
    )

    env_cfg.rewards = {
        "both_balls_held": (
            RewardTermCfg(
                func=(
                    baoding_both_balls_held_reward
                ),
                weight=(
                    HOLD_REWARD_PER_STEP
                ),
            )
        ),
        "ball_dropped": (
            RewardTermCfg(
                func=(
                    baoding_drop_indicator
                ),
                weight=(
                    -DROP_PENALTY
                ),
            )
        ),
    }

    # Preserve the time-limit termination and add immediate task failure when
    # either ball drops.
    env_cfg.terminations[
        "ball_dropped"
    ] = TerminationTermCfg(
        func=(
            baoding_any_ball_dropped
        ),
        time_out=False,
    )

    # A successful 400-step episode should have a return of 400.
    if hasattr(
        env_cfg,
        "scale_rewards_by_dt",
    ):
        env_cfg.scale_rewards_by_dt = (
            False
        )

    return env_cfg


# =============================================================================
# RSL-RL CONFIGURATION AND REGISTRATION
# =============================================================================

def make_baoding_hold_rl_cfg():
    """Reuse the known-working MyoSuite elbow PPO configuration."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _elbow_ppo_runner_cfg,
    )

    return copy.deepcopy(
        _elbow_ppo_runner_cfg()
    )


def validate_static_configuration() -> None:
    if NUM_ENVS <= 0:
        raise ValueError(
            "NUM_ENVS must be positive."
        )

    if PLAY_NUM_ENVS <= 0:
        raise ValueError(
            "PLAY_NUM_ENVS must be positive."
        )

    if MAX_EPISODE_STEPS <= 0:
        raise ValueError(
            "MAX_EPISODE_STEPS must be positive."
        )

    if not (
        0.0
        < SIM_DT
        <= CTRL_DT
    ):
        raise ValueError(
            "The time steps must satisfy "
            "0 < SIM_DT <= CTRL_DT."
        )

    ratio = (
        CTRL_DT
        / SIM_DT
    )

    if (
        abs(
            ratio
            - round(ratio)
        )
        > 1e-9
    ):
        raise ValueError(
            "CTRL_DT must be an integer "
            "multiple of SIM_DT."
        )

    if DROP_HEIGHT <= 0.0:
        raise ValueError(
            "DROP_HEIGHT must be positive."
        )

    if HOLD_REWARD_PER_STEP <= 0.0:
        raise ValueError(
            "HOLD_REWARD_PER_STEP must "
            "be positive."
        )

    if DROP_PENALTY < 0.0:
        raise ValueError(
            "DROP_PENALTY must be "
            "non-negative."
        )

    if not (
        0.0
        < BALL_SIZE_SCALE
        <= 1.0
    ):
        raise ValueError(
            "BALL_SIZE_SCALE must be "
            "in (0, 1]."
        )

    expected_mass_scale = (
        BALL_SIZE_SCALE ** 3
    )

    if (
        abs(
            BALL_MASS_SCALE
            - expected_mass_scale
        )
        > 1e-12
    ):
        raise ValueError(
            "BALL_MASS_SCALE must equal "
            "BALL_SIZE_SCALE ** 3 to "
            "preserve density."
        )

    reset_ranges = (
        (
            "BALL_PAIR_RESET_OFFSET",
            BALL_PAIR_RESET_OFFSET_LOW,
            BALL_PAIR_RESET_OFFSET_HIGH,
        ),
        (
            "BALL_INDIVIDUAL_RESET_JITTER",
            (
                BALL_INDIVIDUAL_RESET_JITTER_LOW
            ),
            (
                BALL_INDIVIDUAL_RESET_JITTER_HIGH
            ),
        ),
    )

    for (
        name,
        low,
        high,
    ) in reset_ranges:
        if (
            len(low) != 3
            or len(high) != 3
        ):
            raise ValueError(
                f"{name} bounds must contain "
                "exactly three values."
            )

        if any(
            low_value > high_value
            for (
                low_value,
                high_value,
            ) in zip(
                low,
                high,
                strict=True,
            )
        ):
            raise ValueError(
                f"Every lower {name} bound "
                "must be less than or equal "
                "to its upper bound."
            )

    vertical_bounds = (
        BALL_PAIR_RESET_OFFSET_LOW[2],
        BALL_PAIR_RESET_OFFSET_HIGH[2],
        BALL_INDIVIDUAL_RESET_JITTER_LOW[2],
        BALL_INDIVIDUAL_RESET_JITTER_HIGH[2],
    )

    if any(
        abs(value) > 1e-12
        for value in vertical_bounds
    ):
        raise ValueError(
            "All Baoding reset z offsets "
            "must remain 0.0."
        )

    if BALL_SEPARATION_MARGIN < 0.0:
        raise ValueError(
            "BALL_SEPARATION_MARGIN must "
            "be non-negative."
        )

    if BALL_RESET_DROP_CLEARANCE <= 0.0:
        raise ValueError(
            "BALL_RESET_DROP_CLEARANCE "
            "must be positive."
        )

    lowest_possible_reset_height = (
        min(
            BALL1_INITIAL_POS[2],
            BALL2_INITIAL_POS[2],
        )
        + BALL_PAIR_RESET_OFFSET_LOW[2]
        + (
            BALL_INDIVIDUAL_RESET_JITTER_LOW[
                2
            ]
        )
    )

    if lowest_possible_reset_height <= (
        DROP_HEIGHT
        + BALL_RESET_DROP_CLEARANCE
    ):
        raise ValueError(
            "The ball reset ranges do not "
            "preserve the requested clearance "
            "above DROP_HEIGHT."
        )

    if (
        BALL1_INITIAL_POS[2]
        <= DROP_HEIGHT
    ):
        raise ValueError(
            "Ball 1 must start above "
            "DROP_HEIGHT."
        )

    if (
        BALL2_INITIAL_POS[2]
        <= DROP_HEIGHT
    ):
        raise ValueError(
            "Ball 2 must start above "
            "DROP_HEIGHT."
        )

    model = hand_model()

    if (
        model.nq
        != EXPECTED_HAND_QPOS_DIM
    ):
        raise RuntimeError(
            "Expected the fixed-base MyoHand "
            "model to have "
            f"nq={EXPECTED_HAND_QPOS_DIM}, "
            f"but got {model.nq}."
        )

    if (
        model.nv
        != EXPECTED_HAND_QVEL_DIM
    ):
        raise RuntimeError(
            "Expected the fixed-base MyoHand "
            "model to have "
            f"nv={EXPECTED_HAND_QVEL_DIM}, "
            f"but got {model.nv}."
        )

    if (
        len(
            hand_muscle_names()
        )
        != EXPECTED_ACTION_DIM
    ):
        raise RuntimeError(
            "Expected 39 MyoHand muscle "
            "actions, but got "
            f"{len(hand_muscle_names())}."
        )

    if OBS_DIM != (
        EXPECTED_HAND_QPOS_DIM
        + EXPECTED_HAND_QVEL_DIM
        + 3
        + 3
        + 3
        + 3
        + EXPECTED_ACTION_DIM
    ):
        raise RuntimeError(
            "The declared Baoding "
            "observation dimension is "
            "inconsistent."
        )

    original_ball_properties(
        BALL1_ENTITY_NAME
    )
    original_ball_properties(
        BALL2_ENTITY_NAME
    )


def register_baoding_hold_task(
    quiet: bool = True,
) -> None:
    """Register the survival-only Baoding task."""
    global _TASK_CONFIGURATION_LOCKED

    validate_static_configuration()

    with silence_output(quiet):
        import myosuite  # noqa: F401

        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            bootstrap_myosuite_mjlab_registry,
        )

        bootstrap_myosuite_mjlab_registry()

    import mjlab.tasks.registry as registry

    existing_tasks = set(
        registry.list_tasks()
    )

    if TASK_ID in existing_tasks:
        _TASK_CONFIGURATION_LOCKED = (
            True
        )
        return

    env_cfg = (
        make_baoding_hold_env_cfg(
            num_envs=NUM_ENVS,
            play=False,
        )
    )

    play_env_cfg = (
        make_baoding_hold_env_cfg(
            num_envs=PLAY_NUM_ENVS,
            play=True,
        )
    )

    rl_cfg = (
        make_baoding_hold_rl_cfg()
    )

    registry.register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=None,
    )

    _TASK_CONFIGURATION_LOCKED = True


# =============================================================================
# STANDALONE REGISTRATION CHECK
# =============================================================================

def main() -> None:
    register_baoding_hold_task(
        quiet=True
    )

    import mjlab.tasks.registry as registry

    env_cfg = registry.load_env_cfg(
        TASK_ID
    )

    play_env_cfg = (
        registry.load_env_cfg(
            TASK_ID,
            play=True,
        )
    )

    rl_cfg = registry.load_rl_cfg(
        TASK_ID
    )

    settings = (
        make_baoding_hold_task_settings(
            num_envs=1
        )
    )

    print()
    print(
        "Registered MJLab Baoding "
        "hold task"
    )
    print("=" * 80)
    print(
        "  task id:             ",
        TASK_ID,
    )
    print(
        "  train env cfg:       ",
        type(env_cfg),
    )
    print(
        "  play env cfg:        ",
        type(play_env_cfg),
    )
    print(
        "  RL cfg:              ",
        type(rl_cfg),
    )
    print(
        "  hand XML:            ",
        hand_xml_path(),
    )
    print(
        "  reference XML:       ",
        reference_baoding_xml_path(),
    )
    print(
        "  scene entities:      ",
        tuple(
            env_cfg.scene.entities
        ),
    )
    print(
        "  train num envs:      ",
        env_cfg.scene.num_envs,
    )
    print(
        "  play num envs:       ",
        play_env_cfg.scene.num_envs,
    )
    print(
        "  max episode steps:   ",
        settings.max_episode_steps,
    )
    print(
        "  sim dt:              ",
        settings.sim_dt,
    )
    print(
        "  control dt:          ",
        settings.ctrl_dt,
    )
    print(
        "  decimation:          ",
        env_cfg.decimation,
    )
    print(
        "  drop height:         ",
        settings.drop_height,
    )
    print(
        "  ball 1 initial pos:  ",
        BALL1_INITIAL_POS,
    )
    print(
        "  ball 2 initial pos:  ",
        BALL2_INITIAL_POS,
    )
    print(
        "  ball size scale:     ",
        BALL_SIZE_SCALE,
    )
    print(
        "  ball mass scale:     ",
        BALL_MASS_SCALE,
    )
    print(
        "  contact z adjustment:",
        -BALL_CONTACT_HEIGHT_COMPENSATION,
    )
    print(
        "  ball 1 radius:       ",
        scaled_ball_radius(
            BALL1_ENTITY_NAME
        ),
    )
    print(
        "  ball 2 radius:       ",
        scaled_ball_radius(
            BALL2_ENTITY_NAME
        ),
    )
    print(
        "  pair reset offset:   ",
        BALL_PAIR_RESET_OFFSET_LOW,
        "to",
        BALL_PAIR_RESET_OFFSET_HIGH,
    )
    print(
        "  per-ball jitter:     ",
        (
            BALL_INDIVIDUAL_RESET_JITTER_LOW
        ),
        "to",
        (
            BALL_INDIVIDUAL_RESET_JITTER_HIGH
        ),
    )
    print(
        "  reward while held:   ",
        HOLD_REWARD_PER_STEP,
    )
    print(
        "  drop penalty:        ",
        DROP_PENALTY,
    )
    print(
        "  maximum return:      ",
        (
            MAX_EPISODE_STEPS
            * HOLD_REWARD_PER_STEP
        ),
    )
    print(
        "  hand joints:         ",
        len(
            hand_joint_names()
        ),
    )
    print(
        "  action muscles:      ",
        len(
            hand_muscle_names()
        ),
    )
    print(
        "  wrapped tendons:     ",
        len(
            hand_tendon_names()
        ),
    )
    print(
        "  observation dim:     ",
        OBS_DIM,
    )
    print("=" * 80)
    print()
    print(
        "Policy observation layout:"
    )
    print(
        "  hand_qpos:      0:23"
    )
    print(
        "  hand_qvel:     23:46"
    )
    print(
        "  ball1_pos:     46:49"
    )
    print(
        "  ball1_lin_vel: 49:52"
    )
    print(
        "  ball2_pos:     52:55"
    )
    print(
        "  ball2_lin_vel: 55:58"
    )
    print(
        "  activation:    58:97"
    )
    print(
        "  total:            97"
    )


if __name__ == "__main__":
    main()