"""Register the first five SDS Baoding curriculum tasks in one MJLab task.

Source basis: Chiappa et al. (Neuron 2024, doi:10.1016/j.neuron.2024.09.002),
the original MyoSuite baoding_v1.py and myohand_baoding.xml loaded at runtime,
and MJLab's event and reward lifecycle. This is an MJLab port of the task:
the authors' separate RSI reset implementation is not imported verbatim.

Expected location:

    <RepoRootFolder>/scripts/mjlab/BaodingBalls/register_baoding_rsi_static.py

Run the registration check from the repository root with:

    python scripts/mjlab/BaodingBalls/register_baoding_rsi_static.py

Training and visualization scripts placed in the same BaodingBalls directory
can import this module directly, for example:

    from register_baoding_rsi_static import TASK_ID, register_baoding_hold_task
"""

from __future__ import annotations

import contextlib
import copy
import math
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

TASK_ID = "myoHandBaodingRSIStatic-v0"

NUM_ENVS = 64
PLAY_NUM_ENVS = 1

MAX_EPISODE_STEPS = 200

SIM_DT = 0.0025
CTRL_DT = 0.025

# Neuron (2024), Table 1, reports 1.3 m. The authors' released
# CustomBaodingEnv._setup defaults to 1.25 m, and their 01_rsi_static
# config does not override it. Match that released first-stage run.
DROP_HEIGHT = 1.25
STAGE_INDEX = 1
ROTATION_PERIOD_SEC: float | None = None
RSI_PLACEMENT_PROBABILITY = 1.0
RANDOM_TASK_CHOICE = False

# The same -distance1-distance2+alive+5*solved reward is used in all stages.
# The training wrapper may normalize it for PPO.
HOLD_REWARD_PER_STEP = 1.0
DROP_PENALTY = 0.0

BALL1_CLASSIC_INITIAL_POS = (-0.227, -0.511, 1.452)
BALL2_CLASSIC_INITIAL_POS = (-0.256, -0.552, 1.442)

# The original balls have a nominal radius of 0.022 m (44 mm diameter).
# Restore their original dimensions for a closer match to the published task.
BALL_SIZE_SCALE = 1.0

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

# The authors' RSI reset replaces only the two ball x/y coordinates with
# the corresponding goal x/y coordinates; each ball keeps its original height.
BALL_PAIR_RESET_OFFSET_LOW = (0.0, 0.0, 0.0)
BALL_PAIR_RESET_OFFSET_HIGH = (0.0, 0.0, 0.0)

BALL_INDIVIDUAL_RESET_JITTER_LOW = (0.0, 0.0, 0.0)
BALL_INDIVIDUAL_RESET_JITTER_HIGH = (0.0, 0.0, 0.0)

# The original MyoSuite Baoding task specifies this ellipse in the coordinate
# frame of the target sites' parent body, not in world x/y coordinates.
MYOSUITE_ELLIPSE_CENTER = (-0.0125, -0.07)
RSI_ELLIPSE_X_RADIUS = 0.025
RSI_ELLIPSE_Y_RADIUS = 0.028

# Two antipodal ball targets should remain around 5-6 cm apart.
MAX_RSI_RESET_PAIR_SEPARATION = 0.075

# Enforce a small free-space margin between the two sphere surfaces.
BALL_SEPARATION_MARGIN = 0.002

# The reset function checks this center-height clearance above the drop limit.
BALL_RESET_DROP_CLEARANCE = 0.10

# A fixed, evenly spaced evaluation set covers the same full uniform phase
# distribution used by random training resets. The viewer cycles over it.
EVAL_PHASE_COUNT = 64


def evaluation_phase(index: int, count: int = EVAL_PHASE_COUNT) -> float:
    """One of ``count`` evenly spaced starting phases."""
    if not 0 <= index < count:
        raise ValueError(f"Phase index must be in [0, {count}).")
    return 2.0 * math.pi * (index + 0.5) / count

PALM_UP_JOINT_POSITION = -1.57

HAND_ENTITY_NAME = "hand"
BALL1_ENTITY_NAME = "ball1"
BALL2_ENTITY_NAME = "ball2"

EXPECTED_HAND_QPOS_DIM = 23
EXPECTED_HAND_QVEL_DIM = 23
EXPECTED_ACTION_DIM = 39

# Observation order:
#
#   hand_qpos         0:23   23 values
#   ball1_pos       23:26    3 values
#   ball1_lin_vel   26:29    3 values
#   ball2_pos       29:32    3 values
#   ball2_lin_vel   32:35    3 values
#   target1_pos     35:38    3 values
#   target2_pos     38:41    3 values
#   target1_err     41:44    3 values
#   target2_err     44:47    3 values
#   activation      47:86   39 values
OBS_DIM = 86

# Keep this object alive for as long as MuJoCo may need the temporary assets.
_HAND_ASSET_VIEW_TMPDIR: tempfile.TemporaryDirectory[str] | None = None

# Once an environment configuration has been placed in MJLab's process-local
# registry, changing these values would not update that registered object.
_TASK_CONFIGURATION_LOCKED = False


def configure_baoding_stage(
    *, index: int, period_sec: float | None,
    rsi_probability: float, random_task_choice: bool, drop_height: float,
) -> None:
    """Select a curriculum environment before it is registered or created."""
    global STAGE_INDEX, ROTATION_PERIOD_SEC, RSI_PLACEMENT_PROBABILITY
    global RANDOM_TASK_CHOICE, DROP_HEIGHT
    # The registered event, reward and termination functions read these
    # globals at runtime. Both training and evaluation environments switch
    # together before they reset for the next curriculum stage.
    if index not in range(1, 6) or (index == 1) != (period_sec is None):
        raise ValueError("Stage 1 is static; stages 2-5 require a rotation period.")
    if period_sec is not None and period_sec <= 0:
        raise ValueError("Rotation period must be positive.")
    if not 0 <= rsi_probability <= 1 or drop_height <= 0:
        raise ValueError("Invalid RSI probability or drop height.")
    if min(BALL1_INITIAL_POS[2], BALL2_INITIAL_POS[2]) <= (
        drop_height + BALL_RESET_DROP_CLEARANCE
    ):
        raise ValueError("The chosen drop height leaves insufficient reset clearance.")
    STAGE_INDEX, ROTATION_PERIOD_SEC = index, period_sec
    RSI_PLACEMENT_PROBABILITY, RANDOM_TASK_CHOICE = (
        rsi_probability, random_task_choice
    )
    DROP_HEIGHT = drop_height


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


def configure_baoding_balls(
    *,
    ball_size_scale: float,
    pair_reset_offset_low: tuple[float, float, float],
    pair_reset_offset_high: tuple[float, float, float],
    individual_reset_jitter_low: tuple[float, float, float],
    individual_reset_jitter_high: tuple[float, float, float],
) -> None:
    """Apply ball settings before the MJLab task is registered.

    Training keeps these experiment-facing values in train_baoding.py and
    calls this function before environment creation. The registration module
    continues to own the derived radius, mass, contact height, and reset logic.
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

    requested_size_scale = float(ball_size_scale)
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
        if requested_configuration == current_configuration:
            return

        raise RuntimeError(
            "The Baoding task is already registered in this process. "
            "Configure the ball settings before calling "
            "register_baoding_hold_task()."
        )

    if not 0.0 < requested_size_scale <= 1.0:
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

    for name, low, high in ranges:
        if len(low) != 3 or len(high) != 3:
            raise ValueError(
                f"{name} bounds must contain exactly three values."
            )

        if any(
            low_value > high_value
            for low_value, high_value in zip(
                low,
                high,
                strict=True,
            )
        ):
            raise ValueError(
                f"Every lower {name} bound must be less than or "
                "equal to its upper bound."
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
            "The Baoding reset z offsets must remain 0.0. Randomizing "
            "height can create an air gap or an initial contact impulse."
        )

    BALL_SIZE_SCALE = requested_size_scale
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
    BALL_MASS_SCALE = BALL_SIZE_SCALE ** 3
    BALL_PAIR_RESET_OFFSET_LOW = requested_pair_low
    BALL_PAIR_RESET_OFFSET_HIGH = requested_pair_high
    BALL_INDIVIDUAL_RESET_JITTER_LOW = requested_jitter_low
    BALL_INDIVIDUAL_RESET_JITTER_HIGH = requested_jitter_high


# =============================================================================
# OUTPUT REDIRECTION
# =============================================================================

@contextlib.contextmanager
def silence_output(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            with contextlib.redirect_stderr(file):
                yield


# =============================================================================
# MYOSUITE HAND ASSETS
# =============================================================================

@lru_cache(maxsize=1)
def hand_xml_path() -> Path:
    """
    Return a usable MyoHand pose XML path.

    This follows the same temporary-asset-layout workaround as the HandReach
    registration. The Baoding scene is assembled from a fixed-base hand and two
    separate floating ball entities. Keeping the balls separate avoids placing
    two free joints in one MJLab entity.
    """
    global _HAND_ASSET_VIEW_TMPDIR

    from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import (
        HandReachCfg,
    )
    from myosuite.utils.asset_path_resolver import (
        get_sim_asset_root,
        resolve_model_xml_path,
    )

    locator_cfg = HandReachCfg(num_envs=1)
    original_xml = Path(locator_cfg.model_path)

    if not original_xml.exists():
        original_xml = Path(resolve_model_xml_path(original_xml))

    if not original_xml.exists():
        raise FileNotFoundError(
            f"Could not find the MyoHand XML: {original_xml}"
        )

    original_hand_dir = original_xml.parent
    original_assets_dir = original_hand_dir.parent

    # MyoSuite has used both of these layouts across installations:
    #
    #   assets/hand/model.xml -> ../myo_sim/...
    #   envs/myo/assets/hand/model.xml -> ../../../../simhive/myo_sim/...
    #
    # If either target is already present, the original XML is usable as-is.
    sibling_myo_sim_dir = original_assets_dir / "myo_sim"
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

    real_myo_sim_dir = Path(get_sim_asset_root("myo_sim"))

    if not real_myo_sim_dir.exists():
        raise FileNotFoundError(
            f"Could not find the myo_sim asset root: {real_myo_sim_dir}"
        )

    _HAND_ASSET_VIEW_TMPDIR = tempfile.TemporaryDirectory(
        prefix="myosuite_baoding_hold_assets_"
    )

    temporary_package_dir = (
        Path(_HAND_ASSET_VIEW_TMPDIR.name)
        / "myosuite"
    )
    temporary_assets_dir = (
        temporary_package_dir
        / "envs"
        / "myo"
        / "assets"
    )
    temporary_hand_dir = temporary_assets_dir / "hand"
    temporary_sibling_myo_sim_dir = (
        temporary_assets_dir / "myo_sim"
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

    # Also expose the compact sibling layout used by some packaged assets.
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

    patched_xml = temporary_hand_dir / original_xml.name

    if not patched_xml.exists():
        raise FileNotFoundError(
            "The temporary MyoHand XML was not created: "
            f"{patched_xml}"
        )

    return patched_xml


@lru_cache(maxsize=1)
def reference_baoding_xml_path() -> Path:
    """Return the classic MyoSuite Baoding XML beside the hand pose XML."""
    path = hand_xml_path().with_name("myohand_baoding.xml")

    if not path.exists():
        raise FileNotFoundError(
            "Could not find myohand_baoding.xml beside the MyoHand pose XML: "
            f"{path}"
        )

    return path


@lru_cache(maxsize=1)
def hand_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(hand_xml_path()))


@lru_cache(maxsize=1)
def reference_baoding_model():
    """
    Load the classic model only to copy the original ball properties.

    The combined model is not registered as one MJLab entity because it contains
    two free joints. The actual MJLab scene uses one entity per ball.
    """
    import mujoco

    return mujoco.MjModel.from_xml_path(
        str(reference_baoding_xml_path())
    )


@lru_cache(maxsize=1)
def reference_goal_cycle_frame():
    """Resolve the original goal-site parent and its palm-local ellipse."""
    import mujoco
    import numpy as np

    model = reference_baoding_model()
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    for joint_name, position in hand_initial_joint_positions().items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(
                f"Baoding reference model has no hand joint {joint_name!r}."
            )
        data.qpos[int(model.jnt_qposadr[joint_id])] = position

    mujoco.mj_forward(model, data)

    target_ids = []
    for name in ("target1_site", "target2_site"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise RuntimeError(f"Baoding reference model has no site {name!r}.")
        target_ids.append(site_id)

    parent_ids = [int(model.site_bodyid[site_id]) for site_id in target_ids]
    if parent_ids[0] != parent_ids[1]:
        raise RuntimeError(
            "The reference goal sites have different parent bodies; the "
            "single-palm RSI transform must be adapted to this XML."
        )

    parent_id = parent_ids[0]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
    if not parent_name or mujoco.mj_name2id(
        hand_model(), mujoco.mjtObj.mjOBJ_BODY, parent_name
    ) < 0:
        raise RuntimeError(
            "The original Baoding goal parent is missing from the MJLab hand: "
            f"{parent_name!r}."
        )
    rotation = np.asarray(data.xmat[parent_id]).reshape(3, 3).copy()
    translation = np.asarray(data.xpos[parent_id]).copy()
    local_center_xy = np.asarray(MYOSUITE_ELLIPSE_CENTER)
    local_centers = [
        np.asarray(
            (local_center_xy[0], local_center_xy[1], model.site_pos[sid, 2])
        )
        for sid in target_ids
    ]
    world_centers = [translation + rotation @ xyz for xyz in local_centers]

    # The authors' RSI reset copies the target x/y into the ball free joints;
    # there is no per-ball correction to the trajectory or changed ball z.
    phases = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    pair_base = world_centers[0] - world_centers[1]
    pair_positions = (
        pair_base[None, :]
        + (2.0 * RSI_ELLIPSE_X_RADIUS * np.cos(phases))[:, None]
        * rotation[:, 0][None, :]
        + (2.0 * RSI_ELLIPSE_Y_RADIUS * np.sin(phases))[:, None]
        * rotation[:, 1][None, :]
    )
    pair_positions[:, 2] = BALL1_INITIAL_POS[2] - BALL2_INITIAL_POS[2]
    max_pair_separation = float(
        np.linalg.norm(pair_positions, axis=1).max()
    )
    if max_pair_separation > MAX_RSI_RESET_PAIR_SEPARATION:
        raise RuntimeError(
            "Baoding RSI reset would separate the balls by up to "
            f"{100.0 * max_pair_separation:.1f} cm; expected at most "
            f"{100.0 * MAX_RSI_RESET_PAIR_SEPARATION:.1f} cm. "
            "Check the reference XML's target and ball poses."
        )

    return (
        tuple(tuple(float(x) for x in center) for center in world_centers),
        tuple(float(x) for x in rotation[:, 0]),
        tuple(float(x) for x in rotation[:, 1]),
        parent_name,
        tuple(tuple(float(x) for x in center) for center in local_centers),
    )


def hand_spec_fn():
    import mujoco

    return mujoco.MjSpec.from_file(str(hand_xml_path()))


@lru_cache(maxsize=1)
def hand_joint_names() -> tuple[str, ...]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])

        if joint_type in (
            int(mujoco.mjtJoint.mjJNT_FREE),
            int(mujoco.mjtJoint.mjJNT_BALL),
        ):
            raise RuntimeError(
                "The fixed-base MyoHand pose model unexpectedly contains "
                "a free or ball joint."
            )

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        if name is None:
            raise RuntimeError(
                f"MyoHand joint {joint_id} has no name."
            )

        names.append(name)

    return tuple(names)


@lru_cache(maxsize=1)
def hand_initial_joint_positions() -> dict[str, float]:
    """
    Reproduce the initial hand pose used by the classic Baoding task.

    The classic environment zeros all 23 hand coordinates and then sets the
    first coordinate to -1.57 rad so that the palm faces upward.
    """
    names = hand_joint_names()

    if len(names) != EXPECTED_HAND_QPOS_DIM:
        raise RuntimeError(
            "Expected 23 MyoHand joints, but found "
            f"{len(names)}: {names}"
        )

    positions = {
        name: 0.0
        for name in names
    }
    positions[names[0]] = PALM_UP_JOINT_POSITION

    return positions


@lru_cache(maxsize=1)
def hand_tendon_names() -> tuple[str, ...]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for actuator_id in range(model.nu):
        tendon_id = int(model.actuator_trnid[actuator_id, 0])

        if tendon_id < 0:
            continue

        tendon_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_id,
        )

        if tendon_name is not None:
            names.append(tendon_name)

    if len(names) == 0:
        raise RuntimeError(
            "Could not find tendon names for the MyoHand actuators."
        )

    return tuple(names)


@lru_cache(maxsize=1)
def hand_muscle_names() -> tuple[str, ...]:
    import mujoco

    model = hand_model()
    names: list[str] = []

    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        if actuator_name is not None:
            names.append(actuator_name)
            continue

        tendon_id = int(model.actuator_trnid[actuator_id, 0])

        if tendon_id < 0:
            continue

        tendon_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_TENDON,
            tendon_id,
        )

        if tendon_name is not None:
            names.append(
                tendon_name.replace("_tendon", "")
            )

    if len(names) == 0:
        raise RuntimeError(
            "Could not find actuator/muscle names for the MyoHand."
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
            f"Unknown ball name: {ball_name!r}"
        )

    model = reference_baoding_model()

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

    if min(body_id, geom_id, site_id) < 0:
        raise RuntimeError(
            "Could not resolve the original body, geom, and site for "
            f"{ball_name!r}."
        )

    return {
        "radius": float(model.geom_size[geom_id, 0]),
        "mass": float(model.body_mass[body_id]),
        "friction": tuple(
            float(value)
            for value in model.geom_friction[geom_id]
        ),
        "solref": tuple(
            float(value)
            for value in model.geom_solref[geom_id]
        ),
        "solimp": tuple(
            float(value)
            for value in model.geom_solimp[geom_id]
        ),
        "margin": float(model.geom_margin[geom_id]),
        "gap": float(model.geom_gap[geom_id]),
        "condim": int(model.geom_condim[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
        "priority": int(model.geom_priority[geom_id]),
        "geom_group": int(model.geom_group[geom_id]),
        "geom_rgba": tuple(
            float(value)
            for value in model.geom_rgba[geom_id]
        ),
        "site_size": float(model.site_size[site_id, 0]),
        "site_group": int(model.site_group[site_id]),
        "site_rgba": tuple(
            float(value)
            for value in model.site_rgba[site_id]
        ),
    }


def scaled_ball_radius(
    ball_name: str,
) -> float:
    return (
        original_ball_properties(ball_name)["radius"]
        * BALL_SIZE_SCALE
    )


def scaled_ball_mass(
    ball_name: str,
) -> float:
    return (
        original_ball_properties(ball_name)["mass"]
        * BALL_MASS_SCALE
    )


def make_ball_spec(
    ball_name: str,
):
    """Create one smaller, density-preserving MyoSuite ball entity."""
    import mujoco

    properties = original_ball_properties(ball_name)

    spec = mujoco.MjSpec()
    spec.modelname = f"MyoSuite {ball_name}"
    spec.option.timestep = SIM_DT

    body = spec.worldbody.add_body(
        name=ball_name,
    )
    body.add_joint(
        name=f"{ball_name}_freejoint",
        type=mujoco.mjtJoint.mjJNT_FREE,
    )

    geom = body.add_geom(
        name=ball_name,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[scaled_ball_radius(ball_name), 0.0, 0.0],
        mass=scaled_ball_mass(ball_name),
        condim=properties["condim"],
        contype=properties["contype"],
        conaffinity=properties["conaffinity"],
        priority=properties["priority"],
        group=properties["geom_group"],
        friction=properties["friction"],
        solref=properties["solref"],
        solimp=properties["solimp"],
        margin=properties["margin"],
        gap=properties["gap"],
        rgba=properties["geom_rgba"],
    )

    # Keep a named center site for inspection and future task stages.
    body.add_site(
        name=f"{ball_name}_site",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[
            properties["site_size"] * BALL_SIZE_SCALE,
            0.0,
            0.0,
        ],
        group=properties["site_group"],
        rgba=properties["site_rgba"],
    )

    if geom is None:
        raise RuntimeError(
            f"Could not create the {ball_name} geom."
        )

    return spec


def ball1_spec_fn():
    return make_ball_spec(BALL1_ENTITY_NAME)


def ball2_spec_fn():
    return make_ball_spec(BALL2_ENTITY_NAME)


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

    if isinstance(env_ids, slice):
        return all_env_ids[env_ids]

    if isinstance(env_ids, torch.Tensor):
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
    low: tuple[float, float, float],
    high: tuple[float, float, float],
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
        high_tensor - low_tensor
    ) * torch.rand(
        (count, 3),
        device=device,
        dtype=dtype,
    )


def reset_baoding_balls_randomly(
    env,
    env_ids,
) -> None:
    """Sample target phase and task; optionally align ball x/y with the goals."""
    import torch

    env_ids = _reset_env_ids_tensor(
        env,
        env_ids,
    )

    if env_ids.numel() == 0:
        return

    ball1 = env.scene[BALL1_ENTITY_NAME]
    ball2 = env.scene[BALL2_ENTITY_NAME]

    ball1_state = (
        ball1.data.default_root_state[env_ids]
        .clone()
    )
    ball2_state = (
        ball2.data.default_root_state[env_ids]
        .clone()
    )

    dtype = ball1_state.dtype
    count = int(env_ids.numel())

    env_origins = env.scene.env_origins[
        env_ids
    ].to(dtype=dtype)

    phase = 2.0 * torch.pi * torch.rand(
        count,
        device=env.device,
        dtype=dtype,
    )
    forced_phase = getattr(env, "_rsi_forced_eval_phase", None)
    if forced_phase is not None:
        if tuple(forced_phase.shape) != (env.num_envs,):
            raise RuntimeError("Evaluation phase override has the wrong shape.")
        phase = forced_phase[env_ids].to(device=env.device, dtype=dtype)

    # 0 = hold, -1 = clockwise, +1 = counterclockwise. The author's
    # task_choice='random' samples all three at each reset.
    direction = (
        torch.randint(0, 3, (count,), device=env.device) - 1
        if RANDOM_TASK_CHOICE
        else torch.zeros(count, device=env.device, dtype=torch.long)
    )
    forced_direction = getattr(env, "_rsi_forced_eval_direction", None)
    if forced_direction is not None:
        if tuple(forced_direction.shape) != (env.num_envs,):
            raise RuntimeError("Evaluation direction override has the wrong shape.")
        direction = forced_direction[env_ids].to(device=env.device, dtype=torch.long)
    place_at_target = torch.rand(count, device=env.device) < RSI_PLACEMENT_PROBABILITY

    centers, x_axis, y_axis, _, _ = reference_goal_cycle_frame()
    x_axis_tensor = torch.as_tensor(x_axis, device=env.device, dtype=dtype)
    y_axis_tensor = torch.as_tensor(y_axis, device=env.device, dtype=dtype)
    radius_x = RSI_ELLIPSE_X_RADIUS * torch.cos(phase)[:, None]
    radius_y = RSI_ELLIPSE_Y_RADIUS * torch.sin(phase)[:, None]
    cycle_offset = radius_x * x_axis_tensor + radius_y * y_axis_tensor
    ball1_position = ball1_state[:, 0:3].clone() + env_origins
    ball2_position = ball2_state[:, 0:3].clone() + env_origins
    target1_xy = (
        torch.as_tensor(centers[0], device=env.device, dtype=dtype)
        + cycle_offset + env_origins
    )[:, 0:2]
    target2_xy = (
        torch.as_tensor(centers[1], device=env.device, dtype=dtype)
        - cycle_offset + env_origins
    )[:, 0:2]
    ball1_position[:, 0:2] = torch.where(
        place_at_target[:, None], target1_xy, ball1_position[:, 0:2]
    )
    ball2_position[:, 0:2] = torch.where(
        place_at_target[:, None], target2_xy, ball2_position[:, 0:2]
    )

    if bool((ball1_position[:, 2] <= DROP_HEIGHT + BALL_RESET_DROP_CLEARANCE).any()) or bool(
        (ball2_position[:, 2] <= DROP_HEIGHT + BALL_RESET_DROP_CLEARANCE).any()
    ):
        raise RuntimeError("A ball starts too close to the drop threshold.")

    # Enforce separation in the horizontal plane. Horizontal separation alone
    # guarantees at least the same full 3-D center distance.
    difference_xy = (
        ball2_position[:, 0:2]
        - ball1_position[:, 0:2]
    )
    distance_xy = torch.linalg.vector_norm(
        difference_xy,
        dim=1,
        keepdim=True,
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
        ).clamp_min(1e-9)
    )
    fallback_direction_xy = (
        nominal_difference_xy
        / nominal_distance_xy
    )

    sampled_direction_xy = (
        difference_xy
        / distance_xy.clamp_min(1e-9)
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
        minimum_center_separation
        - distance_xy
    ).clamp_min(0.0)

    ball1_position[:, 0:2] -= (
        direction_xy * correction
    )
    ball2_position[:, 0:2] += (
        direction_xy * correction
    )

    ball1_state[:, 0:3] = ball1_position
    ball2_state[:, 0:3] = ball2_position

    # Keep valid default quaternions and explicitly remove all initial motion.
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

    if not hasattr(env, "_rsi_static_phase"):
        env._rsi_static_phase = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=dtype,
        )

    env._rsi_static_phase[env_ids] = phase
    if not hasattr(env, "_rsi_direction"):
        env._rsi_direction = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
        env._rsi_episode_step = torch.zeros_like(env._rsi_direction)
        env._rsi_reset_placed = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    env._rsi_direction[env_ids] = direction
    env._rsi_episode_step[env_ids] = 0
    env._rsi_reset_placed[env_ids] = place_at_target


# =============================================================================
# OBSERVATIONS
# =============================================================================

def baoding_obs_hand_qpos(env):
    return env.scene[HAND_ENTITY_NAME].data.joint_pos


def baoding_obs_hand_qvel(env):
    ctrl_dt = env.physics_dt * env.cfg.decimation

    return (
        env.scene[HAND_ENTITY_NAME].data.joint_vel
        * ctrl_dt
    )


def _ball_position(
    env,
    entity_name: str,
):
    return env.scene[entity_name].data.root_link_pos_w


def _ball_linear_velocity(
    env,
    entity_name: str,
):
    ctrl_dt = env.physics_dt * env.cfg.decimation

    return (
        env.scene[entity_name].data.root_link_lin_vel_w
        * ctrl_dt
    )


def baoding_obs_ball1_pos(env):
    return _ball_position(env, BALL1_ENTITY_NAME)


def baoding_obs_ball1_lin_vel(env):
    return _ball_linear_velocity(
        env,
        BALL1_ENTITY_NAME,
    )


def baoding_obs_ball2_pos(env):
    return _ball_position(env, BALL2_ENTITY_NAME)


def baoding_obs_ball2_lin_vel(env):
    return _ball_linear_velocity(
        env,
        BALL2_ENTITY_NAME,
    )


def baoding_obs_activation(env):
    data = env.scene[HAND_ENTITY_NAME].data.data

    return data.act


def _static_targets(env):
    """Stage targets in the current palm frame at each episode's elapsed time.

    The authors move MuJoCo target sites in the hand's parent-body frame.
    Both stationary and rotating targets therefore follow the palm pose.
    """
    import torch

    phase = getattr(env, "_rsi_static_phase", None)
    if phase is None:
        # MJLab may probe observation shapes before the initial reset event.
        return torch.stack(
            (_ball_position(env, BALL1_ENTITY_NAME),
             _ball_position(env, BALL2_ENTITY_NAME)), dim=1
        )

    _, _, _, parent_name, local_centers = reference_goal_cycle_frame()
    hand = env.scene[HAND_ENTITY_NAME]
    parent_id = getattr(env, "_rsi_target_parent_body_idx", None)
    if parent_id is None:
        parent_ids, _ = hand.find_bodies([parent_name], preserve_order=True)
        if len(parent_ids) != 1:
            raise RuntimeError(f"Cannot locate RSI palm body {parent_name!r}.")
        parent_id = int(parent_ids[0])
        env._rsi_target_parent_body_idx = parent_id

    parent_pos = hand.data.body_link_pos_w[:, parent_id]
    parent_quat = hand.data.body_link_quat_w[:, parent_id]
    centers = torch.as_tensor(
        local_centers, device=env.device, dtype=parent_pos.dtype
    )
    if ROTATION_PERIOD_SEC is not None:
        direction = env._rsi_direction.to(dtype=phase.dtype)
        elapsed_steps = env._rsi_episode_step.to(dtype=phase.dtype)
        phase = phase + direction * (2.0 * math.pi * CTRL_DT / ROTATION_PERIOD_SEC) * elapsed_steps
    dx = RSI_ELLIPSE_X_RADIUS * torch.cos(phase)
    dy = RSI_ELLIPSE_Y_RADIUS * torch.sin(phase)
    offsets = torch.stack((dx, dy, torch.zeros_like(dx)), dim=-1)
    local = torch.stack((centers[0] + offsets, centers[1] - offsets), dim=1)

    # MuJoCo and MJLab store quaternions in (w, x, y, z) order.
    quat_vec = parent_quat[:, None, 1:4].expand_as(local)
    cross = 2.0 * torch.cross(quat_vec, local, dim=-1)
    rotated = (
        local + parent_quat[:, None, 0:1] * cross
        + torch.cross(quat_vec, cross, dim=-1)
    )
    return parent_pos[:, None, :] + rotated


def baoding_obs_target1_pos(env):
    return _static_targets(env)[:, 0]


def baoding_obs_target2_pos(env):
    return _static_targets(env)[:, 1]


def baoding_obs_target1_err(env):
    return baoding_obs_target1_pos(env) - baoding_obs_ball1_pos(env)


def baoding_obs_target2_err(env):
    return baoding_obs_target2_pos(env) - baoding_obs_ball2_pos(env)


# =============================================================================
# REWARD AND TERMINATION
# =============================================================================

def baoding_ball1_dropped(env):
    return (
        baoding_obs_ball1_pos(env)[:, 2]
        < DROP_HEIGHT
    )


def baoding_ball2_dropped(env):
    return (
        baoding_obs_ball2_pos(env)[:, 2]
        < DROP_HEIGHT
    )


def baoding_any_ball_dropped(env):
    """True when either ball has crossed the drop-height threshold."""
    return (
        baoding_ball1_dropped(env)
        | baoding_ball2_dropped(env)
    )


def baoding_both_balls_held_reward(env):
    """Return 1 while both balls are retained and 0 after a drop."""
    return (~baoding_any_ball_dropped(env)).float()


def baoding_drop_indicator(env):
    """Return 1 on a dropped state so its negative config weight penalizes it."""
    return baoding_any_ball_dropped(env).float()


def baoding_static_task_reward(env):
    """Compute the dense Baoding reward before MJLab auto-resets envs."""
    import torch

    targets = _static_targets(env)
    distance1 = torch.linalg.vector_norm(
        _ball_position(env, BALL1_ENTITY_NAME) - targets[:, 0], dim=-1
    )
    distance2 = torch.linalg.vector_norm(
        _ball_position(env, BALL2_ENTITY_NAME) - targets[:, 1], dim=-1
    )
    held = ~baoding_any_ball_dropped(env)
    solved = (
        held & (distance1 < 0.015) & (distance2 < 0.015)
    )
    ball1_speed = torch.linalg.vector_norm(
        _ball_linear_velocity(env, BALL1_ENTITY_NAME) / CTRL_DT, dim=-1
    )
    ball2_speed = torch.linalg.vector_norm(
        _ball_linear_velocity(env, BALL2_ENTITY_NAME) / CTRL_DT, dim=-1
    )
    env._rsi_step_components = {
        "target1_distance": distance1,
        "target2_distance": distance2,
        "ball1_speed": ball1_speed,
        "ball2_speed": ball2_speed,
        "static_solved": solved.float(),
    }
    return -distance1 - distance2 + held.float() + 5.0 * solved.float()


# =============================================================================
# MJLAB ENVIRONMENT CONFIGURATION
# =============================================================================

def make_baoding_hold_env_cfg(
    num_envs: int,
    play: bool = False,
):
    from mjlab.actuator import (
        XmlActuatorCfg as _XmlWrappedActuatorCfg,
    )
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import (
        TerminationTermCfg,
    )
    from mjlab.sim import MujocoCfg, SimulationCfg

    from myosuite.core.config import TaskConfig
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
        MyoMuscleActivationActionCfg,
        mjlab_env_cfg_from_task_config,
    )

    del play

    task_settings = make_baoding_hold_task_settings(
        num_envs=num_envs
    )

    decimation = int(
        round(
            task_settings.ctrl_dt
            / task_settings.sim_dt
        )
    )

    if decimation <= 0:
        raise ValueError(
            "Invalid Baoding decimation from "
            f"ctrl_dt={task_settings.ctrl_dt} and "
            f"sim_dt={task_settings.sim_dt}."
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "hand_qpos": ObservationTermCfg(
                    func=baoding_obs_hand_qpos
                ),
                "ball1_pos": ObservationTermCfg(
                    func=baoding_obs_ball1_pos
                ),
                "ball1_lin_vel": ObservationTermCfg(
                    func=baoding_obs_ball1_lin_vel
                ),
                "ball2_pos": ObservationTermCfg(
                    func=baoding_obs_ball2_pos
                ),
                "ball2_lin_vel": ObservationTermCfg(
                    func=baoding_obs_ball2_lin_vel
                ),
                "target1_pos": ObservationTermCfg(
                    func=baoding_obs_target1_pos
                ),
                "target2_pos": ObservationTermCfg(
                    func=baoding_obs_target2_pos
                ),
                "target1_err": ObservationTermCfg(
                    func=baoding_obs_target1_err
                ),
                "target2_err": ObservationTermCfg(
                    func=baoding_obs_target2_err
                ),
                "activation": ObservationTermCfg(
                    func=baoding_obs_activation
                ),
            },
        ),
    }

    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=HAND_ENTITY_NAME,
            actuator_names=hand_muscle_names(),
        ),
    }

    env_cfg = mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(
            max_episode_steps=(
                task_settings.max_episode_steps
            )
        ),
        spec_fn=hand_spec_fn,
        entity_name=HAND_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=hand_tendon_names(),
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
                timestep=task_settings.sim_dt
            ),
        ),
    )

    # Reproduce the classic Baoding reset: open hand with palm facing upward.
    hand_entity_cfg = env_cfg.scene.entities[
        HAND_ENTITY_NAME
    ]
    hand_entity_cfg.init_state.joint_pos = (
        hand_initial_joint_positions()
    )
    hand_entity_cfg.init_state.joint_vel = {
        name: 0.0
        for name in hand_joint_names()
    }

    # Each ball is its own entity. This is required by recent MJLab versions,
    # where an entity may contain at most one free joint.
    env_cfg.scene.entities[BALL1_ENTITY_NAME] = EntityCfg(
        spec_fn=ball1_spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=BALL1_INITIAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )
    env_cfg.scene.entities[BALL2_ENTITY_NAME] = EntityCfg(
        spec_fn=ball2_spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=BALL2_INITIAL_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    # The default hand reset remains first; sample a new RSI phase afterwards.
    env_cfg.events["reset_baoding_rsi_phase"] = EventTermCfg(
        func=reset_baoding_balls_randomly,
        mode="reset",
    )

    env_cfg.rewards = {
        "static_task": RewardTermCfg(func=baoding_static_task_reward, weight=1.0),
    }

    # Preserve the time-limit termination created by the MyoSuite task builder
    # and add an immediate task failure when either ball drops.
    env_cfg.terminations["ball_dropped"] = (
        TerminationTermCfg(
            func=baoding_any_ball_dropped,
            time_out=False,
        )
    )

    # The paper's reward is per control step, rather than reward per second.
    # Older MJLab versions did not expose this flag and used unscaled rewards.
    if hasattr(env_cfg, "scale_rewards_by_dt"):
        env_cfg.scale_rewards_by_dt = False

    return env_cfg


# =============================================================================
# RSL-RL CONFIGURATION AND REGISTRATION
# =============================================================================

def make_baoding_hold_rl_cfg():
    """Reuse the known-working MyoSuite elbow PPO config as a starting point."""
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _elbow_ppo_runner_cfg,
    )

    return copy.deepcopy(_elbow_ppo_runner_cfg())


def validate_static_configuration() -> None:
    if NUM_ENVS <= 0:
        raise ValueError("NUM_ENVS must be positive.")

    if PLAY_NUM_ENVS <= 0:
        raise ValueError("PLAY_NUM_ENVS must be positive.")

    if MAX_EPISODE_STEPS <= 0:
        raise ValueError(
            "MAX_EPISODE_STEPS must be positive."
        )

    if not 0.0 < SIM_DT <= CTRL_DT:
        raise ValueError(
            "The time steps must satisfy "
            "0 < SIM_DT <= CTRL_DT."
        )

    ratio = CTRL_DT / SIM_DT

    if abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(
            "CTRL_DT must be an integer multiple of SIM_DT."
        )

    if DROP_HEIGHT <= 0.0:
        raise ValueError("DROP_HEIGHT must be positive.")

    if HOLD_REWARD_PER_STEP <= 0.0:
        raise ValueError(
            "HOLD_REWARD_PER_STEP must be positive."
        )

    if DROP_PENALTY < 0.0:
        raise ValueError(
            "DROP_PENALTY must be non-negative."
        )

    if not 0.0 < BALL_SIZE_SCALE <= 1.0:
        raise ValueError(
            "BALL_SIZE_SCALE must be in (0, 1]."
        )

    expected_mass_scale = BALL_SIZE_SCALE ** 3

    if abs(
        BALL_MASS_SCALE
        - expected_mass_scale
    ) > 1e-12:
        raise ValueError(
            "BALL_MASS_SCALE must equal "
            "BALL_SIZE_SCALE ** 3 to preserve density."
        )

    reset_ranges = (
        (
            "BALL_PAIR_RESET_OFFSET",
            BALL_PAIR_RESET_OFFSET_LOW,
            BALL_PAIR_RESET_OFFSET_HIGH,
        ),
        (
            "BALL_INDIVIDUAL_RESET_JITTER",
            BALL_INDIVIDUAL_RESET_JITTER_LOW,
            BALL_INDIVIDUAL_RESET_JITTER_HIGH,
        ),
    )

    for name, low, high in reset_ranges:
        if len(low) != 3 or len(high) != 3:
            raise ValueError(
                f"{name} bounds must contain exactly 3 values."
            )

        if any(
            low_value > high_value
            for low_value, high_value in zip(
                low,
                high,
                strict=True,
            )
        ):
            raise ValueError(
                f"Every lower {name} bound must be "
                "less than or equal to its upper bound."
            )

    if BALL_SEPARATION_MARGIN < 0.0:
        raise ValueError(
            "BALL_SEPARATION_MARGIN must be non-negative."
        )

    if BALL_RESET_DROP_CLEARANCE <= 0.0:
        raise ValueError(
            "BALL_RESET_DROP_CLEARANCE must be positive."
        )

    lowest_possible_reset_height = min(
        BALL1_INITIAL_POS[2],
        BALL2_INITIAL_POS[2],
    ) + BALL_PAIR_RESET_OFFSET_LOW[2] + (
        BALL_INDIVIDUAL_RESET_JITTER_LOW[2]
    )

    if lowest_possible_reset_height <= (
        DROP_HEIGHT
        + BALL_RESET_DROP_CLEARANCE
    ):
        raise ValueError(
            "The ball reset ranges do not preserve the requested "
            "clearance above DROP_HEIGHT."
        )

    if BALL1_INITIAL_POS[2] <= DROP_HEIGHT:
        raise ValueError(
            "Ball 1 must start above DROP_HEIGHT."
        )

    if BALL2_INITIAL_POS[2] <= DROP_HEIGHT:
        raise ValueError(
            "Ball 2 must start above DROP_HEIGHT."
        )

    model = hand_model()

    if model.nq != EXPECTED_HAND_QPOS_DIM:
        raise RuntimeError(
            "Expected the fixed-base MyoHand model to have "
            f"nq={EXPECTED_HAND_QPOS_DIM}, but got {model.nq}."
        )

    if model.nv != EXPECTED_HAND_QVEL_DIM:
        raise RuntimeError(
            "Expected the fixed-base MyoHand model to have "
            f"nv={EXPECTED_HAND_QVEL_DIM}, but got {model.nv}."
        )

    if len(hand_muscle_names()) != EXPECTED_ACTION_DIM:
        raise RuntimeError(
            "Expected 39 MyoHand muscle actions, but got "
            f"{len(hand_muscle_names())}."
        )

    if OBS_DIM != (
        EXPECTED_HAND_QPOS_DIM
        + 3
        + 3
        + 3
        + 3
        + 3
        + 3
        + 3
        + 3
        + EXPECTED_ACTION_DIM
    ):
        raise RuntimeError(
            "The declared Baoding observation dimension is inconsistent."
        )

    # Validate both balls against the classic MyoSuite model before registry
    # construction. This catches asset/name mismatches early.
    original_ball_properties(BALL1_ENTITY_NAME)
    original_ball_properties(BALL2_ENTITY_NAME)
    reference_goal_cycle_frame()  # Also checks spacing over the RSI orbit.


def register_baoding_hold_task(
    quiet: bool = True,
) -> None:
    """
    Register the chosen RSI stage in the process-local MJLab registry.

    Every training or visualization process should call this function before
    make_env(...).
    """
    global _TASK_CONFIGURATION_LOCKED

    validate_static_configuration()

    with silence_output(quiet):
        import myosuite  # noqa: F401

        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            bootstrap_myosuite_mjlab_registry,
        )

        bootstrap_myosuite_mjlab_registry()

    import mjlab.tasks.registry as registry

    existing_tasks = set(registry.list_tasks())

    if TASK_ID in existing_tasks:
        _TASK_CONFIGURATION_LOCKED = True
        return

    env_cfg = make_baoding_hold_env_cfg(
        num_envs=NUM_ENVS,
        play=False,
    )
    play_env_cfg = make_baoding_hold_env_cfg(
        num_envs=PLAY_NUM_ENVS,
        play=True,
    )
    rl_cfg = make_baoding_hold_rl_cfg()

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
    register_baoding_hold_task(quiet=True)

    import mjlab.tasks.registry as registry

    env_cfg = registry.load_env_cfg(TASK_ID)
    play_env_cfg = registry.load_env_cfg(
        TASK_ID,
        play=True,
    )
    rl_cfg = registry.load_rl_cfg(TASK_ID)

    settings = make_baoding_hold_task_settings(
        num_envs=1
    )

    print()
    print("Registered MJLab Baoding curriculum task")
    print("=" * 80)
    print("  task id:             ", TASK_ID)
    print("  train env cfg:       ", type(env_cfg))
    print("  play env cfg:        ", type(play_env_cfg))
    print("  RL cfg:              ", type(rl_cfg))
    print("  hand XML:            ", hand_xml_path())
    print(
        "  reference XML:       ",
        reference_baoding_xml_path(),
    )
    print(
        "  scene entities:      ",
        tuple(env_cfg.scene.entities),
    )
    print("  train num envs:      ", env_cfg.scene.num_envs)
    print("  play num envs:       ", play_env_cfg.scene.num_envs)
    print("  max episode steps:   ", settings.max_episode_steps)
    print("  sim dt:              ", settings.sim_dt)
    print("  control dt:          ", settings.ctrl_dt)
    print("  decimation:          ", env_cfg.decimation)
    print("  drop height:         ", settings.drop_height)
    print("  stage:               ", STAGE_INDEX)
    print("  rotation period:     ", ROTATION_PERIOD_SEC)
    print("  RSI placement:       ", RSI_PLACEMENT_PROBABILITY)
    print("  random task choice:  ", RANDOM_TASK_CHOICE)
    print("  ball 1 initial pos:  ", BALL1_INITIAL_POS)
    print("  ball 2 initial pos:  ", BALL2_INITIAL_POS)
    print("  ball size scale:     ", BALL_SIZE_SCALE)
    print("  ball mass scale:     ", BALL_MASS_SCALE)
    print(
        "  contact z adjustment:",
        -BALL_CONTACT_HEIGHT_COMPENSATION,
    )
    print(
        "  ball 1 radius:       ",
        scaled_ball_radius(BALL1_ENTITY_NAME),
    )
    print(
        "  ball 2 radius:       ",
        scaled_ball_radius(BALL2_ENTITY_NAME),
    )
    print(
        "  pair reset offset:   ",
        BALL_PAIR_RESET_OFFSET_LOW,
        "to",
        BALL_PAIR_RESET_OFFSET_HIGH,
    )
    print(
        "  per-ball jitter:     ",
        BALL_INDIVIDUAL_RESET_JITTER_LOW,
        "to",
        BALL_INDIVIDUAL_RESET_JITTER_HIGH,
    )
    print("  reward while held:   ", HOLD_REWARD_PER_STEP)
    print("  drop penalty:        ", DROP_PENALTY)
    print("  RSI ellipse radii:   ", RSI_ELLIPSE_X_RADIUS, RSI_ELLIPSE_Y_RADIUS)
    print("  hand joints:         ", len(hand_joint_names()))
    print("  action muscles:      ", len(hand_muscle_names()))
    print("  wrapped tendons:     ", len(hand_tendon_names()))
    print("  observation dim:     ", OBS_DIM)
    print("=" * 80)
    print()
    print("Policy observation layout:")
    print("  hand_qpos:      0:23")
    print("  ball1_pos:     23:26")
    print("  ball1_lin_vel: 26:29")
    print("  ball2_pos:     29:32")
    print("  ball2_lin_vel: 32:35")
    print("  target1_pos:   35:38")
    print("  target2_pos:   38:41")
    print("  target1_err:   41:44")
    print("  target2_err:   44:47")
    print("  activation:    47:86")
    print("  total:            86")


if __name__ == "__main__":
    main()
