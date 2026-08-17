from __future__ import annotations

import contextlib
import copy
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path


TASK_ID = "myoHandReachRandom-v0"

NUM_ENVS = 64
PLAY_NUM_ENVS = 1

MAX_EPISODE_STEPS = 200

# Keep this for compatibility with older visualizer/training imports.
TIP_SITE_NAME = "IFtip"

# Five fingertips, ordered to match the MJX HandReach target order.
FINGERTIP_SITE_NAMES = (
    "THtip",
    "IFtip",
    "MFtip",
    "RFtip",
    "LFtip",
)

NUM_FINGERTIPS = len(FINGERTIP_SITE_NAMES)
FINGERTIP_POS_DIM = NUM_FINGERTIPS * 3

TARGET_REACH_LOW = (-0.2, -0.6, 1.35)
TARGET_REACH_HIGH = (0.0, -0.45, 1.55)

REACH_THD = 0.05
REACH_REWARD_WEIGHT = 1.0
BONUS_WEIGHT = 4.0
PENALTY_WEIGHT = 50.0

_HAND_ENTITY_NAME = "hand"

# Keep this object alive so the temporary asset folder is not deleted
# while MuJoCo/mjlab still needs it.
_HAND_ASSET_VIEW_TMPDIR = None


@contextlib.contextmanager
def silence_output(enabled: bool = True):
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file), contextlib.redirect_stderr(file):
            yield


def make_hand_reach_task_settings(num_envs: int):
    """
    Small MyoSuite HandReach task settings object.

    Important:
    This is NOT the full mjlab environment config.
    It only stores task-specific settings such as:
        - model_path
        - target range
        - tip site name
        - reward weights
        - num_envs

    The custom wrapper in train_handreach.py will provide the active targets
    and reward. The mjlab registration only exposes observations/actions.
    """
    from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import (
        HandReachCfg,
    )

    return HandReachCfg(
        max_episode_steps=MAX_EPISODE_STEPS,
        tip_site_name=TIP_SITE_NAME,
        target_reach_range=(
            TARGET_REACH_LOW,
            TARGET_REACH_HIGH,
        ),
        reach_thd=REACH_THD,
        reach_reward_weight=REACH_REWARD_WEIGHT,
        bonus_weight=BONUS_WEIGHT,
        penalty_weight=PENALTY_WEIGHT,
        num_envs=num_envs,
    )


@lru_cache(maxsize=1)
def hand_xml_path() -> Path:
    """
    Return a HandReach XML path whose relative <include> files work.

    HandReachCfg points to something like:

        myosuite/envs/myo/assets/hand/myohand_pose.xml

    That XML includes:

        ../myo_sim/scene/myosuite_scene.xml

    But in this checkout, myo_sim is usually located at:

        myosuite/simhive/myo_sim

    So we create a temporary layout like:

        /tmp/.../assets/hand/myohand_pose.xml
        /tmp/.../assets/myo_sim/scene/myosuite_scene.xml

    Then MuJoCo can resolve the relative include correctly.
    """
    global _HAND_ASSET_VIEW_TMPDIR

    from myosuite.utils.asset_path_resolver import (
        get_sim_asset_root,
        resolve_model_xml_path,
    )

    task_settings = make_hand_reach_task_settings(num_envs=1)

    original_xml = Path(task_settings.model_path)

    if not original_xml.exists():
        original_xml = Path(resolve_model_xml_path(original_xml))

    if not original_xml.exists():
        raise FileNotFoundError(f"Could not find HandReach XML: {original_xml}")

    original_hand_dir = original_xml.parent
    original_assets_dir = original_hand_dir.parent

    expected_myo_sim_dir = original_assets_dir / "myo_sim"

    if expected_myo_sim_dir.exists():
        return original_xml

    real_myo_sim_dir = Path(get_sim_asset_root("myo_sim"))

    if not real_myo_sim_dir.exists():
        raise FileNotFoundError(f"Could not find myo_sim asset root: {real_myo_sim_dir}")

    _HAND_ASSET_VIEW_TMPDIR = tempfile.TemporaryDirectory(
        prefix="myosuite_handreach_assets_"
    )

    tmp_assets_dir = Path(_HAND_ASSET_VIEW_TMPDIR.name) / "assets"
    tmp_hand_dir = tmp_assets_dir / "hand"
    tmp_myo_sim_dir = tmp_assets_dir / "myo_sim"

    tmp_assets_dir.mkdir(parents=True, exist_ok=True)

    # Copy the hand folder instead of symlinking it. This avoids MuJoCo resolving
    # the symlink back to the original path and breaking relative includes.
    shutil.copytree(
        original_hand_dir,
        tmp_hand_dir,
        dirs_exist_ok=True,
    )

    try:
        tmp_myo_sim_dir.symlink_to(
            real_myo_sim_dir,
            target_is_directory=True,
        )
    except OSError:
        shutil.copytree(
            real_myo_sim_dir,
            tmp_myo_sim_dir,
            dirs_exist_ok=True,
        )

    patched_xml = tmp_hand_dir / original_xml.name

    if not patched_xml.exists():
        raise FileNotFoundError(f"Temporary HandReach XML was not created: {patched_xml}")

    return patched_xml


@lru_cache(maxsize=1)
def hand_model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(hand_xml_path()))


def hand_spec_fn():
    import mujoco

    return mujoco.MjSpec.from_file(str(hand_xml_path()))


@lru_cache(maxsize=1)
def hand_tendon_names() -> tuple[str, ...]:
    """
    Return the tendon names targeted by the muscle actuators.

    This mirrors the elbow registration logic:
    mjlab wraps XML tendons through XmlActuatorCfg.
    """
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
        raise RuntimeError("Could not find any tendon names for the hand actuators.")

    return tuple(names)


@lru_cache(maxsize=1)
def hand_muscle_names() -> tuple[str, ...]:
    """
    Return actuator names used by MyoMuscleActivationActionCfg.

    For MyoSuite muscle XMLs, actuator names are usually the muscle names.
    If an actuator has no name, we fall back to the tendon name without
    the '_tendon' suffix.
    """
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

        if tendon_name is None:
            continue

        names.append(tendon_name.replace("_tendon", ""))

    if len(names) == 0:
        raise RuntimeError("Could not find any actuator/muscle names for the hand.")

    return tuple(names)


@lru_cache(maxsize=None)
def hand_tip_site_id(tip_site_name: str) -> int:
    import mujoco

    model = hand_model()

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        tip_site_name,
    )

    if site_id < 0:
        available_sites = []

        for index in range(model.nsite):
            name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                index,
            )

            if name is not None:
                available_sites.append(name)

        raise RuntimeError(
            f"Could not find site {tip_site_name!r}. "
            f"Available sites: {available_sites}"
        )

    return int(site_id)


@lru_cache(maxsize=1)
def hand_tip_site_ids() -> tuple[int, ...]:
    """
    Return the five fingertip site IDs in the same order as FINGERTIP_SITE_NAMES.
    """
    return tuple(hand_tip_site_id(site_name) for site_name in FINGERTIP_SITE_NAMES)


def hand_obs_qpos(env):
    return env.scene[_HAND_ENTITY_NAME].data.joint_pos


def hand_obs_qvel(env):
    ctrl_dt = env.physics_dt * env.cfg.decimation

    return env.scene[_HAND_ENTITY_NAME].data.joint_vel * ctrl_dt


def hand_obs_tip_pos(env):
    """
    Return all five fingertip positions.

    Shape:
        [num_envs, 15]

    Layout:
        THtip xyz,
        IFtip xyz,
        MFtip xyz,
        RFtip xyz,
        LFtip xyz

    This uses the same working mjlab access path as the one-finger baseline:
        env.scene["hand"].data.data.site_xpos
    """
    data = env.scene[_HAND_ENTITY_NAME].data.data

    site_ids = hand_tip_site_ids()
    site_xpos = data.site_xpos

    tip_pos = site_xpos[:, site_ids, :]

    return tip_pos.reshape(tip_pos.shape[0], FINGERTIP_POS_DIM)


def hand_obs_target_pos_placeholder(env):
    import torch

    joint_pos = env.scene[_HAND_ENTITY_NAME].data.joint_pos

    return torch.zeros(
        (joint_pos.shape[0], FINGERTIP_POS_DIM),
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )


def hand_obs_target_err_placeholder(env):
    import torch

    joint_pos = env.scene[_HAND_ENTITY_NAME].data.joint_pos

    return torch.zeros(
        (joint_pos.shape[0], FINGERTIP_POS_DIM),
        device=joint_pos.device,
        dtype=joint_pos.dtype,
    )


def hand_obs_act(env):
    data = env.scene[_HAND_ENTITY_NAME].data.data

    return data.act


def make_hand_reach_env_cfg(
    num_envs: int,
    play: bool = False,
):
    """
    Build the full mjlab ManagerBasedRlEnvCfg.

    This is the important part:
    we do NOT register HandReachCfg directly.

    HandReachCfg is only used as task settings.
    This function wraps those settings into a real mjlab environment config,
    like the elbow registration does.
    """
    from mjlab.actuator import XmlActuatorCfg as _XmlWrappedActuatorCfg
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.sim import MujocoCfg, SimulationCfg

    from myosuite.core.config import TaskConfig
    from myosuite.envs.myo.backends.mjlab.mjlab_task_builder import (
        MyoMuscleActivationActionCfg,
        mjlab_env_cfg_from_task_config,
    )

    task_settings = make_hand_reach_task_settings(num_envs=num_envs)

    decimation = int(round(task_settings.ctrl_dt / task_settings.sim_dt))

    if decimation <= 0:
        raise ValueError(
            f"Invalid HandReach decimation from ctrl_dt={task_settings.ctrl_dt} "
            f"and sim_dt={task_settings.sim_dt}"
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=hand_obs_qpos),
                "qvel": ObservationTermCfg(func=hand_obs_qvel),
                "tip_pos": ObservationTermCfg(func=hand_obs_tip_pos),
                "target_pos": ObservationTermCfg(func=hand_obs_target_pos_placeholder),
                "target_err": ObservationTermCfg(func=hand_obs_target_err_placeholder),
                "act": ObservationTermCfg(func=hand_obs_act),
            },
        ),
    }

    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=_HAND_ENTITY_NAME,
            actuator_names=hand_muscle_names(),
        ),
    }

    return mjlab_env_cfg_from_task_config(
        cfg=TaskConfig(max_episode_steps=task_settings.max_episode_steps),
        spec_fn=hand_spec_fn,
        entity_name=_HAND_ENTITY_NAME,
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=hand_tendon_names(),
                transmission_type=TransmissionType.TENDON,
            ),
        ),
        observations=observations,
        actions=actions,
        num_envs=num_envs,
        decimation=decimation,
        sim_cfg=SimulationCfg(
            njmax=256,
            nconmax=128,
            mujoco=MujocoCfg(timestep=task_settings.sim_dt),
        ),
    )


def make_hand_reach_rl_cfg():
    """
    Reuse the working elbow PPO config for now.

    This only provides a valid RSL-RL config. We tune it inside train_handreach.py.
    """
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _elbow_ppo_runner_cfg,
    )

    return copy.deepcopy(_elbow_ppo_runner_cfg())


def register_hand_reach_task(quiet: bool = True) -> None:
    """
    Register myoHandReachRandom-v0 into the mjlab task registry.

    This registration is process-local, so every training/visualization script
    should call this before make_env(...).
    """
    with silence_output(quiet):
        import myosuite  # noqa: F401

        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            bootstrap_myosuite_mjlab_registry,
        )

        bootstrap_myosuite_mjlab_registry()

    import mjlab.tasks.registry as registry

    existing_tasks = set(registry.list_tasks())

    if TASK_ID in existing_tasks:
        return

    env_cfg = make_hand_reach_env_cfg(
        num_envs=NUM_ENVS,
        play=False,
    )

    play_env_cfg = make_hand_reach_env_cfg(
        num_envs=PLAY_NUM_ENVS,
        play=True,
    )

    rl_cfg = make_hand_reach_rl_cfg()

    registry.register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=None,
    )


def main() -> None:
    register_hand_reach_task(quiet=True)

    import mjlab.tasks.registry as registry

    print()
    print("Registered mjlab tasks containing 'Hand' or 'Reach':")
    print("=" * 80)

    found = False

    for task_id in registry.list_tasks():
        if "hand" in task_id.lower() or "reach" in task_id.lower():
            print(task_id)
            found = True

    if not found:
        print("No Hand/Reach tasks found.")

    print("=" * 80)

    env_cfg = registry.load_env_cfg(TASK_ID)
    play_env_cfg = registry.load_env_cfg(TASK_ID, play=True)
    rl_cfg = registry.load_rl_cfg(TASK_ID)

    task_settings = make_hand_reach_task_settings(num_envs=1)

    print()
    print("Loaded configs:")
    print("  task id:      ", TASK_ID)
    print("  train env cfg:", type(env_cfg))
    print("  play env cfg: ", type(play_env_cfg))
    print("  rl cfg:       ", type(rl_cfg))

    print()
    print("Important HandReach settings:")
    print("  XML:             ", hand_xml_path())
    print("  train num_envs:  ", env_cfg.scene.num_envs)
    print("  play num_envs:   ", play_env_cfg.scene.num_envs)
    print("  max steps:       ", MAX_EPISODE_STEPS)
    print("  sim dt:          ", task_settings.sim_dt)
    print("  ctrl dt:         ", task_settings.ctrl_dt)
    print("  decimation:      ", env_cfg.decimation)
    print("  fingertips:      ", FINGERTIP_SITE_NAMES)
    print("  fingertip ids:   ", hand_tip_site_ids())
    print("  action muscles:  ", len(hand_muscle_names()))
    print("  wrapped tendons: ", len(hand_tendon_names()))
    print("  target range:    ", (TARGET_REACH_LOW, TARGET_REACH_HIGH))
    print("  reach_thd:       ", REACH_THD)
    print()
    print("Expected observation layout after this register file:")
    print("  qpos        23")
    print("  qvel        23")
    print("  tip_pos     15")
    print("  target_pos  15")
    print("  target_err  15")
    print("  act         39")
    print("  total       130")


if __name__ == "__main__":
    main()