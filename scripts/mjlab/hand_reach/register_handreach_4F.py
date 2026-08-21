from __future__ import annotations

import contextlib
import copy
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path


TASK_ID = "myoHandReach4F-v0"

NUM_ENVS = 64
PLAY_NUM_ENVS = 1

MAX_EPISODE_STEPS = 200

# The order here defines the observation order everywhere else:
# THtip xyz, followed by IFtip xyz, MFtip xyz, and RFtip xyz.
FINGERTIP_SITE_NAMES = (
    "THtip",
    "IFtip",
    "MFtip",
    "RFtip",
)

NUM_FINGERTIPS = len(FINGERTIP_SITE_NAMES)
FINGERTIP_POS_DIM = NUM_FINGERTIPS * 3

# HandReachCfg requires one tip site. The custom training wrapper will calculate
# the actual four-finger targets, errors, rewards, and solved conditions.
TIP_SITE_NAME = "IFtip"

TARGET_REACH_LOW = (-0.2, -0.6, 1.35)
TARGET_REACH_HIGH = (0.0, -0.45, 1.55)

REACH_THD = 0.05
REACH_REWARD_WEIGHT = 1.0
BONUS_WEIGHT = 4.0
PENALTY_WEIGHT = 50.0

_HAND_ENTITY_NAME = "hand"

# This object must remain alive while MuJoCo uses the temporary assets.
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
        raise FileNotFoundError(
            f"Could not find HandReach XML: {original_xml}"
        )

    original_hand_dir = original_xml.parent
    original_assets_dir = original_hand_dir.parent

    expected_myo_sim_dir = original_assets_dir / "myo_sim"

    if expected_myo_sim_dir.exists():
        return original_xml

    real_myo_sim_dir = Path(get_sim_asset_root("myo_sim"))

    if not real_myo_sim_dir.exists():
        raise FileNotFoundError(
            f"Could not find myo_sim asset root: {real_myo_sim_dir}"
        )

    _HAND_ASSET_VIEW_TMPDIR = tempfile.TemporaryDirectory(
        prefix="myosuite_handreach_4f_assets_"
    )

    tmp_assets_dir = Path(_HAND_ASSET_VIEW_TMPDIR.name) / "assets"
    tmp_hand_dir = tmp_assets_dir / "hand"
    tmp_myo_sim_dir = tmp_assets_dir / "myo_sim"

    tmp_assets_dir.mkdir(parents=True, exist_ok=True)

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
        raise FileNotFoundError(
            f"Temporary HandReach XML was not created: {patched_xml}"
        )

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
            "Could not find any tendon names for the hand actuators."
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

        if tendon_name is None:
            continue

        names.append(tendon_name.replace("_tendon", ""))

    if len(names) == 0:
        raise RuntimeError(
            "Could not find any actuator/muscle names for the hand."
        )

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
    return tuple(
        hand_tip_site_id(site_name)
        for site_name in FINGERTIP_SITE_NAMES
    )


def hand_obs_qpos(env):
    return env.scene[_HAND_ENTITY_NAME].data.joint_pos


def hand_obs_qvel(env):
    ctrl_dt = env.physics_dt * env.cfg.decimation

    return env.scene[_HAND_ENTITY_NAME].data.joint_vel * ctrl_dt


def hand_obs_tip_pos(env):
    import torch

    data = env.scene[_HAND_ENTITY_NAME].data.data

    fingertip_positions = [
        data.site_xpos[:, site_id, :]
        for site_id in hand_tip_site_ids()
    ]

    tip_pos = torch.cat(fingertip_positions, dim=-1)

    if tip_pos.shape[-1] != FINGERTIP_POS_DIM:
        raise RuntimeError(
            "Unexpected fingertip observation dimension: "
            f"expected {FINGERTIP_POS_DIM}, got {tip_pos.shape[-1]}."
        )

    return tip_pos


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

    del play

    task_settings = make_hand_reach_task_settings(num_envs=num_envs)

    decimation = int(
        round(task_settings.ctrl_dt / task_settings.sim_dt)
    )

    if decimation <= 0:
        raise ValueError(
            "Invalid HandReach decimation from "
            f"ctrl_dt={task_settings.ctrl_dt} "
            f"and sim_dt={task_settings.sim_dt}"
        )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(
                    func=hand_obs_qpos
                ),
                "qvel": ObservationTermCfg(
                    func=hand_obs_qvel
                ),
                "tip_pos": ObservationTermCfg(
                    func=hand_obs_tip_pos
                ),
                "target_pos": ObservationTermCfg(
                    func=hand_obs_target_pos_placeholder
                ),
                "target_err": ObservationTermCfg(
                    func=hand_obs_target_err_placeholder
                ),
                "act": ObservationTermCfg(
                    func=hand_obs_act
                ),
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
        cfg=TaskConfig(
            max_episode_steps=task_settings.max_episode_steps
        ),
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
            mujoco=MujocoCfg(
                timestep=task_settings.sim_dt
            ),
        ),
    )


def make_hand_reach_rl_cfg():
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _elbow_ppo_runner_cfg,
    )

    return copy.deepcopy(_elbow_ppo_runner_cfg())


def register_hand_reach_task(
    quiet: bool = True,
) -> None:
    if NUM_FINGERTIPS != 4:
        raise ValueError(
            "register_handreach_4F.py expects exactly four fingertips."
        )

    if FINGERTIP_POS_DIM != 12:
        raise ValueError(
            "Four xyz fingertip positions must have dimension 12."
        )

    # Validate all four MuJoCo site names before registering the environment.
    hand_tip_site_ids()

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

    env_cfg = registry.load_env_cfg(TASK_ID)
    play_env_cfg = registry.load_env_cfg(
        TASK_ID,
        play=True,
    )
    rl_cfg = registry.load_rl_cfg(TASK_ID)

    task_settings = make_hand_reach_task_settings(num_envs=1)

    print()
    print("Registered four-finger HandReach task")
    print("=" * 80)
    print("  task id:           ", TASK_ID)
    print("  train env cfg:     ", type(env_cfg))
    print("  play env cfg:      ", type(play_env_cfg))
    print("  RL cfg:            ", type(rl_cfg))
    print("  XML:               ", hand_xml_path())
    print("  train num envs:    ", env_cfg.scene.num_envs)
    print("  play num envs:     ", play_env_cfg.scene.num_envs)
    print("  max steps:         ", MAX_EPISODE_STEPS)
    print("  sim dt:            ", task_settings.sim_dt)
    print("  control dt:        ", task_settings.ctrl_dt)
    print("  decimation:        ", env_cfg.decimation)
    print("  fingertip sites:   ", FINGERTIP_SITE_NAMES)
    print("  fingertip IDs:     ", hand_tip_site_ids())
    print("  fingertip count:   ", NUM_FINGERTIPS)
    print("  fingertip dimension:", FINGERTIP_POS_DIM)
    print("  action muscles:    ", len(hand_muscle_names()))
    print("  wrapped tendons:   ", len(hand_tendon_names()))
    print(
        "  target range:      ",
        (TARGET_REACH_LOW, TARGET_REACH_HIGH),
    )
    print("  reach threshold:   ", REACH_THD)
    print("=" * 80)
    print()
    print("Expected policy observation layout:")
    print("  qpos:        0:23")
    print("  qvel:       23:46")
    print("  tip_pos:    46:58")
    print("  target_pos: 58:70")
    print("  target_err: 70:82")
    print("  act:        82:121")
    print("  total dimension: 121")


if __name__ == "__main__":
    main()