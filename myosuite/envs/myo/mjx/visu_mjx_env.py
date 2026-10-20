import time
import mujoco
import jax
import mujoco.viewer
from brax.io import model
import argparse
import os
import numpy as np

from train_jax_ppo import load_env_and_network_factory
from brax.training.agents.ppo import networks as ppo_networks


# Visualize an MJX environment interactively.
local_path = os.path.dirname(os.path.realpath(__file__))

# Print target/current angle every N viewer steps.
# Set to 1 to print every frame, or None to disable.
DEBUG_POSE_ERROR_EVERY_N_STEPS = 25

# Print HandReach target/fingertip debug every N viewer steps.
# Set to 1 to print every frame, or None to disable.
DEBUG_REACH_EVERY_N_STEPS = 10

# Match the training/evaluation contact-buffer conditions.
# This does NOT show 64 hands in the viewer. It only sizes MJX/Warp internals
# the same way as training and compare_handreach_policies.py.
CONTACT_BUFFER_NUM_ENVS = 64

# Use the same seed convention as compare_handreach_policies.py.
VISU_SEED = 0


def hide_default_pose_target_markers(m):
    """
    Hide static target/goal markers that are misleading for pose tasks.
    Pose tasks use joint-angle targets, not spatial targets.
    """
    for site_id in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, site_id)
        name = name or ""

        if "target" in name.lower() or "goal" in name.lower():
            m.site_rgba[site_id, 3] = 0.0

    for geom_id in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        name = name or ""

        if "target" in name.lower() or "goal" in name.lower():
            m.geom_rgba[geom_id, 3] = 0.0


def get_pose_target_qpos_indices(env, m):
    indices = []

    if not hasattr(env._config, "target_jnt_range"):
        return None

    for joint_name in env._config.target_jnt_range.keys():
        joint_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            print(f"Warning: target joint not found in model: {joint_name}")
            continue

        qpos_index = int(m.jnt_qposadr[joint_id])
        indices.append(qpos_index)

    if len(indices) == 0:
        return None

    return np.array(indices, dtype=np.int32)


def choose_pose_marker_site_id(m):
    preferred_keywords = [
        "hand",
        "wrist",
        "tip",
        "end",
        "endpoint",
        "distal",
    ]

    forbidden_keywords = [
        "target",
        "goal",
    ]

    for site_id in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, site_id)
        name = name or ""
        lname = name.lower()

        if any(word in lname for word in forbidden_keywords):
            continue

        if any(word in lname for word in preferred_keywords):
            return site_id

    return None


def choose_pose_marker_geom_id(m, d):
    root_pos = np.array(d.xpos[0], dtype=np.float64)

    best_geom_id = None
    best_distance = -np.inf

    forbidden_keywords = [
        "floor",
        "ground",
        "target",
        "goal",
    ]

    for geom_id in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        name = name or ""
        lname = name.lower()

        if any(word in lname for word in forbidden_keywords):
            continue

        if m.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        geom_pos = np.array(d.geom_xpos[geom_id], dtype=np.float64)
        distance = float(np.linalg.norm(geom_pos - root_pos))

        if distance > best_distance:
            best_distance = distance
            best_geom_id = geom_id

    return best_geom_id


def get_marker_position_from_data(m, d, marker_site_id, marker_geom_id):
    if marker_site_id is not None:
        return np.array(d.site_xpos[marker_site_id], dtype=np.float64)

    if marker_geom_id is not None:
        return np.array(d.geom_xpos[marker_geom_id], dtype=np.float64)

    return np.array(d.xpos[0], dtype=np.float64)


def compute_pose_target_marker_position(
    m,
    target_d,
    base_qpos,
    target_angles,
    target_qpos_indices,
    marker_site_id,
    marker_geom_id,
):
    target_qpos = np.asarray(base_qpos, dtype=np.float64).copy()
    target_angles = np.asarray(target_angles, dtype=np.float64).reshape(-1)

    if (
        target_qpos_indices is not None
        and len(target_qpos_indices) == len(target_angles)
    ):
        target_qpos[target_qpos_indices] = target_angles
    else:
        n = min(len(target_angles), m.nq)
        target_qpos[:n] = target_angles[:n]

    target_d.qpos[:] = target_qpos
    target_d.qvel[:] = 0.0

    mujoco.mj_forward(m, target_d)

    return get_marker_position_from_data(
        m=m,
        d=target_d,
        marker_site_id=marker_site_id,
        marker_geom_id=marker_geom_id,
    )


def get_actual_endpoint_position(m, d, marker_site_id, marker_geom_id):
    return get_marker_position_from_data(
        m=m,
        d=d,
        marker_site_id=marker_site_id,
        marker_geom_id=marker_geom_id,
    )


def add_sphere_marker(viewer, position, radius=0.035, rgba=None):
    scn = viewer.user_scn

    if scn.ngeom >= scn.maxgeom:
        return

    geom = scn.geoms[scn.ngeom]

    size = np.array([radius, 0.0, 0.0], dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)

    if rgba is None:
        rgba = np.array([0.0, 1.0, 0.0, 0.65], dtype=np.float32)
    else:
        rgba = np.array(rgba, dtype=np.float32)

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        position,
        mat,
        rgba,
    )

    scn.ngeom += 1


def print_pose_error_if_needed(
    viewer_step_idx,
    state,
    target_qpos_indices,
):
    if DEBUG_POSE_ERROR_EVERY_N_STEPS is None:
        return

    if viewer_step_idx % DEBUG_POSE_ERROR_EVERY_N_STEPS != 0:
        return

    if "target_angles" not in state.info:
        return

    target_angles = np.asarray(
        state.info["target_angles"],
        dtype=np.float64,
    ).reshape(-1)

    qpos = np.asarray(
        state.data.qpos,
        dtype=np.float64,
    ).reshape(-1)

    if (
        target_qpos_indices is not None
        and len(target_qpos_indices) == len(target_angles)
    ):
        current_angles = qpos[target_qpos_indices]
    else:
        n = min(len(target_angles), len(qpos))
        target_angles = target_angles[:n]
        current_angles = qpos[:n]

    error = target_angles - current_angles

    print(
        "\nviewer_step:",
        viewer_step_idx,
        "env_step_count:",
        int(np.asarray(state.info["step_count"])),
        "\ntarget rad:",
        target_angles,
        "\ncurrent rad:",
        current_angles,
        "\nerror rad:",
        error,
        "\nerror norm rad:",
        float(np.linalg.norm(error)),
        "\ntarget deg:",
        np.rad2deg(target_angles),
        "\ncurrent deg:",
        np.rad2deg(current_angles),
        "\nerror deg:",
        np.rad2deg(error),
    )


def print_reach_error_if_needed(viewer_step_idx, env, state):
    if DEBUG_REACH_EVERY_N_STEPS is None:
        return

    if viewer_step_idx % DEBUG_REACH_EVERY_N_STEPS != 0:
        return

    if "targets" not in state.info:
        return

    if not hasattr(env, "_tip_sids"):
        return

    tip_pos = np.asarray(state.data.site_xpos[env._tip_sids])
    targets = np.asarray(state.info["targets"])
    err = targets - tip_pos

    combined_dist = float(np.linalg.norm(err.reshape(-1)))
    per_tip_dist = np.linalg.norm(err, axis=1)

    n_targets = len(targets)
    effective_far_th = float(env._config.far_th * n_targets)
    near_th = float(n_targets * 0.0125)

    print("\n--- HandReach debug ---")
    print("viewer_step:", viewer_step_idx)
    print("env_step_count:", int(np.asarray(state.info["step_count"])))
    print("done:", float(np.asarray(state.done)))
    print("reward:", float(np.asarray(state.reward)))
    print("combined reach_dist:", combined_dist)
    print("effective far_th:", effective_far_th)
    print("near_th:", near_th)
    print("per-tip distances:", per_tip_dist)

    names = list(env._config.target_reach_range.keys())

    for name, target, tip, dist in zip(names, targets, tip_pos, per_tip_dist):
        print(f"  {name}: dist={dist:.5f}, target={target}, tip={tip}")


def main(env_name, load_policy, use_reset):
    env, _, network_factory = load_env_and_network_factory(
        env_name,
        "warp",
        config_overrides={"num_envs": CONTACT_BUFFER_NUM_ENVS},
    )

    policy_fn = None

    if load_policy:
        ppo_network = network_factory(
            env.observation_size,
            env.action_size,
        )

        params = model.load_params(local_path + "/" + "VISU_policy.pickle")

        make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
        inference_fn = make_inference_fn(params, deterministic=True)

        @jax.jit
        def policy_fn(obs, rng):
            action, _ = inference_fn(obs, rng)
            return action

    spec = mujoco.MjSpec.from_file(env.xml_path)
    spec = env.preprocess_spec(spec)

    spec.add_sensor(
        name="reward+target",
        type=mujoco.mjtSensor.mjSENS_USER,
        dim=1 + env.mjx_model.nq,
    )

    m = spec.compile()
    d = mujoco.MjData(m)

    target_d = mujoco.MjData(m)

    target_qpos_indices = get_pose_target_qpos_indices(env, m)
    pose_marker_site_id = choose_pose_marker_site_id(m)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    key = jax.random.PRNGKey(VISU_SEED)
    key, reset_key = jax.random.split(key)
    state = jit_reset(reset_key)

    is_pose_env = "target_angles" in state.info

    d.qpos[:] = np.asarray(state.data.qpos)
    d.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(m, d)

    pose_marker_geom_id = None

    if is_pose_env:
        hide_default_pose_target_markers(m)

        if pose_marker_site_id is not None:
            marker_site_name = mujoco.mj_id2name(
                m,
                mujoco.mjtObj.mjOBJ_SITE,
                pose_marker_site_id,
            )
            print(f"Pose endpoint marker site: {marker_site_name}")
        else:
            pose_marker_geom_id = choose_pose_marker_geom_id(m, d)

            if pose_marker_geom_id is not None:
                marker_geom_name = mujoco.mj_id2name(
                    m,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    pose_marker_geom_id,
                )
                print("No distal pose marker site found.")
                print(f"Using fallback geom as endpoint: {marker_geom_name}")
            else:
                print("No distal pose marker site or geom found. Falling back to root.")

        print(f"Pose target qpos indices: {target_qpos_indices}")
        print("Green marker = target endpoint")
        print("Red marker   = actual current endpoint")
        print("Angle debug  = printed every", DEBUG_POSE_ERROR_EVERY_N_STEPS, "viewer steps")

    if "targets" in state.info:
        print("HandReach debug = printed every", DEBUG_REACH_EVERY_N_STEPS, "viewer steps")
        print("If done is repeatedly 1.0, the target shuffling is caused by early done/reset.")

    viewer_step_idx = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if load_policy:
                key, policy_key = jax.random.split(key)
                actions = policy_fn(state.obs, policy_key)
            else:
                actions = d.ctrl

            state = state.replace(
                data=state.data.replace(
                    mocap_pos=d.mocap_pos,
                    xfrc_applied=d.xfrc_applied,
                )
            )

            prev_step_count = (
                int(np.asarray(state.info["step_count"]))
                if "step_count" in state.info
                else -1
            )

            state = jit_step(state, actions)

            done_now = float(np.asarray(state.done)) > 0.5

            truncated_now = (
                "step_count" in state.info
                and int(np.asarray(state.info["step_count"])) == 0
                and prev_step_count >= env._config.max_episode_steps - 1
                and not done_now
            )

            episode_finished = done_now or truncated_now

            if use_reset and episode_finished:
                if done_now:
                    print("\nEpisode ended by done. Full visualizer reset.")
                elif truncated_now:
                    print("\nEpisode ended by max_episode_steps. Full visualizer reset.")

                key, reset_key = jax.random.split(key)
                state = jit_reset(reset_key)

            print_pose_error_if_needed(
                viewer_step_idx=viewer_step_idx,
                state=state,
                target_qpos_indices=target_qpos_indices,
            )

            print_reach_error_if_needed(
                viewer_step_idx=viewer_step_idx,
                env=env,
                state=state,
            )

            d.sensordata[0] = float(np.asarray(state.reward))

            if "target_angles" in state.info:
                target_angles_np = np.asarray(state.info["target_angles"])
                n = min(len(target_angles_np), len(d.sensordata) - 1)
                d.sensordata[1:1 + n] = target_angles_np[:n]
            else:
                qpos_np = np.asarray(state.data.qpos)
                n = min(len(qpos_np), len(d.sensordata) - 1)
                d.sensordata[1:1 + n] = qpos_np[:n]

            if load_policy:
                d.ctrl[:] = np.asarray(state.data.ctrl)

            d.qpos[:] = np.asarray(state.data.qpos)
            d.qvel[:] = np.asarray(state.data.qvel)

            if "targets" in state.info:
                for site_id, target_pos in zip(
                    env._target_sids,
                    state.info["targets"],
                ):
                    m.site_pos[site_id] = np.asarray(target_pos)

            mujoco.mj_forward(m, d)

            viewer.user_scn.ngeom = 0

            if "target_angles" in state.info:
                target_marker_pos = compute_pose_target_marker_position(
                    m=m,
                    target_d=target_d,
                    base_qpos=state.data.qpos,
                    target_angles=state.info["target_angles"],
                    target_qpos_indices=target_qpos_indices,
                    marker_site_id=pose_marker_site_id,
                    marker_geom_id=pose_marker_geom_id,
                )

                actual_marker_pos = get_actual_endpoint_position(
                    m=m,
                    d=d,
                    marker_site_id=pose_marker_site_id,
                    marker_geom_id=pose_marker_geom_id,
                )

                add_sphere_marker(
                    viewer=viewer,
                    position=target_marker_pos,
                    radius=0.035,
                    rgba=[0.0, 1.0, 0.0, 0.65],
                )

                add_sphere_marker(
                    viewer=viewer,
                    position=actual_marker_pos,
                    radius=0.025,
                    rgba=[1.0, 0.0, 0.0, 0.65],
                )

            viewer.sync()

            viewer_step_idx += 1

            time_until_next_step = env._config.ctrl_dt - (time.time() - step_start)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MJX environment")

    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxFingerPoseRandom-v0",
    )

    parser.add_argument(
        "--load_policy",
        action="store_true",
    )

    parser.add_argument(
        "--use_reset",
        action="store_true",
    )

    args = parser.parse_args()
    main(args.env_name, args.load_policy, args.use_reset)