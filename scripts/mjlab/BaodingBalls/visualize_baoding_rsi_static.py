#!/usr/bin/env python3
"""Replay stages 1-5 of the Baoding RSI curriculum with visible targets.

Place this file beside train_baoding_rsi_static.py and
register_baoding_rsi_static.py in scripts/mjlab/BaodingBalls/.

From the repository root, run:

    python scripts/mjlab/BaodingBalls/visualize_baoding_rsi_static.py \
        --policy-path my_baoding_rsi_static_policy.pt --eval-batch

The MJLab environment runs the actual policy and physics. A separate MyoSuite
MuJoCo model displays the observed hand and ball positions; it is never
stepped. Transparent spheres mark palm-relative targets. A green line
means the ball is within 1.5 cm of its target; red means it is farther away.
With --eval-batch, the first 64-environment batch matches the stage evaluator;
subsequent batches use fresh random seeds, phases, and tasks until the viewer
closes. The displayed environment changes between batches unless explicitly
selected with --environment-index.
Without it, each displayed episode selects a random environment from a small
batch, then resets the batch with a fresh printed seed and random phases.
Positions, target-minus-ball vectors, and distances are also shown in cm.
"""

from __future__ import annotations

import argparse
import secrets
import time
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

# Import the trainer before reading registration constants: the trainer
# configures ball dimensions and registers this task when MJLab is imported.
from train_baoding_rsi_static import (
    BALL_SIZE_SCALE,
    CURRICULUM_STAGES,
    REPO_ROOT,
    TASK_ID,
    apply_training_config,
    configure_lstm_models,
    get_deterministic_action,
    import_mjlab_tools,
    make_wrapped_env,
    redirect_output,
    set_runtime_reset_randomization,
    stage_policy_name,
    validate_lstm_runner,
)
from register_baoding_rsi_static import (
    BALL1_CLASSIC_INITIAL_POS,
    BALL2_CLASSIC_INITIAL_POS,
    CTRL_DT,
    EXPECTED_ACTION_DIM,
    EXPECTED_HAND_QPOS_DIM,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    evaluation_phase,
    hand_joint_names,
    reference_baoding_xml_path,
)


TARGET_THRESHOLD = 0.015  # metres; matches every stage's solved reward term.
NUM_VIEWER_ENVS = 8


def checkpoint_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def build_runner(device: str, seed: int, quiet: bool, num_envs: int, stage):
    set_runtime_reset_randomization(stage)
    make_env, load_rl_cfg = import_mjlab_tools(quiet=quiet)
    from rsl_rl.runners import OnPolicyRunner

    env = make_wrapped_env(
        make_env=make_env,
        env_id=TASK_ID,
        device=device,
        num_envs=num_envs,
        seed=seed,
        quiet=quiet,
        normalize_rewards=False,
        stage=stage,
    )
    try:
        rl_cfg = load_rl_cfg(TASK_ID)
        rl_cfg.obs_groups = {
            "policy": ("policy",),
            "critic": ("policy",),
            "actor": ("policy",),
        }
        train_cfg = asdict(rl_cfg)
        apply_training_config(train_cfg)
        configure_lstm_models(train_cfg)
        with redirect_output(quiet):
            runner = OnPolicyRunner(
                env=env, train_cfg=train_cfg, log_dir=None, device=device
            )
        validate_lstm_runner(runner)
        return env, runner
    except BaseException:
        env.close()
        raise


@dataclass(frozen=True)
class Frame:
    hand_qpos: np.ndarray
    ball1: np.ndarray
    ball2: np.ndarray
    target1: np.ndarray
    target2: np.ndarray
    activations: np.ndarray

    @property
    def positions(self):
        return ((self.ball1, self.target1), (self.ball2, self.target2))


def frame_from_obs(obs, env_index: int = 0) -> Frame:
    values_tensor = obs["policy"]
    if values_tensor.ndim != 2 or values_tensor.shape[1] != OBS_DIM:
        raise RuntimeError(
            f"Expected an observation with {OBS_DIM} entries, "
            f"got {tuple(values_tensor.shape)}. Use a checkpoint from this curriculum."
        )
    if not 0 <= env_index < values_tensor.shape[0]:
        raise ValueError("Selected environment is outside the observation batch.")
    values = values_tensor[env_index].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("The policy observation contains a nonfinite value.")
    # Layout is defined by register_baoding_rsi_static. These are local
    # per-environment positions after the training wrapper removes env_origins.
    return Frame(
        hand_qpos=values[0:23].copy(),
        ball1=values[23:26].copy(),
        ball2=values[29:32].copy(),
        target1=values[35:38].copy(),
        target2=values[38:41].copy(),
        activations=values[47:86].copy(),
    )


def render_model_and_addresses():
    model = mujoco.MjModel.from_xml_path(str(reference_baoding_xml_path()))

    # The original XML contains its own target sites, but their phase is
    # independent of this MJLab episode's sampled phase; hide those sites.
    for site_id in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)
        if name and "target" in name.lower():
            model.site_rgba[site_id, 3] = 0.0
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and "target" in name.lower():
            model.geom_rgba[geom_id, 3] = 0.0

    joints = hand_joint_names()
    if len(joints) != EXPECTED_HAND_QPOS_DIM:
        raise RuntimeError("Hand joint layout differs from the Baoding task.")
    hand_addresses = []
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or int(model.jnt_type[jid]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise RuntimeError(f"Missing scalar hand joint in viewer: {name}")
        hand_addresses.append(int(model.jnt_qposadr[jid]))

    ball_addresses = []
    marker_radii = []
    for name in ("ball1", "ball2"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if body_id < 0 or geom_id < 0:
            raise RuntimeError(f"Missing ball in reference XML: {name}")
        free_joints = [
            jid for jid in range(model.njnt)
            if int(model.jnt_bodyid[jid]) == body_id
            and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free_joints) != 1:
            raise RuntimeError(f"Expected one free joint on {name}.")
        ball_addresses.append(int(model.jnt_qposadr[free_joints[0]]))
        model.geom_size[geom_id, 0] *= BALL_SIZE_SCALE
        marker_radii.append(float(model.geom_size[geom_id, 0]) + 0.006)
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site"
        )
        if site_id >= 0:
            model.site_size[site_id, 0] *= BALL_SIZE_SCALE

    return model, tuple(hand_addresses), tuple(ball_addresses), tuple(marker_radii)


def mirror_frame(model, data, hand_addresses, ball_addresses, frame: Frame, t: float):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    for address, value in zip(hand_addresses, frame.hand_qpos):
        data.qpos[address] = value
    for address, position in zip(ball_addresses, (frame.ball1, frame.ball2)):
        data.qpos[address:address + 3] = position
        if np.linalg.norm(data.qpos[address + 3:address + 7]) < 1e-10:
            data.qpos[address + 3:address + 7] = (1.0, 0.0, 0.0, 0.0)
    if model.na:
        data.act[:] = 0.0
        count = min(model.na, EXPECTED_ACTION_DIM)
        data.act[:count] = frame.activations[:count]
    data.time = t
    mujoco.mj_forward(model, data)


def add_target_markers(viewer, frame: Frame, marker_radii):
    scene = viewer.user_scn
    scene.ngeom = 0
    colors = ((1.0, 0.7, 0.1, 0.32), (1.0, 0.2, 0.55, 0.32))
    identity = np.eye(3).reshape(-1)

    for (ball, target), radius, color in zip(
        frame.positions, marker_radii, colors
    ):
        if scene.ngeom + 2 > scene.maxgeom:
            raise RuntimeError("MuJoCo viewer has too few user geoms for targets.")

        marker = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            marker, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array((radius, 0.0, 0.0)), target, identity,
            np.asarray(color, dtype=np.float32),
        )
        scene.ngeom += 1

        distance = float(np.linalg.norm(target - ball))
        if distance > 1e-8:
            line = scene.geoms[scene.ngeom]
            line_color = (0.1, 1.0, 0.2, 1.0) if distance < TARGET_THRESHOLD else (
                1.0, 0.08, 0.08, 1.0
            )
            mujoco.mjv_initGeom(
                line, mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3), np.zeros(3), identity,
                np.asarray(line_color, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom=line, type=mujoco.mjtGeom.mjGEOM_LINE, width=3.0,
                from_=ball, to=target,
            )
            scene.ngeom += 1


def show_overlay(viewer, episode: int, step: int, frame: Frame) -> None:
    # set_texts is available in current MuJoCo; the terminal printout and
    # markers still work with older installations without the overlay API.
    if not hasattr(viewer, "set_texts"):
        return
    d1, d2 = (100.0 * np.linalg.norm(target - ball)
              for ball, target in frame.positions)
    overlay = (
        mujoco.mjtFontScale.mjFONTSCALE_150.value,
        mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
        "Episode\nStep\nBall 1 error\nBall 2 error\nBoth within 1.5 cm",
        f"{episode}\n{step}/{MAX_EPISODE_STEPS}\n"
        f"{d1:.2f} cm\n{d2:.2f} cm\n"
        f"{'YES' if d1 < 1.5 and d2 < 1.5 else 'NO'}",
    )
    viewer.set_texts(overlay)


def print_frame(episode: int, step: int, frame: Frame):
    print(f"episode={episode} step={step:03d}/{MAX_EPISODE_STEPS}")
    print("  ball    current xyz [m]           target xyz [m]            target - current [cm]   distance [cm]")
    for index, (ball, target) in enumerate(frame.positions, start=1):
        delta_cm = 100.0 * (target - ball)
        print(
            f"  {index}     "
            f"{np.array2string(ball, precision=3, suppress_small=True):<25}"
            f"{np.array2string(target, precision=3, suppress_small=True):<26}"
            f"{np.array2string(delta_cm, precision=2, suppress_small=True):<25}"
            f"{np.linalg.norm(delta_cm):.2f}"
        )


def stage_from_checkpoint(path: Path, override: int | None):
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    recorded = checkpoint.get("infos", {}).get("curriculum_stage", {}).get("index")
    if recorded is not None and override is not None and recorded != override:
        raise ValueError(f"Checkpoint is stage {recorded}, but --stage={override}.")
    index = override or recorded
    if index not in range(1, 6):
        raise ValueError("Cannot identify stage from checkpoint; specify --stage 1..5.")
    return CURRICULUM_STAGES[index - 1]


def print_reset(env, index: int, seed: int) -> None:
    phase = float(env.env._rsi_static_phase[index])
    direction = int(env.env._rsi_direction[index])
    placed = bool(env.env._rsi_reset_placed[index])
    task = {-1: "clockwise", 0: "hold", 1: "counterclockwise"}[direction]
    print(f"  reset seed={seed} env={index} phase={phase:.4f} rad "
          f"task={task} RSI_placed={placed}")


def next_reset_seed(rng, previous: int) -> int:
    """Choose a fresh seed instead of replaying the preceding initialization."""
    seed = rng.randrange(2**31)
    while seed == previous:
        seed = rng.randrange(2**31)
    return seed


def next_displayed_index(rng, count: int, previous: int) -> int:
    """Vary the displayed environment when more than one is available."""
    if count == 1:
        return 0
    candidate = rng.randrange(count - 1)
    return candidate + (candidate >= previous)


def main(args) -> None:
    path = checkpoint_path(args.policy_path or stage_policy_name(CURRICULUM_STAGES[0]))
    if not path.is_file():
        raise FileNotFoundError(f"Baoding checkpoint does not exist: {path}")
    if args.dt < 0 or args.print_every < 0:
        raise ValueError("--dt and --print-every must be nonnegative.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be at least 1.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device} but CUDA is unavailable.")
    stage = stage_from_checkpoint(path, args.stage)
    count = stage.eval_num_envs if args.eval_batch else args.viewer_envs
    if count < 1:
        raise ValueError("--viewer-envs must be positive.")
    if args.environment_index is not None and not 0 <= args.environment_index < count:
        raise ValueError(f"--environment-index must be in [0, {count}).")

    # An optional --seed makes the series reproducible, but every ordinary
    # episode still receives its own distinct seed and random initial phase.
    reset_rng = random.Random(args.seed) if args.seed is not None else secrets.SystemRandom()
    evaluation_seed = (args.seed or 0) + 20_000 + stage.index * 1_000
    current_seed = evaluation_seed if args.eval_batch else reset_rng.randrange(2**31)
    selection_rng = random.Random(args.seed) if args.seed is not None else secrets.SystemRandom()
    shown_index = (
        args.environment_index if args.environment_index is not None
        else selection_rng.randrange(count)
    )
    env = None
    try:
        env, runner = build_runner(
            args.device, current_seed, args.quiet, count, stage
        )
        try:
            runner.load(str(path), load_optimizer=False)
        except TypeError:
            runner.load(str(path))
        policy = runner.get_inference_policy(device=args.device)
        if args.eval_batch:
            env.env._rsi_forced_eval_phase = torch.as_tensor(
                [evaluation_phase(i, count) for i in range(count)],
                device=env.device, dtype=torch.float32,
            )
            env.env._rsi_forced_eval_direction = (
                torch.as_tensor(
                    [(i % 3) - 1 for i in range(count)],
                    device=env.device, dtype=torch.long,
                ) if stage.random_task_choice else None
            )
        # No phase override in ordinary playback: RSI samples a fresh phase.
        obs = env.reset(seed=current_seed)
        policy.reset()
        physics_dt = float(env.physics_dt)
        control_dt = physics_dt * int(env.cfg.decimation)

        model, hand_addresses, ball_addresses, marker_radii = render_model_and_addresses()
        data = mujoco.MjData(model)
        frame = frame_from_obs(obs, shown_index)
        mirror_frame(model, data, hand_addresses, ball_addresses, frame, 0.0)
        lookat = 0.5 * (
            np.asarray(BALL1_CLASSIC_INITIAL_POS)
            + np.asarray(BALL2_CLASSIC_INITIAL_POS)
        )
        print(f"Policy: {path} | stage {stage.index}: {stage.slug}")
        print(f"Environment: {TASK_ID} | device: {args.device}")
        print(f"MJLab physics dt: {physics_dt:.6f} s | control dt: {control_dt:.6f} s")
        if not np.isclose(control_dt, CTRL_DT, rtol=0, atol=1e-9):
            print(f"WARNING: control dt differs from configured {CTRL_DT:.6f} s.")
        print("Transparent yellow/pink = palm-relative targets; green/red = within/outside 1.5 cm.")
        print("MJLab runs the physics; the MuJoCo viewer mirrors observations.")
        if args.eval_batch:
            print(f"Evaluation batch: seed={evaluation_seed}, {count} environments; "
                  f"first displayed environment={shown_index} (0-based).")
        else:
            print(f"Ordinary playback: {count} environments; fresh seed and "
                  "random displayed environment each episode.")
        print_reset(env, shown_index, current_seed)
        print_frame(1, 0, frame)

        episode = 1
        episode_step = 0
        episode_return = 0.0
        solved_steps = 0
        completed_count = 0
        drop_count = 0
        batch_active = torch.ones(count, device=env.device, dtype=torch.bool)
        batch_dropped = torch.zeros_like(batch_active)
        batch_returns = torch.zeros(count, device=env.device)
        batch_solved = torch.zeros(count, device=env.device)
        batch_step = 0
        batch_number = 1
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = lookat
            viewer.cam.distance = 0.55
            add_target_markers(viewer, frame, marker_radii)
            show_overlay(viewer, episode, episode_step, frame)
            viewer.sync()
            while viewer.is_running():
                start = time.monotonic()
                with torch.no_grad():
                    action = get_deterministic_action(policy, obs)
                    obs, reward, done, info = env.step(action)
                baoding_info = info["baoding"]
                policy.reset(done)
                if args.eval_batch:
                    batch_step += 1
                    active_before = batch_active.clone()
                    batch_returns[active_before] += reward[active_before]
                    batch_solved[active_before] += baoding_info["static_solved"][active_before]
                    batch_dropped |= active_before & baoding_info["terminated"].bool()
                    batch_active &= ~done.bool()
                episode_step += 1
                episode_return += float(reward[shown_index].item())
                solved_steps += int(baoding_info["static_solved"][shown_index].item())
                completed = bool(done[shown_index].item())
                previously_shown_index = shown_index
                if completed:
                    terminated = bool(baoding_info["terminated"][shown_index].item())
                    completed_count += 1
                    drop_count += int(terminated)
                    reported_return = (
                        float(batch_returns[shown_index]) if args.eval_batch
                        else episode_return
                    )
                    reported_solved = (
                        int(batch_solved[shown_index]) if args.eval_batch
                        else solved_steps
                    )
                    print(
                        f"episode={episode} env={shown_index} seed={current_seed} "
                        f"ended ({'dropped' if terminated else 'time limit'}) "
                        f"at step={episode_step}, return={reported_return:.2f}, "
                        f"solved={reported_solved}/{MAX_EPISODE_STEPS}, "
                        f"last errors=({100*float(baoding_info['target1_distance'][shown_index]):.2f}, "
                        f"{100*float(baoding_info['target2_distance'][shown_index]):.2f}) cm"
                    )
                    if args.max_episodes is not None and episode >= args.max_episodes:
                        break
                    episode += 1
                    episode_step = 0
                    episode_return = 0.0
                    solved_steps = 0
                    if args.eval_batch:
                        # Display another first-episode environment instead of
                        # continuing to display the auto-reset of a completed one.
                        candidates = torch.nonzero(batch_active).flatten().tolist()
                        if candidates:
                            shown_index = selection_rng.choice(candidates)
                            episode_step = batch_step
                            print_reset(env, shown_index, current_seed)
                        else:
                            shown_index = -1
                    else:
                        current_seed = next_reset_seed(reset_rng, current_seed)
                        obs = env.reset(seed=current_seed)
                        policy.reset()
                        if args.environment_index is None:
                            shown_index = next_displayed_index(
                                selection_rng, count, previously_shown_index
                            )
                        print(f"Episode {episode}:")
                        print_reset(env, shown_index, current_seed)
                        print_frame(episode, 0, frame_from_obs(obs, shown_index))

                if args.eval_batch and (batch_step >= MAX_EPISODE_STEPS or not bool(batch_active.any())):
                    drops = int(batch_dropped.sum().item())
                    label = "Matching evaluation batch" if batch_number == 1 else f"Random batch {batch_number}"
                    print(
                        f"{label}: drop_fraction={drops}/{count} = {drops/count:.3f}; "
                        f"mean_reward={float(batch_returns.mean()):.2f}; "
                        f"mean_solved_fraction={float(batch_solved.mean())/MAX_EPISODE_STEPS:.3f} "
                        f"(seed {current_seed}, first episode per environment)."
                    )
                    # Later batches are exploratory; leave the fixed evaluation
                    # grid behind so every reset samples new task initializations.
                    env.env._rsi_forced_eval_phase = None
                    env.env._rsi_forced_eval_direction = None
                    batch_number += 1
                    current_seed = next_reset_seed(reset_rng, current_seed)
                    obs = env.reset(seed=current_seed)
                    policy.reset()
                    batch_active.fill_(True)
                    batch_dropped.zero_()
                    batch_returns.zero_()
                    batch_solved.zero_()
                    batch_step = 0
                    shown_index = (
                        args.environment_index if args.environment_index is not None
                        else next_displayed_index(
                            selection_rng, count, previously_shown_index
                        )
                    )
                    episode_step = 0
                    episode_return = 0.0
                    solved_steps = 0
                    print(f"Episode {episode}, batch {batch_number}:")
                    print_reset(env, shown_index, current_seed)
                    print_frame(episode, 0, frame_from_obs(obs, shown_index))
                frame = frame_from_obs(obs, shown_index)
                mirror_frame(
                    model, data, hand_addresses, ball_addresses, frame,
                    episode_step * control_dt,
                )
                add_target_markers(viewer, frame, marker_radii)
                show_overlay(viewer, episode, episode_step, frame)
                viewer.sync()
                if args.print_every and episode_step % args.print_every == 0:
                    print_frame(episode, episode_step, frame)
                pause = args.dt - (time.monotonic() - start)
                if pause > 0:
                    time.sleep(pause)
        if not args.eval_batch and completed_count:
            print(f"Shown episodes: dropped={drop_count}/{completed_count} "
                  f"({drop_count/completed_count:.3f}).")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--stage", type=int, choices=range(1, 6), default=None,
                        help="Read from checkpoint; use this for a checkpoint without stage metadata.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Reproduce a sequence of random episode seeds; batch default matches training seed 0.")
    parser.add_argument("--viewer-envs", type=int, default=NUM_VIEWER_ENVS,
                        help="Parallel environments in ordinary playback (default: 8).")
    parser.add_argument("--environment-index", "--start-phase-index", type=int,
                        default=None, dest="environment_index",
                        help="Display a specific environment (default: random each reset).")
    parser.add_argument("--eval-batch", action="store_true",
                        help="Show the exact evaluation batch first, then cycle fresh random batches until closed.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dt", type=float, default=CTRL_DT,
                        help="Minimum seconds between displayed frames; physics step remains fixed.")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
