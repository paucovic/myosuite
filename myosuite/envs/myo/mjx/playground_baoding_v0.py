import mujoco
from mujoco import mjx
from mujoco_playground import State
import jax
import jax.numpy as jp

from myosuite.envs.myo.mjx.mjx_base_env import MjxMyoBase


class MjxBaodingEnvV0(MjxMyoBase):

    def __init__(self, config, config_overrides=None):
        super().__init__(config, config_overrides)

        self._object1_sid_int = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "ball1_site")
        self._object2_sid_int = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "ball2_site")
        self._target1_sid_int = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "target1_site")
        self._target2_sid_int = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "target2_site")

        #self._target1_z = jp.array(self._mj_model.site_pos[self._target1_sid_int, 2])
        #self._target2_z = jp.array(self._mj_model.site_pos[self._target2_sid_int, 2])

        self._object1_sid = jp.array(self._object1_sid_int)
        self._object2_sid = jp.array(self._object2_sid_int)
        self._target1_sid = jp.array(self._target1_sid_int)
        self._target2_sid = jp.array(self._target2_sid_int)

        #print("ball1_site id:", self._object1_sid)
        #print("ball2_site id:", self._object2_sid)
        #print("target1_site id:", self._target1_sid)
        #print("target2_site id:", self._target2_sid)

    
    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng)

        # Important: use qpos0, not zeros, because the balls/free joints
        # probably need valid initial positions and quaternions.
        qpos = jp.array(self._mj_model.qpos0)
        qpos = qpos.at[:-14].set(0.0)
        qpos = qpos.at[0].set(-1.57)
        
        qvel = jp.zeros(self.mjx_model.nv)

        data = self._get_data(qpos, qvel)

        step_count = jp.array(0, dtype=jp.int32)

        object1_init_pos = qpos[-14:-11]
        object2_init_pos = qpos[-7:-4]

        target1_z = object1_init_pos[2]
        target2_z = object2_init_pos[2]

        target1_pos, target2_pos = self._get_target_positions(step_count, target1_z, target2_z)

        target1_offset = object1_init_pos - target1_pos
        target2_offset = object2_init_pos - target2_pos
        
        target1_pos = target1_pos + target1_offset
        target2_pos = target2_pos + target2_offset

        info = {
            "rng": rng,
            "step_count": step_count,
            "target1_pos": target1_pos,
            "target2_pos": target2_pos,
            "target1_z": target1_z,
            "target2_z": target2_z,
            "target1_offset": target1_offset,
            "target2_offset": target2_offset,
        }

        obs = self._get_obs(data, info)

        reward = jp.array(0.0)
        done = jp.array(0.0)

        zero = jp.array(0.0)

        metrics = {
            "pos_dist_1": zero,
            "pos_dist_2": zero,
            "max_dist": zero,
            "danger_margin": zero,

            "is_fall_1": zero,
            "is_fall_2": zero,
            "too_far_1": zero,
            "too_far_2": zero,

            "pos_reward_1": zero,
            "pos_reward_2": zero,
            "act_reg_reward": zero,
            "alive_reward": zero,
            "solved_bonus": zero,
            "drop_penalty": zero,
            "solved_frac": zero,
        }
       
        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:

        # Advance simulation one step
        state = self._step_simulation(state, action)

        # Update moving targets
        target1_pos, target2_pos = self._get_target_positions(
            state.info["step_count"],
            state.info["target1_z"],
            state.info["target2_z"],
        )

        target1_pos = target1_pos + state.info["target1_offset"]
        target2_pos = target2_pos + state.info["target2_offset"]

        state = state.replace(
            info={
                **state.info,
                "target1_pos": target1_pos,
                "target2_pos": target2_pos,
            }
        )

        #compute obs/reward/done/metrics against the current target.
        state = state.replace(obs=self._get_obs(state.data, state.info))
        state = state.replace(reward=self._get_reward(state.data, state.info))
        state = state.replace(done=self._get_done(state))
        state = state.replace(metrics={**state.metrics, **self._get_metrics(state)})

        #Handle episode
        state = state.replace(info=self._get_info(state))

        return state

    def _get_rewards(self, data: mjx.Data, info: dict) -> dict:
        b = self._get_baoding_state(data, info)

        target1_dist = jp.linalg.norm(b["target1_err"])
        target2_dist = jp.linalg.norm(b["target2_err"])

        object1_z = b["object1_pos"][2]
        object2_z = b["object2_pos"][2]

        is_fall_1 = object1_z < self._config.drop_th
        is_fall_2 = object2_z < self._config.drop_th
        is_fall = jp.logical_or(is_fall_1, is_fall_2)

        too_far_1 = target1_dist > self._config.hold_dist_th
        too_far_2 = target2_dist > self._config.hold_dist_th
        too_far = jp.logical_or(too_far_1, too_far_2)

        failure = jp.logical_or(is_fall, too_far)

        solved = (
            (target1_dist < self._config.proximity_th)
            & (target2_dist < self._config.proximity_th)
            & (~failure)
        )

        act_mag = jp.mean(jp.square(data.act))

        return {
            "pos_dist_1": -1.0 * target1_dist,
            "pos_dist_2": -1.0 * target2_dist,
            "act_reg": -0.1 * act_mag,
            "alive": 0.8 * (~failure).astype(jp.float32),
            "solved_bonus": 0.0 * solved.astype(jp.float32),
            "drop_penalty": -50.0 * failure.astype(jp.float32),
        }
  
    def _get_obs(self, data: mjx.Data, info: dict) -> dict:
        
        b = self._get_baoding_state(data, info)

        obs = jp.concatenate([b["hand_pos"], b["object1_pos"], b["object1_velp"], b["object2_pos"], b["object2_velp"], b["target1_pos"], b["target2_pos"], b["target1_err"], b["target2_err"],b["target_phase"], data.act])

        return {"state": obs}

    def _get_baoding_state(self, data: mjx.Data, info: dict) -> dict:

        hand_pos = data.qpos[:-14]

        object1_pos = data.site_xpos[self._object1_sid]
        object2_pos = data.site_xpos[self._object2_sid]

        target1_pos = info["target1_pos"]
        target2_pos = info["target2_pos"]
        
        target1_err = target1_pos - object1_pos
        target2_err = target2_pos - object2_pos

        object1_velp = data.qvel[-12:-9] * self._config.ctrl_dt
        object2_velp = data.qvel[-6:-3] * self._config.ctrl_dt

        t = info["step_count"].astype(jp.float32) * self._config.ctrl_dt
        phase = 2.0 * jp.pi * (t / self._config.goal_time_period)
        target_phase = jp.array([jp.sin(phase), jp.cos(phase)])

        return {"hand_pos": hand_pos,"target_phase": target_phase, "object1_pos": object1_pos, "object2_pos": object2_pos, "target1_pos": target1_pos, "target2_pos": target2_pos, "target1_err": target1_err, "target2_err": target2_err, "object1_velp": object1_velp, "object2_velp": object2_velp}
    
    def _get_done(self, state: State) -> float:
        b = self._get_baoding_state(state.data, state.info)

        object1_z = b["object1_pos"][2]
        object2_z = b["object2_pos"][2]

        is_fall_1 = object1_z < self._config.drop_th
        is_fall_2 = object2_z < self._config.drop_th

        dist1 = jp.linalg.norm(b["target1_err"])
        dist2 = jp.linalg.norm(b["target2_err"])

        too_far_1 = dist1 > self._config.hold_dist_th
        too_far_2 = dist2 > self._config.hold_dist_th

        done = is_fall_1 | is_fall_2 | too_far_1 | too_far_2

        return done.astype(jp.float32)

    def _get_metrics(self, state: State) -> dict:
        b = self._get_baoding_state(state.data, state.info)

        target1_dist = jp.linalg.norm(b["target1_err"])
        target2_dist = jp.linalg.norm(b["target2_err"])

        solved = (
            (target1_dist < self._config.proximity_th)
            & (target2_dist < self._config.proximity_th)
            & (state.done == 0.0)
        )

        object1_z = b["object1_pos"][2]
        object2_z = b["object2_pos"][2]

        is_fall_1 = object1_z < self._config.drop_th
        is_fall_2 = object2_z < self._config.drop_th

        too_far_1 = target1_dist > self._config.hold_dist_th
        too_far_2 = target2_dist > self._config.hold_dist_th

        rewards = self._get_rewards(state.data, state.info)

        max_dist = jp.maximum(target1_dist, target2_dist)
        danger_start = 0.65 * self._config.hold_dist_th
        danger_margin = jp.maximum(max_dist - danger_start, 0.0)

        return {
            "pos_dist_1": target1_dist,
            "pos_dist_2": target2_dist,
            "max_dist": max_dist,
            "danger_margin": danger_margin,
            "is_fall_1": is_fall_1.astype(jp.float32),
            "is_fall_2": is_fall_2.astype(jp.float32),
            "too_far_1": too_far_1.astype(jp.float32),
            "too_far_2": too_far_2.astype(jp.float32),
            "pos_reward_1": rewards["pos_dist_1"],
            "pos_reward_2": rewards["pos_dist_2"],
            "act_reg_reward": rewards["act_reg"],
            "alive_reward": rewards["alive"],
            "solved_bonus": rewards["solved_bonus"],
            "drop_penalty": rewards["drop_penalty"],
            "solved_frac": solved.astype(jp.float32) / self._config.max_episode_steps,
        }

    def _get_target_positions(self, step_count: jp.ndarray, target1_z: jp.ndarray, target2_z: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:

        sign = 0.0 #Direction of rotation. 0 means no movement, 1 means CCW, -1 means CW.

        t = step_count.astype(jp.float32) * self._config.ctrl_dt
        angle = sign * 2.0 * jp.pi * (t / self._config.goal_time_period)

        angle1 = angle + self._config.ball_1_starting_angle
        angle2 = angle + self._config.ball_2_starting_angle

        target1_xy = jp.array([
            self._config.goal_xradius * jp.cos(angle1) + self._config.center_pos[0],
            self._config.goal_yradius * jp.sin(angle1) + self._config.center_pos[1],
        ])

        target2_xy = jp.array([
            self._config.goal_xradius * jp.cos(angle2) + self._config.center_pos[0],
            self._config.goal_yradius * jp.sin(angle2) + self._config.center_pos[1],
        ])

        target1_pos = jp.array([target1_xy[0], target1_xy[1], target1_z])
        target2_pos = jp.array([target2_xy[0], target2_xy[1], target2_z])

        return target1_pos, target2_pos

    def _get_info(self, state: State) -> dict:
        done = state.done

        truncation = jp.where(
            state.info["step_count"] >= self._config.max_episode_steps,
            1.0 - done,
            jp.array(0.0),
        )

        reset_episode = jp.logical_or(done, truncation)

        step_count = jp.where(
            reset_episode,
            jp.array(0, dtype=jp.int32),
            state.info["step_count"]
        )

        rng, _ = jax.random.split(state.info["rng"])

        return {
            **state.info,
            "rng": rng,
            "step_count": step_count,
        }