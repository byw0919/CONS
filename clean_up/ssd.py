"""Wrapper around Sequential Social Dilemma environment."""

from clean_up import maps
from clean_up.cleanup import CleanupEnv
import numpy as np
from gym import spaces
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace


class Env(object):

    def __init__(self, config_env):

        self.name = 'ssd'
        self.config = config_env
        self.dim_obs = [self.config.obs_height,
                        self.config.obs_width, 3]
        self._max_steps = self.config.max_steps

        self.cleaning_penalty = self.config.cleaning_penalty
        # Original space (not necessarily in this order, see
        # the original ssd files):
        # no-op, up, down, left, right, turn-ccw, turn-cw, penalty, clean
        if (self.config.disable_left_right_action and
                self.config.disable_rotation_action):
            self.l_action = 4
            self.cleaning_action_idx = 3
            # up, down, no-op, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 8}
        elif self.config.disable_left_right_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # up, down, no-op, rotate cw, rotate ccw, clean
            self.map_to_orig = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8}
        elif self.config.disable_rotation_action:
            self.l_action = 6
            self.cleaning_action_idx = 5
            # left, right, up, down, no-op, clean
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8}
        else:  # full action space except penalty beam
            self.l_action = 8
            self.cleaning_action_idx = 7
            self.map_to_orig = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 8}

        self.obs_cleaned_1hot = self.config.obs_cleaned_1hot

        self.n_agents = self.config.n_agents

        if self.config.map_name == 'cleanup_small_sym':
            ascii_map = maps.CLEANUP_SMALL_SYM
        elif self.config.map_name == 'cleanup_10x10_sym':
            ascii_map = maps.CLEANUP_10x10_SYM
        elif self.config.map_name == 'cleanup_large_sym':
            ascii_map = maps.CLEANUP_LARGE_SYM

        self.env = CleanupEnv(ascii_map=ascii_map,
                              num_agents=self.n_agents, render=False,
                              shuffle_spawn=self.config.shuffle_spawn,
                              global_ref_point=self.config.global_ref_point,
                              view_size=self.config.view_size,
                              random_orientation=self.config.random_orientation,
                              cleanup_params=self.config.cleanup_params,
                              beam_width=self.config.beam_width,
                              config=self.config)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(self.l_action)] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([self.env.observation_space] * self.n_agents)
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render.modes": []}
        self.seed()

        # length of action input to learned reward function
        if self.config.obs_cleaned_1hot:
            self.l_action_for_r = 2
        else:
            self.l_action_for_r = self.l_action

        self.steps = 0

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.l_action
        env_info["n_agents"] = self.n_agents
        env_info["state_shape"] = self.dim_obs[0] * self.dim_obs[1] * self.dim_obs[2] * self.n_agents
        env_info["obs_shape"] = self.dim_obs[0] * self.dim_obs[1] * self.dim_obs[2]
        env_info["episode_limit"] = self._max_steps
        return env_info

    def get_obs(self):
        return self.process_obs(self.env.get_agent_obs())

    def get_state(self):
        _all_obs = self.process_obs(self.env.get_agent_obs())
        state = np.array(_all_obs).flatten()
        return state

    def get_avail_agent_actions(self, i):
        return [1] * (self.l_action)

    def process_obs(self, obs_list):
        return [(obs / 256.0).tolist() for obs in obs_list]

    def seed(self, seed=None):
        self.env.seed(seed)

    def reset(self):
        """Resets the environemnt.
        Returns:
            List of agent observations
        """
        obs = self.env.reset()
        self.steps = 0

        return self.process_obs(obs)

    def step(self, actions):
        """Takes a step in env.

        Args:
            actions: list of integers
        Returns:
            List of observations, list of rewards, done, info
        """
        actions = [self.map_to_orig[a] for a in actions]
        actions_dict = {'agent-%d' % idx: actions[idx]
                        for idx in range(self.n_agents)}

        # all objects returned by env.step are dicts
        obs_next, rewards, dones, info = self.env.step(actions_dict)  # rewards是每个智能体的奖励
        self.steps += 1

        obs_next = self.process_obs(obs_next)
        rewards = list(rewards.values())

        if self.steps == self._max_steps:
            done = [True] * self.n_agents
        else:
            done = list(dones.values())[0: -1]
        rewards = [sum(rewards)] * self.n_agents

        return obs_next, rewards, done, info

    def render(self):
        self.env.render()

    def rateofwaste(self):
        rateofwaste = self.env.compute_amountofwaste()
        if rateofwaste == 1:
            rateofwaste = 0.99
        return rateofwaste