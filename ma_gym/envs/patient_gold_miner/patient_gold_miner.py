import copy
import itertools
import logging
from typing import List, Tuple, Union

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

Coordinates = Tuple[int, int]


class Agent:

    def __init__(self, id: int, pos: Coordinates, score: list):
        self.id = id
        self.pos = pos
        self.score = score

class Gold:

    def __init__(self, id: int, pos: Coordinates, cap_num: list):
        self.id = id
        self.pos = pos
        self.cap_num = cap_num

class Stone:

    def __init__(self, id: int, pos: Coordinates, cap_num: list):
        self.id = id
        self.pos = pos
        self.cap_num = cap_num

class PGM(gym.Env):
    """Patient Gold Miner"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape: Coordinates = (8, 9), n_agents: int = 3, n_golds: int = 1,
                 n_stones: int = 2, agent_view: Tuple[int, int] = (1, 2), full_observable: bool = False,
                 step_cost: float = -0.2, stone_reward: float = 0.3, stone_disappear_steps: int = 8,
                 gold_punish_steps: int = 8, gold_base_punish: float = -1, gold_reward: float = 20,
                 max_steps: int = 25):
        assert 0 < n_agents
        assert n_agents + n_golds + n_stones <= np.prod(grid_shape)
        assert 1 <= agent_view[0] <= grid_shape[0] and 1 <= agent_view[1] <= grid_shape[1]

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_golds = n_golds
        self._n_stones = n_stones
        self._agent_view = agent_view  # observation view, (horizontal, vertical)
        self.full_observable = full_observable
        self._step_cost = step_cost
        self._stone_reward = stone_reward
        self._stone_disappear_steps = stone_disappear_steps
        self._gold_punish_steps = gold_punish_steps
        self._gold_base_punish = gold_base_punish
        self._gold_reward = gold_reward

        self._max_steps = max_steps
        self.steps_beyond_done = 0
        self.seed()

        self._agents = []  # List[Agent]
        self._golds = []  # List[Gold]
        self._stones = []  # List[Stone]

        # Relative coordinates refer to the coordinates in non pad grid. These are the only
        # coordinates visible to user. Extended coordinates refer to the coordinates in pad grid.
        self._agent_map = None
        self._gold_map = None
        self._stone_map = None
        self._total_episode_reward = None
        self._agent_dones = None
        self.ag_gold = [0] * self.n_agents
        self.ag_stone = [0] * self.n_agents

        mask_size = np.prod(tuple(2 * v + 1 for v in self._agent_view))

        self._obs_len = (2 + 1 + 3 * mask_size)

        obs_high = np.array([1.] * self._obs_len, dtype=np.float32)
        obs_low = np.array([0.] * self._obs_len, dtype=np.float32)
        if self.full_observable:
            obs_high = np.tile(obs_high, self.n_agents)
            obs_low = np.tile(obs_low, self.n_agents)
        self.action_space = MultiAgentActionSpace([spaces.Discrete(len(ACTIONS_IDS))] * self.n_agents)
        self.observation_space = MultiAgentObservationSpace([spaces.Box(obs_low, obs_high)] * self.n_agents)

        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        self._viewer = None

    def get_env_info(self):
        env_info = {}
        env_info["n_actions"] = self.action_space[0].n
        env_info["n_agents"] = self.n_agents
        env_info["state_shape"] = self.observation_space[0].shape[0] * self.n_agents
        env_info["obs_shape"] = self.observation_space[0].shape[0]
        env_info["episode_limit"] = self._max_steps
        return env_info

    def get_obs(self):
        return self.get_agent_obs()

    def get_state(self):
        _all_obs = self.get_agent_obs()
        state = np.array(_all_obs).flatten()
        return state

    def get_avail_agent_actions(self, i):
        return [1] * (self.action_space[0].n)

    def reset(self) -> List[List[float]]:
        self._init_episode()
        self._step_count = 0
        self._total_episode_reward = np.zeros(self.n_agents)
        self._agent_dones = [False] * self.n_agents
        self.steps_beyond_done = 0
        self.ag_gold = [0] * self.n_agents
        self.ag_stone = [0] * self.n_agents

        return self.get_agent_obs()

    def _init_episode(self):
        """Initialize environment for new episode.

        Fills `self._agents`, self._agent_map` and `self._gold_map` and `self._stone_map` with new values.
        """
        init_positions = self._generate_init_pos()
        agent_id, gold_id, stone_id = 0, self.n_agents, self.n_agents + self._n_golds
        self._agents = []
        self._golds = []  # List[Gold]
        self._stones = []  # List[Stone]
        self._agent_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
            self.n_agents
        ), dtype=np.int32)
        self._gold_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)
        self._stone_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)

        for pos, cell in np.ndenumerate(init_positions):
            pos = self._to_extended_coordinates(pos)
            if cell == PRE_IDS['agent']:
                self._agent_map[pos[0], pos[1], agent_id] = 1
                self._agents.append(Agent(agent_id, pos=pos, score=[False] * self._n_golds))
                agent_id += 1
            elif cell == PRE_IDS['gold']:
                self._gold_map[pos] = 1
                self._golds.append(Gold(gold_id, pos=pos, cap_num=[0] * self.n_agents))
                gold_id += 1
            elif cell == PRE_IDS['stone']:
                self._stone_map[pos] = 1
                self._stones.append(Stone(stone_id, pos=pos, cap_num=[0] * self.n_agents))
                stone_id += 1

    def _to_extended_coordinates(self, relative_coordinates):
        """Translate relative coordinates in to the extended coordinates."""
        return relative_coordinates[0] + self._agent_view[0], relative_coordinates[1] + self._agent_view[1]

    def _to_relative_coordinates(self, extended_coordinates):
        """Translate extended coordinates in to the relative coordinates."""
        return extended_coordinates[0] - self._agent_view[0], extended_coordinates[1] - self._agent_view[1]

    def _generate_init_pos(self) -> np.ndarray:
        """Returns randomly selected initial positions for agents, golds and stones in relative coordinates.

        No agent or golds or stones share the same cell in initial positions.
        """
        init_pos = np.array(
            [PRE_IDS['agent']] * self.n_agents +
            [PRE_IDS['gold']] * self._n_golds +
            [PRE_IDS['stone']] * self._n_stones +
            [PRE_IDS['empty']] * (np.prod(self._grid_shape) - self.n_agents - self._n_golds - self._n_stones)
        )
        self.np_random.shuffle(init_pos)
        return np.reshape(init_pos, self._grid_shape)

    def render(self, mode='human'):
        img = copy.copy(self._base_img)

        mask = (
            slice(self._agent_view[0], self._agent_view[0] + self._grid_shape[0]),
            slice(self._agent_view[1], self._agent_view[1] + self._grid_shape[1]),
        )

        # Iterate over all grid positions
        for pos, agent_strength, gold_level, stone_level in self._view_generator(mask):
            if gold_level and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                gold_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            elif stone_level and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                stone_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            else:
                cell_size = (CELL_SIZE, CELL_SIZE)
                gold_pos = stone_pos = agent_pos = (pos[0], pos[1])

            if gold_level != 0:
                fill_cell(img, pos=gold_pos, cell_size=cell_size, fill=GOLD_COLOR, margin=0.1)
                write_cell_text(img, text=str(gold_level), pos=gold_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if stone_level != 0:
                fill_cell(img, pos=stone_pos, cell_size=cell_size, fill=STONE_COLOR, margin=0.1)
                write_cell_text(img, text=str(stone_level), pos=stone_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if agent_strength != 0:
                draw_circle(img, pos=agent_pos, cell_size=cell_size, fill=AGENT_COLOR, radius=0.30)
                write_cell_text(img, text=str(agent_strength), pos=agent_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            # from gym.envs.classic_control import rendering
            # if self._viewer is None:
            #     self._viewer = rendering.SimpleImageViewer()
            # self._viewer.imshow(img)
            # return self._viewer.isopen

            from .rendering import Viewer
            if self._viewer is None:
                self._viewer = Viewer((self._grid_shape[0], self._grid_shape[1]))
            return self._viewer.render(self, return_rgb_array=mode=="rgb_array")

    def _view_generator(self, mask: Tuple[slice, slice]) -> Tuple[Coordinates, int, int]:
        """Yields position, number of agent and tree strength for all cells defined by `mask`.

        Args:
            mask: tuple of slices in extended coordinates.
        """
        agent_iter = np.ndenumerate(np.sum(self._agent_map[mask], axis=2))
        gold_iter = np.nditer(self._gold_map[mask])
        stone_iter = np.nditer(self._stone_map[mask])
        for (pos, n_a), n_g, n_s in zip(agent_iter, gold_iter, stone_iter):
            yield pos, n_a, n_g, n_s


    def _entity_not_in_obsrange(self, pos_i, pos_j):
        if abs(pos_i[0] - pos_j[0]) <= self._agent_view[1] and abs(pos_i[1] - pos_j[1]) <= self._agent_view[0]:
            return False
        else:
            return True


    def get_agent_obs(self) -> List[List[float]]:
        """Returns list of observations for each agent."""
        obs = np.zeros((self.n_agents, self._obs_len))
        for i, (agent_id, agent) in enumerate(self._agent_generator()):
            rel_pos = self._to_relative_coordinates(agent.pos)
            obs[i, 0] = rel_pos[0] / (self._grid_shape[0] - 1)  # Coordinate
            obs[i, 1] = rel_pos[1] / (self._grid_shape[1] - 1)  # Coordinate
            obs[i, 2] = self._step_count / self._max_steps  # Steps

            for j, (_, agent_strength, gold_level, stone_level) in zip(
                    itertools.count(start=3, step=3),
                    self._agent_view_generator(agent.pos, self._agent_view)):
                obs[i, j] = agent_strength / self.n_agents
                obs[i, j + 1] = gold_level
                obs[i, j + 2] = stone_level

        # Convert it from numpy array
        obs = obs.tolist()

        if self.full_observable:
            obs = [feature for agent_obs in obs for feature in agent_obs]
            obs = [obs] * self.n_agents

        return obs  # [self.n_agents, self._obs_len]

    def _agent_generator(self) -> Tuple[int, Agent]:
        """Yields agent_id and agent for all agents in environment."""
        for agent_id, agent in enumerate(self._agents):
            yield agent_id, agent

    def _agent_view_generator(self, pos: Coordinates, view_range: Tuple[int, int]):
        """Yields position, number of agent and gold level and stone level for cells in distance of `view_range` from `pos`.  """
        mask = (
            slice(pos[0] - view_range[0], pos[0] + view_range[0] + 1),
            slice(pos[1] - view_range[1], pos[1] + view_range[1] + 1),
        )
        yield from self._view_generator(mask)

    def step(self, agents_action: List[int]):

        self._step_count += 1
        rewards = np.full(self.n_agents, self._step_cost)

        # Move agents
        for (agent_id, agent), action in zip(self._agent_generator(), agents_action):
            if not self._agent_dones[agent_id]:
                self._update_agent_pos(agent, action)

        # capture gold and calculate reward
        mask_gold = (np.sum(self._agent_map, axis=2) > 0) & self._gold_map
        for gold in self._golds:
            if mask_gold[gold.pos] > 0:
                a_l = list(np.where(self._agent_map[gold.pos] == 1)[0])
                for index in a_l:
                    if 0 >= gold.cap_num[index] > - self._gold_punish_steps:
                        rewards[index] += self._gold_base_punish
                        gold.cap_num[index] -= 1
                    else:
                        if not (self._agents[index].score[gold.id - self.n_agents]):
                            rewards[index] += self._gold_reward
                            self._agents[index].score[gold.id - self.n_agents] = True
                            gold.cap_num[index] = 1
            if min(gold.cap_num) >= 1:
                self._gold_map[gold.pos] = 0
        # capture stone and calculate reward
        mask_stone = (np.sum(self._agent_map, axis=2) > 0) & self._stone_map
        for stone in self._stones:
            if mask_stone[stone.pos] > 0:
                a_l = list(np.where(self._agent_map[stone.pos] == 1)[0])
                for index in a_l:
                    if stone.cap_num[index] < self._stone_disappear_steps:
                        rewards[index] += self._stone_reward
                        stone.cap_num[index] += 1
            if min(stone.cap_num) >= self._stone_disappear_steps:
                self._stone_map[stone.pos] = 0

        self._total_episode_reward += rewards
        if (self._gold_map.max() == 0) or (self._step_count >= self._max_steps):
            self._agent_dones = [True] * self.n_agents

            gold_cap = [gold.cap_num for gold in self._golds]
            stone_cap = [stone.cap_num for stone in self._stones]
            for ag in range(self.n_agents):
                for i in range(self._n_golds):
                    if gold_cap[i][ag] >= 1:
                        self.ag_gold[ag] += 1
                for i in range(self._n_stones):
                    self.ag_stone[ag] += stone_cap[i][ag]
            self.ag_stone = [ele/self._stone_disappear_steps for ele in self.ag_stone]
            print("Gold mining:{}, Stone collecting:{}".format(self.ag_gold, self.ag_stone))

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def _update_agent_pos(self, agent: Agent, move: int):
        """Moves `agent` according the `move` command."""
        next_pos = self._next_pos(agent.pos, move)

        # Remove agent from old position
        self._agent_map[agent.pos[0], agent.pos[1], agent.id] = 0

        # Add agent to the new position
        agent.pos = next_pos
        self._agent_map[next_pos[0], next_pos[1], agent.id] = 1

    def _next_pos(self, curr_pos: Coordinates, move: int) -> Coordinates:
        """Returns next valid position in extended coordinates given by `move` command relative to `curr_pos`."""
        if move == ACTIONS_IDS['noop']:
            next_pos = curr_pos
        elif move == ACTIONS_IDS['down']:
            next_pos = (curr_pos[0] + 1, curr_pos[1])
        elif move == ACTIONS_IDS['left']:
            next_pos = (curr_pos[0], curr_pos[1] - 1)
        elif move == ACTIONS_IDS['up']:
            next_pos = (curr_pos[0] - 1, curr_pos[1])
        elif move == ACTIONS_IDS['right']:
            next_pos = (curr_pos[0], curr_pos[1] + 1)
        else:
            raise ValueError('Unknown action {}. Valid action are {}'.format(move, list(ACTIONS_IDS.values())))
        # np.clip is significantly slower, see: https://github.com/numpy/numpy/issues/14281
        # return tuple(np.clip(next_pos,
        #                      (self._agent_view[0], self._agent_view[1]),
        #                      (self._agent_view[0] + self._grid_shape[0] - 1,
        #                       self._agent_view[1] + self._grid_shape[1] - 1),
        #                      ))
        return (
            min(max(next_pos[0], self._agent_view[0]), self._grid_shape[0] + self._agent_view[0] - 1),
            min(max(next_pos[1], self._agent_view[1]), self._grid_shape[1] + self._agent_view[1] - 1),
        )

    def seed(self, n: Union[None, int] = None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
GOLD_COLOR = (242, 186, 2)
STONE_COLOR = (152, 82, 20)
WALL_COLOR = 'black'

CELL_SIZE = 35

ACTIONS_IDS = {
    'noop': 0,
    'down': 1,
    'left': 2,
    'up': 3,
    'right': 4,
}

PRE_IDS = {
    'empty': 0,
    'wall': 1,
    'agent': 2,
    'gold': 3,
    'stone': 4,
}
