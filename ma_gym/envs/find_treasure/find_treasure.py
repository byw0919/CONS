import copy
import itertools
import logging
from typing import List, Tuple, Union

import gym
import random
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)

Coordinates = Tuple[int, int]

BOX_TYPE = {
    'EASY': 1,  # Yellow box
    'HARD': 2,  # Red box
}

class Agent:

    def __init__(self, id: int, pos: Coordinates, action=None):
        self.id = id
        self.pos = pos
        self.act = action

class Box:

    def __init__(self, id: int, pos: Coordinates, type: int, open: bool = False, something: int = 0):
        self.id = id
        self.pos = pos
        self.type = type
        self.open = open
        self.something = something  # 0-Nothing has ever existed; 1-There is something; 2-There used to be something


class FindTreasure(gym.Env):
    """Find the Treasure"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape: Coordinates = (8, 8), n_agents: int = 4, n_treasures: int = 1,
                 n_box1: int = 3, n_box2: int = 6, agent_view: Tuple[int, int] = (2, 2), full_observable: bool = False,
                 step_cost: float = -0.01, open_penalty1: float = -1, open_penalty2: float = -2, coin_reward: float = 2,
                 treasure_reward: float = 15, max_steps: int = 50):

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._n_box1 = n_box1
        self._n_box2 = n_box2
        self._n_coins = self._n_box1
        self._n_treasures = n_treasures

        self._agent_view = agent_view
        self.full_observable = full_observable
        self._step_cost = step_cost
        self._coin_reward = coin_reward
        self._treasure_reward = treasure_reward

        self._op_pnt1 = open_penalty1
        self._op_pnt2 = open_penalty2

        self._max_steps = max_steps
        self.steps_beyond_done = 0
        self.seed()

        self._agents = []  # List[Agent]
        self._boxs = []  # List[Box]

        # Relative coordinates refer to the coordinates in non pad grid. These are the only
        # coordinates visible to user. Extended coordinates refer to the coordinates in pad grid.
        self._agent_map = None
        self._box_map = None

        self._total_episode_reward = None
        self._agent_dones = None

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

        return self.get_agent_obs()

    @property
    def _box_1(self):
        return [box for box in self._boxs if box.type == 1]

    @property
    def _box_2(self):
        return [box for box in self._boxs if box.type == 2]

    @property
    def _close_box(self):
        return [box for box in self._boxs if not box.open]

    def _get_box_with_id(self, id):
        return [box for box in self._boxs if box.id == id][0]

    def _get_pick_agent(self, acts):
        return [i for i in range(len(acts)) if acts[i] == ACTIONS_IDS['pick']]

    def _get_open_agent(self, acts):
        return [i for i in range(len(acts)) if acts[i] == ACTIONS_IDS['open']]

    def _get_agent_with_pos(self, pos):
        a = [ag for ag in self._agents if ag.pos == pos]
        if len(a) == 0:
            return None
        else:
            return a

    def _get_entity_with_pos(self, pos):
        l = [box for box in self._boxs if box.pos == pos]
        if len(l) == 0:
            return 0, None
        else:
            box = l[0]
            if box.open:
                if box.something == 2:
                    return 0, box
                elif box.type == 1:
                    return 1, box  # coin
                elif box.type == 2 and box.something == 1:  # treasure
                    return 2, box  # treasure
                else:
                    return 0, box
            else:
                if box.type == 1:
                    return 3, box  # yellow box
                else:
                    return 4, box  # red box

    def _init_episode(self):
        init_positions = self._generate_init_pos()

        agent_id, box1_id, box2_id = 0, 0, self._n_box1
        self._agents, self._boxs = [], []
        self._agent_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
            self.n_agents
        ), dtype=np.int32)

        self._box_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)

        self._sth_map = np.zeros((
            self._grid_shape[0] + 2 * (self._agent_view[0]),
            self._grid_shape[1] + 2 * (self._agent_view[1]),
        ), dtype=np.int32)


        for pos, cell in np.ndenumerate(init_positions):
            pos = self._to_extended_coordinates(pos)
            if cell == PRE_IDS['agent']:
                self._agent_map[pos[0], pos[1], agent_id] = 1
                self._agents.append(Agent(agent_id, pos=pos))
                agent_id += 1
            elif cell == PRE_IDS['box_1']:
                self._box_map[pos] = 1
                self._boxs.append(Box(box1_id, pos=pos, type=1, something=1))
                box1_id += 1
            elif cell == PRE_IDS['box_2']:
                self._box_map[pos] = 1
                self._boxs.append(Box(box2_id, pos=pos, type=2))
                box2_id += 1

        treasure_box_id = random.sample(range(self._n_box1, self._n_box1 + self._n_box2), self._n_treasures)
        for id in treasure_box_id:
            box = self._get_box_with_id(id)
            box.something = 1

    def _to_extended_coordinates(self, relative_coordinates):
        """Translate relative coordinates in to the extended coordinates."""
        return relative_coordinates[0] + self._agent_view[0], relative_coordinates[1] + self._agent_view[1]

    def _to_relative_coordinates(self, extended_coordinates):
        """Translate extended coordinates in to the relative coordinates."""
        return extended_coordinates[0] - self._agent_view[0], extended_coordinates[1] - self._agent_view[1]

    def _generate_init_pos(self) -> np.ndarray:
        """
        Returns randomly selected initial positions for agents, points and boxs in relative coordinates.
        No agent boxs share the same cell in initial positions.
        """
        init_pos = np.array(
            [PRE_IDS['agent']] * self.n_agents +
            [PRE_IDS['box_1']] * self._n_box1 +
            [PRE_IDS['box_2']] * self._n_box2 +
            [PRE_IDS['empty']] * (np.prod(self._grid_shape) - self.n_agents - self._n_box1 - self._n_box2)
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
        for pos, agent_strength, sth_level, box_level in self._view_generator(mask):
            if sth_level and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                sth_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            elif box_level and agent_strength:
                cell_size = (CELL_SIZE, CELL_SIZE / 2)
                box_pos = (pos[0], 2 * pos[1])
                agent_pos = (pos[0], 2 * pos[1] + 1)
            else:
                cell_size = (CELL_SIZE, CELL_SIZE)
                sth_pos = box_pos = agent_pos = (pos[0], pos[1])

            if sth_level != 0:  # item
                extended_pos = self._to_extended_coordinates(pos)
                entity, box = self._get_entity_with_pos(extended_pos)

                if entity == 1:  # coin
                    sth_color = 'green'
                else:  # treasure
                    assert entity == 2
                    sth_color = 'orange'
                draw_circle(img, pos=sth_pos, cell_size=cell_size, fill=sth_color, radius=0.10)
                write_cell_text(img, text=str(box.id), pos=sth_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if box_level != 0:  # box
                extended_pos = self._to_extended_coordinates(pos)
                entity, box = self._get_entity_with_pos(extended_pos)

                if entity == 3:  # yellow box
                    box_color = (242, 186, 2)
                elif entity == 4:  # red box
                    assert entity == 4
                    box_color = 'red'
                fill_cell(img, pos=box_pos, cell_size=cell_size, fill=box_color, margin=0.1)
                write_cell_text(img, text=str(box.id), pos=box_pos,
                                cell_size=cell_size, fill='white', margin=0.4)

            if agent_strength != 0:
                extended_pos = self._to_extended_coordinates(pos)
                ags = self._get_agent_with_pos(extended_pos)
                pick_ag = sum([1 if ag.act == ACTIONS_IDS['pick'] else 0 for ag in ags])
                open_ag = sum([1 if ag.act == ACTIONS_IDS['open'] else 0 for ag in ags])
                if pick_ag == 0 and open_ag == 0:
                    agent_color = normal_agent_color
                elif pick_ag > 0 and open_ag == 0:
                    agent_color = pick_agent_color
                elif pick_ag == 0 and open_ag > 0:
                    agent_color = open_agent_color
                else:
                    agent_color = both_agent_color

                draw_circle(img, pos=agent_pos, cell_size=cell_size, fill=agent_color, radius=0.30)
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
        sth_iter = np.nditer(self._sth_map[mask])
        box_iter = np.nditer(self._box_map[mask])
        for (pos, n_a), n_s, n_b in zip(agent_iter, sth_iter, box_iter):
            yield pos, n_a, n_s, n_b

    def get_agent_obs(self) -> List[List[float]]:
        """Returns list of observations for each agent."""
        obs = np.zeros((self.n_agents, self._obs_len))
        for i, (agent_id, agent) in enumerate(self._agent_generator()):
            rel_pos = self._to_relative_coordinates(agent.pos)
            # obs_1: agent obs: Pos (2) + Step (1) + Neighborhood (3 * mask_size)
            obs[i, 0] = rel_pos[0] / (self._grid_shape[0] - 1)  # Coordinate
            obs[i, 1] = rel_pos[1] / (self._grid_shape[1] - 1)  # Coordinate
            obs[i, 2] = self._step_count / self._max_steps  # Steps

            mask_pos = [pos for pos in itertools.product(range(agent.pos[0] - self._agent_view[0],
                                                               agent.pos[0] + self._agent_view[0] + 1),
                                                         range(agent.pos[1] - self._agent_view[1],
                                                               agent.pos[1] + self._agent_view[1] + 1))]
            for pos, j, (_, agent_strength, sth_level, box_level) in zip(mask_pos,
                    itertools.count(start=3, step=3),
                    self._agent_view_generator(agent.pos, self._agent_view)):
                obs[i, j] = agent_strength / self.n_agents
                entity, box = self._get_entity_with_pos(pos)  # type, entity object
                if entity == 0:  # none
                    obs[i, j + 1] = 0
                    obs[i, j + 2] = 0
                elif entity == 1:  # coin
                    obs[i, j + 1] = 0
                    obs[i, j + 2] = 0.5
                elif entity == 2:  # treasure
                    obs[i, j + 1] = 0
                    obs[i, j + 2] = 1
                elif entity == 3:  # yellow box
                    obs[i, j + 1] = 0.5
                    obs[i, j + 2] = 0
                elif entity == 4:  # red box
                    obs[i, j + 1] = 1
                    obs[i, j + 2] = 0

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
        """Yields position, number of agent and poit level and box level for cells in distance of `view_range` from `pos`.  """
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
            agent.act = action

        open_ag = self._get_open_agent(agents_action)
        pick_ag = self._get_pick_agent(agents_action)

        mask_sth = (np.sum(self._agent_map, axis=2) > 0) & (self._sth_map > 0)  # Which items have agents at them
        if np.any(mask_sth):
            pending = self._agent_map[mask_sth]
            poses = np.array(np.where(mask_sth)).transpose()  # Their positions
            for x, pos in zip(pending, poses):
                ag_at_this_grid = np.where(x==1)[0]
                l_ag = list(set(pick_ag).intersection(set(ag_at_this_grid)))  # Intersection of "agents executing the 'pick' action" and "agents on this grid"
                if len(l_ag) > 0:  # something at this grid is picked up
                    entity, box = self._get_entity_with_pos((pos[0], pos[1]))
                    assert box.open
                    assert box.something == 1
                    rewards += self._sth_map[pos[0], pos[1]] / self.n_agents
                    self._sth_map[pos[0], pos[1]] = 0
                    box.something = 2  # There used to be something in the box, but has been collected

        mask_box = (np.sum(self._agent_map, axis=2) > 0) & (self._box_map > 0)  # Which boxes have agents at them
        if np.any(mask_box):
            pending = self._agent_map[mask_box]
            poses = np.array(np.where(mask_box)).transpose()  # Their positions
            for x, pos in zip(pending, poses):
                ag_at_this_grid = np.where(x == 1)[0]
                if len(ag_at_this_grid) < 2:
                    continue
                l_ag = list(set(open_ag).intersection(set(ag_at_this_grid)))  # Intersection of "agents executing the 'open' action" and "agents on this grid"
                if len(l_ag) >= 2:  # Two or more agents execute the 'open' action, and the box is opened
                    self._box_map[pos[0], pos[1]] = 0
                    entity, box = self._get_entity_with_pos((pos[0], pos[1]))
                    if entity == 0 or entity == 1 or entity == 2:  # nothing in it
                        raise ValueError('Error1')
                    elif entity == 3:
                        box.open = True
                        rewards += self._op_pnt1 / self.n_agents
                        self._sth_map[pos[0], pos[1]] = self._coin_reward
                    elif entity == 4:
                        box.open = True
                        rewards += self._op_pnt2 / self.n_agents
                        if box.something == 1:
                            self._sth_map[pos[0], pos[1]] = self._treasure_reward

        if self._step_count >= self._max_steps:
            self._agent_dones = [True] * self.n_agents

            num_get_coin = sum([1 if box.something == 2 else 0 for box in self._box_1])
            num_get_treasure = sum([1 if box.something == 2 else 0 for box in self._box_2])
            num_open_box1 = sum([1 if box.open else 0 for box in self._box_1])
            num_open_box2 = sum([1 if box.open else 0 for box in self._box_2])
            print("open {} box-yellow and {} box-red, get {} coins and {} treasures.".format(num_open_box1,
                                                                                 num_open_box2,
                                                                                 num_get_coin,
                                                                                 num_get_treasure))

        return self.get_agent_obs(), rewards, self._agent_dones, {'episode_steps':self._step_count}

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
        if move == ACTIONS_IDS['noop'] or move == ACTIONS_IDS['pick'] or move == ACTIONS_IDS['open'] :
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

pick_agent_color = (128, 0, 128)  # purple
open_agent_color = (0, 128, 128)  # cyan
both_agent_color = 'pink'
normal_agent_color = 'black'


CELL_SIZE = 35
#
ACTIONS_IDS = {
    'noop': 0,
    'down': 1,
    'left': 2,
    'up': 3,
    'right': 4,
    'open': 5,
    'pick': 6,
}

PRE_IDS = {
    'empty': 0,
    'wall': 1,
    'agent': 2,
    'box_1': 3,  # Yellow box
    'box_2': 4,  # Red box
}
