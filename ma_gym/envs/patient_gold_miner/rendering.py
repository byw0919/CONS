"""
2D rendering
"""

import math
import os
import sys

import numpy as np
import math
import six
from gym import error
from ..utils.draw import draw_circle, draw_grid, fill_cell, write_cell_text
from gym.envs.classic_control import rendering

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 50
        self.icon_size = 20

        self.width = self.cols * self.grid_size + 1
        self.height = self.rows * self.grid_size + 1
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "icons")]
        pyglet.resource.reindex()

        self.img_gold = pyglet.resource.image("gold.png")
        self.img_stone = pyglet.resource.image("stone.png")
        self.img_agent = pyglet.resource.image("miner.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = rendering.Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_gold(env)
        self._draw_stone(env)
        self._draw_players(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,
                        self.grid_size * r,
                        self.grid_size * self.cols,
                        self.grid_size * r,
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        self.grid_size * c,
                        0,
                        self.grid_size * c,
                        self.grid_size * self.rows,
                    ),
                ),
                ("c3B", (*_BLACK, *_BLACK)),
            )
        batch.draw()

    def _draw_gold(self, env):
        idxes = list(zip(*env._gold_map.nonzero()))
        ridxes = [env._to_relative_coordinates(pos) for pos in idxes]
        golds = []
        batch = pyglet.graphics.Batch()

        for row, col in ridxes:
            golds.append(
                pyglet.sprite.Sprite(
                    self.img_gold,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for g in golds:
            g.update(scale=self.grid_size / g.width)
        batch.draw()

    def _draw_stone(self, env):
        idxes = list(zip(*env._stone_map.nonzero()))
        ridxes = [env._to_relative_coordinates(pos) for pos in idxes]
        stones = []
        batch = pyglet.graphics.Batch()

        for row, col in ridxes:
            stones.append(
                pyglet.sprite.Sprite(
                    self.img_stone,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for s in stones:
            s.update(scale=self.grid_size / s.width)
        batch.draw()

    def _draw_players(self, env):
        players = []
        batch = pyglet.graphics.Batch()

        for player in env._agents:
            (row, col) = env._to_relative_coordinates(player.pos)

            players.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    self.grid_size * col,
                    self.height - self.grid_size * (row + 1),
                    batch=batch,
                )
            )
        for p in players:
            p.update(scale=self.grid_size / p.width)
        batch.draw()
        for p in env._agents:
            self._draw_badge(*env._to_relative_coordinates(p.pos), p.id, pos_x=3/4, pos_y=1/5)

    def _draw_badge(self, row, col, level, pos_x, pos_y):
        resolution = 6
        radius = self.grid_size / 5

        badge_x = col * self.grid_size + pos_x * self.grid_size
        badge_y = self.height - self.grid_size * (row + 1) + pos_y * self.grid_size

        # make a circle
        verts = []
        for i in range(resolution):
            angle = 2 * math.pi * i / resolution
            x = radius * math.cos(angle) + badge_x
            y = radius * math.sin(angle) + badge_y
            verts += [x, y]
        circle = pyglet.graphics.vertex_list(resolution, ("v2f", verts))
        glColor3ub(*_BLACK)
        circle.draw(GL_POLYGON)
        glColor3ub(*_BLACK)
        circle.draw(GL_LINE_LOOP)
        label = pyglet.text.Label(
            str(level),
            bold=True,
            font_name="Times New Roman",
            font_size=12,
            x=badge_x,
            y=badge_y + 1,
            anchor_x="center",
            anchor_y="center",
        )
        label.draw()

