# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import chex
import matplotlib.animation
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image
from matplotlib.axes import Axes
from numpy.typing import NDArray

#import jumanji.environments
from .maze_generation import EMPTY, WALL
from Final_project.pac_man.env_basic.viewer_basic import Viewer



class MazeViewer(Viewer):
    FONT_STYLE = "monospace"
    FIGURE_SIZE = (10.0, 10.0)
    # EMPTY is white, WALL is black
    COLORS: ClassVar[Dict[int, List[int]]] = {EMPTY: [1, 1, 1], WALL: [0, 0, 0]}

    def __init__(self, name: str, render_mode: str = "human") -> None:
        """Viewer for a maze environment.

        Args:
            name: the window name to be used when initialising the window.
            render_mode: the mode used to render the environment. Must be one of:
                - "human": render the environment on screen.
                - "rgb_array": return a numpy array frame representing the environment.
        """
        self._name = name
        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

        self._display: Callable[[plt.Figure], Optional[NDArray]]
        if render_mode == "rgb_array":
            self._display = self._display_rgb_array
        elif render_mode == "human":
            self._display = self._display_human
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def render(self, maze: chex.Array) -> Optional[NDArray]:
        """
        Render maze.

        Args:
            maze: the maze to render.

        Returns:
            RGB array if the render_mode is RenderMode.RGB_ARRAY.
        """
        self._clear_display()
        fig, ax = self._get_fig_ax()
        ax.clear()
        self._add_grid_image(maze, ax)
        return self._display(fig)

    def animate(
        self,
        mazes: Sequence[chex.Array],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Create an animation from a sequence of mazes.

        Args:
            mazes: sequence of `Maze` corresponding to consecutive timesteps.
            interval: delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = plt.subplots(num=f"{self._name}Animation", figsize=self.FIGURE_SIZE)
        plt.close(fig)

        def make_frame(maze_index: int) -> None:
            ax.clear()
            maze = mazes[maze_index]
            self._add_grid_image(maze, ax)

        # Create the animation object.
        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(mazes),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def close(self) -> None:
        plt.close(self._name)

    def _get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        recreate = not plt.fignum_exists(self._name)
        fig = plt.figure(self._name, figsize=self.FIGURE_SIZE)
        if recreate:
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        else:
            ax = fig.get_axes()[0]
        return fig, ax

    def _add_grid_image(self, maze: chex.Array, ax: Axes) -> image.AxesImage:
        img = self._create_grid_image(maze)
        ax.set_axis_off()
        return ax.imshow(img)

    def _create_grid_image(self, maze: chex.Array) -> NDArray:
        img = np.zeros((*maze.shape, 3))
        for tile_value, color in self.COLORS.items():
            img[np.where(maze == tile_value)] = color
        # Draw black frame around maze by padding axis 0 and 1
        img = np.pad(img, ((1, 1), (1, 1), (0, 0)))  # type: ignore
        return img

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            # Required to update render when using Jupyter Notebook.
            fig.canvas.draw()
            #if jumanji.environments.is_colab():
                #plt.show(self._name)
        else:
            # Required to update render when not using Jupyter Notebook.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _display_rgb_array(self, fig: plt.Figure) -> NDArray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())

    def _clear_display(self) -> None:
        #if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)



