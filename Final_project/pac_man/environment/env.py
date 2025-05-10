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

from functools import cached_property
from typing import Any, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation
from chex import PRNGKey

import Final_project.pac_man.env_basic.specs_basic as specs
from Final_project.pac_man.env_basic.environment_basic import Environment
from .constants import DEFAULT_MAZE, MOVES
from .generator import AsciiGenerator, Generator
from .types import Observation, Position, State
from .utils import (
    check_ghost_collisions,
    ghost_move,
    player_step,
)
from .viewer import PacManViewer
from Final_project.pac_man.env_basic.types_basic import TimeStep, restart, termination, transition
from Final_project.pac_man.env_basic.viewer_basic import Viewer


class PacMan(Environment[State, specs.DiscreteArray, Observation]):
    """PacMan游戏环境的JAX实现"""

    def __init__(
            self,
            generator: Optional[Generator] = None,
            viewer: Optional[Viewer[State]] = None,
            time_limit: Optional[int] = None,
    ) -> None:
        self.generator = generator or AsciiGenerator(DEFAULT_MAZE)
        self.x_size = self.generator.x_size
        self.y_size = self.generator.y_size
        self.pellet_spaces = self.generator.pellet_spaces
        super().__init__()

        self._viewer = viewer or PacManViewer("Pacman", render_mode="human")
        self.time_limit = time_limit or 1000

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """定义观察规范"""
        player_locations = specs.Spec(
            Position,
            "PositionSpec",
            y=specs.BoundedArray((), jnp.int32, 0, self.x_size - 1, "y_coordinate"),
            x=specs.BoundedArray((), jnp.int32, 0, self.y_size - 1, "x_coordinate"),
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            grid=specs.BoundedArray(
                shape=(self.x_size, self.y_size),
                dtype=jnp.int32,
                name="grid",
                minimum=0,
                maximum=1,
            ),
            player_locations=player_locations,
            ghost_locations=specs.Array(
                shape=(4, 2),
                dtype=jnp.int32,
                name="ghost_locations",
            ),
            power_up_locations=specs.Array(
                shape=(4, 2),
                dtype=jnp.int32,
                name="power_up_locations",
            ),
            pellet_locations=specs.Array(
                shape=self.pellet_spaces.shape,
                dtype=jnp.int32,
                name="pellet_locations",
            ),
            action_mask=specs.BoundedArray(
                shape=(5,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            frightened_state_time=specs.Array((), jnp.int32, "frightened_state_time"),
            score=specs.Array((), jnp.int32, "score"),
            ghost_visible=specs.Array(shape=(4,), dtype=bool, name="ghost_visible"),
            ghost_masked_locations=specs.Array(
                shape=(4, 2),
                dtype=jnp.int32,
                name="ghost_masked_locations",
            )
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """返回动作规范"""
        return specs.DiscreteArray(5, name="action")

    def __repr__(self) -> str:
        return (
            f"PacMan(\n"
            f"\tnum_rows={self.x_size!r},\n"
            f"\tnum_cols={self.y_size!r},\n"
            f"\ttime_limit={self.time_limit!r}, \n"
            f"\tgenerator={self.generator!r}, \n"
            ")"
        )

    def reset(self, key: PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """重置环境"""
        state = self.generator(key)

        # 初始化幽灵可见性和掩码位置
        ghost_visible = jnp.ones(4, dtype=bool)
        ghost_masked_locations = state.ghost_locations
        state = state.replace(
            ghost_visible=ghost_visible,
            ghost_masked_locations=ghost_masked_locations,
            old_ghost_locations=state.ghost_locations,
        )

        obs = self._observation_from_state(state)
        timestep = restart(observation=obs)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """执行一个时间步的环境动态"""
        current_ghost_locations = state.ghost_locations
        state = state.replace(old_ghost_locations=current_ghost_locations)

        # 更新幽灵可见性
        def toggle_visibility(state: State) -> State:
            new_ghost_visible = ~state.ghost_visible
            ghost_masked_locations = jnp.where(
                new_ghost_visible[:, None],
                state.ghost_locations,
                jnp.full((4, 2), -1)
            )
            return state.replace(
                ghost_visible=new_ghost_visible,
                ghost_masked_locations=ghost_masked_locations,
            )

        state = jax.lax.cond(
            (state.step_count % 5) == 0,
            toggle_visibility,
            lambda s: s,
            state
        )

        # 更新状态
        updated_state, collision_rewards = self._update_state(state, action)
        next_state = updated_state.replace(step_count=state.step_count + 1)

        # 检查终止条件
        num_pellets = next_state.pellets
        dead = next_state.dead
        time_limit_exceeded = next_state.step_count >= self.time_limit
        all_pellets_found = num_pellets == 0
        done = time_limit_exceeded | dead | all_pellets_found

        reward = jnp.asarray(collision_rewards)
        observation = self._observation_from_state(next_state)

        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )

        return next_state, timestep

    def _update_state(self, state: State, action: chex.Array) -> Tuple[State, int]:
        """更新环境状态"""
        key = state.key
        key, _ = jax.random.split(key)

        # 移动玩家
        next_player_pos = player_step(
            state=state, action=action, x_size=self.x_size, y_size=self.y_size, steps=1
        )
        next_player_pos = self.check_wall_collisions(state, next_player_pos)
        state = state.replace(last_direction=jnp.array(action, jnp.int32))

        # 移动幽灵
        old_ghost_locations = state.ghost_locations
        ghost_paths, ghost_actions, key = ghost_move(state, self.x_size, self.y_size)

        # 检查与幽灵的碰撞
        state, done, ghost_col_rewards = check_ghost_collisions(ghost_paths, next_player_pos, state)
        state = state.replace(player_locations=next_player_pos, dead=done)

        # 检查能量豆
        power_up_locations, eat, power_up_rewards = self.check_power_up(state)

        # 检查豆子收集
        collision_rewards, pellets, num_pellets = self.check_rewards(state)

        # 更新状态
        state = state.replace(
            ghost_init_steps=state.ghost_init_steps - 1,
            old_ghost_locations=old_ghost_locations,
            pellet_locations=pellets,
            pellets=num_pellets,
            key=key,
            power_up_locations=power_up_locations,
            ghost_actions=ghost_actions,
            ghost_starts=state.ghost_starts - 1,
            score=jnp.array(state.score + collision_rewards + power_up_rewards + ghost_col_rewards, jnp.int32),
            frightened_state_time=jax.lax.cond(
                eat > 0,
                lambda: jnp.array(30, jnp.int32),
                lambda: jnp.array(state.frightened_state_time - 1, jnp.int32)
            )
        )

        return state, collision_rewards + power_up_rewards + ghost_col_rewards

    def check_rewards(self, state: State) -> Tuple[int, chex.Array, int]:
        """检查奖励和豆子状态"""
        pellet_spaces = jnp.array(state.pellet_locations)
        player_space = state.player_locations
        ps = jnp.array([player_space.y, player_space.x])

        power_up_space = jnp.array(state.power_up_locations)
        player_space_power = state.player_locations
        pp = jnp.array([player_space_power.y, player_space_power.x])

        ate_pellet = jnp.any(jnp.all(ps == pellet_spaces, axis=-1))
        ate_power = jnp.any(jnp.all(pp == power_up_space, axis=-1))
        num_pellets = state.pellets - ate_pellet.astype(int)
        rewards = ate_pellet * 10.0+ate_power*100.0
        mask = jnp.logical_not(jnp.all(ps == pellet_spaces, axis=-1))
        pellet_spaces = pellet_spaces * mask[..., None]

        return rewards, pellet_spaces, num_pellets

    def player_step(self, state: State, action: int, steps: int = 1) -> Position:
        """计算玩家新位置"""
        position = state.player_locations

        move_left = lambda position: (position.y, position.x - steps)
        move_up = lambda position: (position.y - steps, position.x)
        move_right = lambda position: (position.y, position.x + steps)
        move_down = lambda position: (position.y + steps, position.x)
        no_op = lambda position: (position.y, position.x)

        new_pos_row, new_pos_col = jax.lax.switch(
            action, [move_left, move_up, move_right, move_down, no_op], position
        )

        return Position(x=new_pos_col % self.x_size, y=new_pos_row % self.y_size)

    def check_power_up(self, state: State) -> Tuple[chex.Array, chex.Numeric, chex.Numeric]:
        """检查能量豆状态"""
        power_up_locations = jnp.array(state.power_up_locations)
        player_space = state.player_locations
        player_loc = jnp.array([player_space.y, player_space.x])

        on_powerup = (player_loc == power_up_locations).all(axis=-1).any()
        eat = on_powerup.astype(int)
        mask = (player_loc == power_up_locations).all(axis=-1)
        power_up_locations = power_up_locations * (~mask).reshape(4, 1)

        return power_up_locations, eat, eat * 50.0

    def check_wall_collisions(self, state: State, new_player_pos: Position) -> Any:
        """检查墙壁碰撞"""
        grid = state.grid
        location_value = grid[new_player_pos.x, new_player_pos.y]

        return jax.lax.cond(
            location_value == 1,
            lambda x: new_player_pos,
            lambda x: state.player_locations,
            0,
        )

    def _compute_action_mask(self, state: State) -> chex.Array:
        """计算动作掩码"""
        grid = state.grid
        player_pos = state.player_locations

        def is_move_valid(agent_position: Position, move: chex.Array) -> chex.Array:
            y, x = jnp.array([agent_position.y, agent_position.x]) + move
            return grid[x][y]

        action_mask = jax.vmap(is_move_valid, in_axes=(None, 0))(player_pos, MOVES) * jnp.array(
            [True, True, True, True, False]
        )

        return action_mask

    def _observation_from_state(self, state: State) -> Observation:
        """从状态创建观察"""
        action_mask = self._compute_action_mask(state).astype(bool)
        return Observation(
            grid=state.grid,
            player_locations=state.player_locations,
            ghost_locations=state.ghost_locations,
            power_up_locations=state.power_up_locations,
            frightened_state_time=state.frightened_state_time,
            pellet_locations=state.pellet_locations,
            action_mask=action_mask,
            score=state.score,
            ghost_visible=state.ghost_visible,
            ghost_masked_locations=state.ghost_masked_locations
        )

    def render(self, state: State) -> Any:
        """渲染当前状态"""
        return self._viewer.render(state)

    def animate(
            self,
            states: Sequence[State],
            interval: int = 200,
            save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """创建动画"""
        return self._viewer.animate(
            states=states,
            interval=interval,
            save_path=save_path,
        )

    def close(self) -> None:
        """关闭环境"""
        self._viewer.close()