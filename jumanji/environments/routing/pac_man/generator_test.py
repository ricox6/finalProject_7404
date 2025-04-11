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


import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.routing.pac_man.constants import DEFAULT_MAZE
from jumanji.environments.routing.pac_man.generator import AsciiGenerator


class TestAsciiGenerator:
    @pytest.fixture
    def key(self) -> chex.PRNGKey:
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def instance_generator(self) -> AsciiGenerator:
        return AsciiGenerator(DEFAULT_MAZE)

    def test_ascii_instance_generator_values(
        self,
        key: chex.PRNGKey,
        instance_generator: AsciiGenerator,
    ) -> None:
        state = instance_generator(key)

        assert state.step_count == 0
        assert state.grid.shape[0] == 31
        assert state.grid.shape[1] == 28
        assert state.pellets == 318
        assert state.frightened_state_time == 0
        assert jnp.array_equal(state.old_ghost_locations, state.ghost_locations)
        assert state.dead is False

        '''
        初始目标点 (init_targets)
        作用：在游戏开始时，鬼怪会移动到这些位置。这些目标点帮助鬼怪迅速占据有利位置，以便更好地追捕玩家。
        位置标记：在ASCII迷宫中用字符 'T' 标记。
        用途：初始化鬼怪的目标位置，确保它们在游戏开始时有明确的移动方向。
        散射目标点 (scatter_targets)
        作用：当鬼怪处于受惊状态（即玩家吃了强化物后，鬼怪变蓝且可被吃掉）时，鬼怪会移动到这些散射目标点。这些目标点通常位于迷宫的不同角落，使鬼怪分散开来，增加玩家的生存机会。
        位置标记：在ASCII迷宫中用字符 'S' 标记。
        用途：在受惊状态下，引导鬼怪远离玩家，增加游戏的策略性和挑战性。
        '''