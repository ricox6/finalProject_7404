'''from typing import Sequence, Tuple, Optional, NamedTuple, Any

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.pac_man import Observation, PacMan
from Final_project.pac_man.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from Final_project.pac_man.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


class LSTMState(NamedTuple):
    """LSTM state for the recurrent network."""
    hidden: chex.Array
    cell: chex.Array

def make_actor_critic_networks_pacman(
        pac_man: PacMan,
        num_channels: Sequence[int],
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        lstm_hidden_size: int = 128,
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `PacMan` environment with LSTM layer."""
    num_actions = jnp.array(pac_man.action_spec.num_values)
    parametric_action_distribution = CategoricalParametricDistribution(num_actions=num_actions)
    policy_network = make_network_pac_man(
        pac_man=pac_man,
        critic=False,
        conv_n_channels=num_channels,
        mlp_units=policy_layers,
        num_actions=num_actions,
        lstm_hidden_size=lstm_hidden_size,
    )
    value_network = make_network_pac_man(
        pac_man=pac_man,
        critic=True,
        conv_n_channels=num_channels,
        mlp_units=value_layers,
        num_actions=num_actions,
        lstm_hidden_size=lstm_hidden_size,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def process_image(observation) -> chex.Array:
    """JAX安全的process_image版本"""

    # 确保所有输入字段都是 jnp.ndarray，且类型合理
    grid = jnp.array(observation.grid, dtype=jnp.float32)
    player_x = jnp.int32(observation.player_locations.x)
    player_y = jnp.int32(observation.player_locations.y)
    ghost_pos = jnp.array(observation.ghost_locations, dtype=jnp.int32)  # shape (4, 2)
    pellets_loc = jnp.array(observation.power_up_locations, dtype=jnp.int32)  # shape (M, 2)
    frightened_time = jnp.array(observation.frightened_state_time, dtype=jnp.float32)
    pellet_locations = jnp.array(observation.pellet_locations, dtype=jnp.int32)  # shape (K, 2)
    action_mask = jnp.array(observation.action_mask, dtype=bool)

    layer_1 = grid * 0.66
    layer_2 = grid * 0.0
    layer_3 = grid * 0.33

    # 对 pellet_locations 做循环处理
    def pellet_body(i, layers):
        l1, l2, l3 = layers
        loc = pellet_locations[i]
        cond = jnp.any(loc > 0)  # 判断坐标是否有效

        def true_fn(layers):
            l1, l2, l3 = layers
            x, y = loc[0], loc[1]
            l3 = l3.at[y, x].set(1.0)
            l2 = l2.at[y, x].set(0.8)
            l1 = l1.at[y, x].set(0.6)
            return l1, l2, l3

        def false_fn(layers):
            return layers

        return jax.lax.cond(cond, true_fn, false_fn, layers)

    layers = (layer_1, layer_2, layer_3)
    layers = jax.lax.fori_loop(0, pellet_locations.shape[0], pellet_body, layers)
    layer_1, layer_2, layer_3 = layers

    # Power pellets
    def power_pellet_body(i, layers):
        l1, l2, l3 = layers
        p = pellets_loc[i]
        x, y = p[0], p[1]
        l1 = l1.at[y, x].set(0.5)
        l2 = l2.at[y, x].set(0.0)
        l3 = l3.at[y, x].set(0.5)
        return l1, l2, l3

    layers = (layer_1, layer_2, layer_3)
    layers = jax.lax.fori_loop(0, pellets_loc.shape[0], power_pellet_body, layers)
    layer_1, layer_2, layer_3 = layers

    # 设置player位置（黄色）
    layer_1 = layer_1.at[player_y, player_x].set(1.0)
    layer_2 = layer_2.at[player_y, player_x].set(1.0)
    layer_3 = layer_3.at[player_y, player_x].set(0.0)

    cr = jnp.array([1, 1, 0, 1], dtype=jnp.float32)
    cg = jnp.array([0, 0.7, 1, 0.7], dtype=jnp.float32)
    cb = jnp.array([0, 1, 1, 0.35], dtype=jnp.float32)

    scatter = frightened_time[0] / 60.0  # 标量

    def ghost_body(i, layers):
        l1, l2, l3 = layers
        x, y = ghost_pos[i, 0], ghost_pos[i, 1]
        l1 = l1.at[y, x].set(cr[0])
        l2 = l2.at[y, x].set(cg[0] + scatter)
        l3 = l3.at[y, x].set(cb[0] + scatter)
        return l1, l2, l3

    layers = (layer_1, layer_2, layer_3)
    layers = jax.lax.fori_loop(0, ghost_pos.shape[0], ghost_body, layers)
    layer_1, layer_2, layer_3 = layers

    # 清除(0,0)点
    layer_1 = layer_1.at[0, 0].set(0.0)
    layer_2 = layer_2.at[0, 0].set(0.0)
    layer_3 = layer_3.at[0, 0].set(0.0)

    rgb = jnp.stack([layer_1, layer_2, layer_3], axis=-1)
    return rgb

def make_network_pac_man(
        pac_man: PacMan,
        critic: bool,
        conv_n_channels: Sequence[int],
        mlp_units: Sequence[int],
        num_actions: int,
        lstm_hidden_size: int = 128,
) -> FeedForwardNetwork:
    """Create a network for the PacMan environment with an LSTM layer.

    Args:
        pac_man: PacMan environment instance
        critic: whether this is the critic network
        conv_n_channels: sequence of output channels for convolutional layers
        mlp_units: sequence of units for MLP layers
        num_actions: number of possible actions
        lstm_hidden_size: size of the LSTM hidden state

    Returns:
        A recurrent feed-forward network
    """

    def network_fn(
            observation: Observation,
            state: Optional[LSTMState] = None,
            is_training: bool = False
    ) -> Tuple[chex.Array, LSTMState]:
        # Create CNN layers
        conv_layers = [
            [
                hk.Conv2D(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in conv_n_channels
        ]

        # Define CNN torso
        torso = hk.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                hk.Flatten(),
            ]
        )

        # Process image observation
        rgb_observation = process_image(observation)  # (B, G, G, 3)
        obs = rgb_observation.astype(float)

        # Extract features
        player_pos = jnp.array([observation.player_locations.x, observation.player_locations.y])
        player_pos = jnp.stack(player_pos, axis=-1)
        scatter_time = observation.frightened_state_time / 60
        scatter_time = jnp.expand_dims(scatter_time, axis=-1)
        ghost_locations_x = observation.ghost_locations[:, :, 0]
        ghost_locations_y = observation.ghost_locations[:, :, 1]

        # Get embedding from CNN
        embedding = torso(obs)  # (B, H)

        # Concatenate with other features
        combined_features = jnp.concatenate(
            [embedding, player_pos, ghost_locations_x, ghost_locations_y, scatter_time],
            axis=-1,
        )  # (B, H+...)

        # Initialize LSTM state if not provided
        batch_size = 1  # Assuming batch size is 1 during inference
        if state is None:
            state = LSTMState(
                hidden=jnp.zeros((batch_size, lstm_hidden_size)),
                cell=jnp.zeros((batch_size, lstm_hidden_size))
            )

        # Apply LSTM
        lstm = hk.LSTM(lstm_hidden_size)
        lstm_output, new_state = lstm(combined_features, state)

        # Apply head based on whether it's critic or actor
        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            output = jnp.squeeze(head(lstm_output), axis=-1)
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            mask = observation.action_mask
            logits = head(lstm_output)
            output = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)

        return output, new_state

    # Use hk.transform to handle parameters
    transformed = hk.without_apply_rng(hk.transform_with_state(network_fn))

    # Create wrapper functions to maintain API compatibility
    def init_fn(
            rng: jax.random.PRNGKey,
            observation: Any,
    ) -> Tuple[hk.Params, hk.State]:
        """初始化网络参数和状态"""
        params, state = transformed.init(rng, observation)
        return params, state

    def apply_fn(
            params_and_state: Tuple[hk.Params, hk.State],
            observation: Any,
            lstm_state: Optional[Any] = None,
    ) -> Tuple[Any, hk.State]:
        """前向传播，返回输出和新网络状态"""
        params, network_state = params_and_state
        output, new_network_state = transformed.apply(params, network_state, observation, lstm_state)
        return output, new_network_state

    return FeedForwardNetwork(init=init_fn, apply=apply_fn)'''
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

from typing import Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.pac_man import Observation, PacMan
from Final_project.pac_man.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from Final_project.pac_man.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)


def make_actor_critic_networks_pacman(
    pac_man: PacMan,
    num_channels: Sequence[int],
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
) -> ActorCriticNetworks:
    """Make actor-critic networks for the `PacMan` environment."""
    num_actions = np.asarray(pac_man.action_spec.num_values)
    parametric_action_distribution = CategoricalParametricDistribution(num_actions=num_actions)
    policy_network = make_network_pac_man(
        pac_man=pac_man,
        critic=False,
        conv_n_channels=num_channels,
        mlp_units=policy_layers,
        num_actions=num_actions,
    )
    value_network = make_network_pac_man(
        pac_man=pac_man,
        critic=True,
        conv_n_channels=num_channels,
        mlp_units=value_layers,
        num_actions=num_actions,
    )
    return ActorCriticNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def process_image(observation: Observation) -> chex.Array:
    """Process the `Observation` to be usable by the critic model.

    Args:
        observation: the observation as returned by the environment.

    Returns:
        rgb: a 2D, RGB image of the current observation.
    """

    layer_1 = jnp.array(observation.grid) * 0.66
    layer_2 = jnp.array(observation.grid) * 0.0
    layer_3 = jnp.array(observation.grid) * 0.33
    player_loc = observation.player_locations
    ghost_pos = observation.ghost_locations
    pellets_loc = observation.power_up_locations
    is_scatter = observation.frightened_state_time[0]
    idx = observation.pellet_locations

    # Pellets are light orange
    for i in range(len(idx)):
        if jnp.array(idx[i]).sum != 0:
            loc = idx[i]
            layer_3 = layer_3.at[loc[1], loc[0]].set(1)
            layer_2 = layer_2.at[loc[1], loc[0]].set(0.8)
            layer_1 = layer_1.at[loc[1], loc[0]].set(0.6)

    # Power pellet is purple
    for i in range(len(pellets_loc)):
        p = pellets_loc[i]
        layer_1 = layer_1.at[p[1], p[0]].set(0.5)
        layer_2 = layer_2.at[p[1], p[0]].set(0)
        layer_3 = layer_3.at[p[1], p[0]].set(0.5)

    # Set player is yellow
    layer_1 = layer_1.at[player_loc.x, player_loc.y].set(1)
    layer_2 = layer_2.at[player_loc.x, player_loc.y].set(1)
    layer_3 = layer_3.at[player_loc.x, player_loc.y].set(0)

    cr = jnp.array([1, 1, 0, 1])
    cg = jnp.array([0, 0.7, 1, 0.7])
    cb = jnp.array([0, 1, 1, 0.35])

    layers = (layer_1, layer_2, layer_3)
    scatter = 1 * (is_scatter / 60)

    def set_ghost_colours(
        layers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        layer_1, layer_2, layer_3 = layers
        for i in range(4):
            y = ghost_pos[i][0]
            x = ghost_pos[i][1]
            layer_1 = layer_1.at[x, y].set(cr[0])
            layer_2 = layer_2.at[x, y].set(cg[0] + scatter)
            layer_3 = layer_3.at[x, y].set(cb[0] + scatter)
        return layer_1, layer_2, layer_3

    layers = set_ghost_colours(layers)
    layer_1, layer_2, layer_3 = layers
    layer_1 = layer_1.at[0, 0].set(0)
    layer_2 = layer_2.at[0, 0].set(0)
    layer_3 = layer_3.at[0, 0].set(0)
    obs = [layer_1, layer_2, layer_3]
    rgb = jnp.stack(obs, axis=-1)

    return rgb


def make_network_pac_man(
    pac_man: PacMan,
    critic: bool,
    conv_n_channels: Sequence[int],
    mlp_units: Sequence[int],
    num_actions: int,
) -> FeedForwardNetwork:
    def network_fn(observation: Observation) -> chex.Array:
        conv_layers = [
            [
                hk.Conv2D(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in conv_n_channels
        ]
        torso = hk.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                hk.Flatten(),
            ]
        )

        rgb_observation = process_image(observation)  # (B, G, G, 3)
        obs = rgb_observation.astype(float)

        # Get player position, scatter_time and ghost locations
        player_pos = jnp.array([observation.player_locations.x, observation.player_locations.y])
        player_pos = jnp.stack(player_pos, axis=-1)
        scatter_time = observation.frightened_state_time / 60
        scatter_time = jnp.expand_dims(scatter_time, axis=-1)
        ghost_locations_x = observation.ghost_locations[:, :, 0]
        ghost_locations_y = observation.ghost_locations[:, :, 1]

        # Get shared embedding from RGB data
        embedding = torso(obs)  # (B, H)

        # Concatenate with vector data
        output = output = jnp.concatenate(
            [embedding, player_pos, ghost_locations_x, ghost_locations_y, scatter_time],
            axis=-1,
        )  # (B, H+...)

        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            return jnp.squeeze(head(output), axis=-1)
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            logits = head(output)
            return jnp.where(observation.action_mask, logits, jnp.finfo(jnp.float32).min)

    init, apply = hk.without_apply_rng(hk.transform(network_fn))
    return FeedForwardNetwork(init=init, apply=apply)
