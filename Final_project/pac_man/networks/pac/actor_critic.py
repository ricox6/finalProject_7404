from typing import Sequence, Tuple, Optional, NamedTuple, List
import chex
import haiku as hk
import jax
from future.backports.http.cookiejar import debug
from jax import debug
import jax.numpy as jnp
# import numpy as np
from jumanji.environments.routing.pac_man import Observation, PacMan
from Final_project.pac_man.networks.actor_critic import (
    ActorCriticNetworks,
    FeedForwardNetwork,
)
from Final_project.pac_man.networks.parametric_distribution import (
    CategoricalParametricDistribution,
)
from Final_project.pac_man.train.types import LSTMState


def make_actor_critic_networks_pacman(
    pac_man: PacMan,
    num_channels: Sequence[int],
    policy_layers: Sequence[int],
    value_layers: Sequence[int],
    lstm_hidden_size: int = 128,
    sequence_length: int = 10
) -> ActorCriticNetworks:
    """Make actor-critic networks with sequence processing."""
    num_actions = int(pac_man.action_spec.num_values)
    parametric_action_distribution = CategoricalParametricDistribution(num_actions=num_actions)

    policy_network = make_network_pac_man(
        pac_man=pac_man,
        critic=False,
        conv_n_channels=num_channels,
        mlp_units=policy_layers,
        num_actions=num_actions,
        lstm_hidden_size=lstm_hidden_size,
        sequence_length=sequence_length
    )

    value_network = make_network_pac_man(
        pac_man=pac_man,
        critic=True,
        conv_n_channels=num_channels,
        mlp_units=value_layers,
        num_actions=num_actions,
        lstm_hidden_size=lstm_hidden_size,
        sequence_length=sequence_length
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
        lstm_hidden_size: int = 128,
        sequence_length: int = 10
) -> FeedForwardNetwork:
    """Network with built-in LSTM state handling without structural changes"""

    def network_fn(
            observation_seq: List[Observation],
            lstm_state: Optional[LSTMState] = None
    ) -> Tuple[chex.Array, LSTMState]:
        batch_size = observation_seq[0].grid.shape[0] if hasattr(observation_seq[0].grid, 'shape') else 1
        processed_features = []
        for obs in observation_seq:
            rgb_obs = process_image(obs)
            conv_out = rgb_obs

            for dim in conv_n_channels:
                conv_out = hk.Conv2D(dim, (3, 3))(conv_out)
                conv_out = jax.nn.relu(conv_out)
            conv_out = hk.Flatten()(conv_out)  # [B, D_conv]

            try:
                player_pos = jnp.array([obs.player_locations.x, obs.player_locations.y])
                player_pos = jnp.stack(player_pos, axis=-1)
                scatter_time = obs.frightened_state_time / 60
                scatter_time = jnp.expand_dims(scatter_time, axis=-1)
                ghost_locations_x = obs.ghost_locations[:, :, 0]
                ghost_locations_y = obs.ghost_locations[:, :, 1]

                features = jnp.concatenate([
                    conv_out,  # [B, D_conv]
                    player_pos,  # [B, 2]
                    ghost_locations_x,
                    ghost_locations_y,# [B, 8]
                    scatter_time  # [B, 1]
                ], axis=-1)# [B, D_total]


            except:
                player_pos = jnp.array([obs.player_locations.x, obs.player_locations.y])
                player_pos = jnp.stack(player_pos, axis=-1)
                scatter_time = obs.frightened_state_time / 60
                scatter_time = jnp.expand_dims(scatter_time, axis=-1)
                ghost_locations_x = obs.ghost_locations[:, :, 0]
                ghost_locations_y = obs.ghost_locations[:, :, 1]

                player_pos = jnp.expand_dims(player_pos, 0)

                features = jnp.concatenate([
                    conv_out,  # [B, D_conv]
                    player_pos,  # [B, 2]
                    ghost_locations_x,
                    ghost_locations_y,  # [B, 8]
                    scatter_time  # [B, 1]
                ], axis=-1)  # [B, D_total]

            processed_features.append(features)

        sequence = jnp.stack(processed_features, axis=0)

        lstm = hk.LSTM(lstm_hidden_size)

        try:
            a = lstm_state.hidden
            haiku_state = lstm_state
        except:
            haiku_state = lstm.initial_state(batch_size)

        output_seq, new_haiku_state = hk.dynamic_unroll(lstm, sequence, haiku_state)
        last_output = output_seq[-1]

        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            output = head(last_output)  # [B, 1]
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            logits = head(last_output)  # [B, A]

            action_mask = jnp.array(observation_seq[-1].action_mask, bool)
            if action_mask.ndim == 1:
                action_mask = jnp.tile(action_mask, (batch_size, 1))
            key=jax.random.PRNGKey(0)
            logits=logits+0.5*jax.random.normal(key,logits.shape)
            output = jnp.where(
                action_mask,
                logits,
                jnp.finfo(jnp.float32).min
            )

        return output, LSTMState.from_haiku(new_haiku_state)

    transformed = hk.transform_with_state(network_fn)


    def init_fn(rng: chex.PRNGKey, obs_seq: List[Observation]):
        params, state = transformed.init(rng, obs_seq, None)
        return params, state


    def apply_fn(params_and_state, obs_seq: List[Observation], lstm_state: Optional[LSTMState] = None):
        params, state = params_and_state
        (output, new_lstm_state), new_state = transformed.apply(
            params, state, None, obs_seq, lstm_state
        )
        return output, {'lstm': new_lstm_state, **new_state}


    return FeedForwardNetwork(init=init_fn, apply=apply_fn)