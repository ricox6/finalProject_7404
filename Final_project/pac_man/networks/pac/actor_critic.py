from typing import Sequence, Tuple, Optional, NamedTuple, List
import chex
import haiku as hk
import jax
from jax import lax
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
    '''
    def network_fn(
            observation_seq: List[Observation],
            lstm_state: Optional[LSTMState] = None
    ) -> Tuple[chex.Array, LSTMState]:
        # 保持原有处理流程
        batch_size = observation_seq[0].grid.shape[0] if hasattr(observation_seq[0].grid, 'shape') else 1

        processed_features = []
        for obs in observation_seq:
            rgb_obs = process_image(obs)
            conv_out = rgb_obs
            print(rgb_obs.shape)
            for dim in conv_n_channels:
                conv_out = hk.Conv2D(dim, (3, 3))(conv_out)
                conv_out = jax.nn.relu(conv_out)
            conv_out = hk.Flatten()(conv_out)[0]

            # 准备其他特征并确保维度一致
            player_pos = jnp.array([obs.player_locations.x, obs.player_locations.y])  # [2]
            ghost_pos = obs.ghost_locations.flatten()  # [8]
            scatter_time = jnp.array([obs.frightened_state_time / 60.0])  # [1]

            # 确保所有特征都是一维的
            player_pos = jnp.reshape(player_pos, (-1,))  # 强制展平
            ghost_pos = jnp.reshape(ghost_pos, (-1,))  # 强制展平
            scatter_time = jnp.reshape(scatter_time, (-1,))  # 强制展平

            def reduce_dimension(arr, target_dim):
                """将1D数组压缩到目标维度"""
                factor = arr.shape[0] // target_dim
                return jnp.mean(arr.reshape(target_dim, factor), axis=1)

            player_pos = reduce_dimension(player_pos, int(player_pos.shape[0]/batch_size))
            ghost_pos = reduce_dimension(ghost_pos, int(ghost_pos.shape[0]/batch_size))
            scatter_time = reduce_dimension(scatter_time, int(scatter_time.shape[0]/batch_size))
            print(player_pos.shape, ghost_pos.shape)
            features = jnp.concatenate([
                conv_out,
                player_pos,
                ghost_pos,
                scatter_time
            ])
            processed_features.append(features)

        sequence = jnp.stack(processed_features)

        if len(sequence.shape) != 2:
            sequence = jnp.reshape(sequence, (sequence_length, -1))

        # LSTM处理（保持原有结构）
        lstm = hk.LSTM(lstm_hidden_size)

        # 状态初始化（保持原有逻辑）
        try:
            a = lstm_state.hidden
            haiku_state = lstm_state
        except:
            haiku_state = lstm.initial_state(1)

        lstm_input = jnp.expand_dims(sequence, 1)
        output_seq, new_haiku_state = hk.dynamic_unroll(lstm, lstm_input, haiku_state)

        new_state = LSTMState(
            hidden=jnp.asarray(new_haiku_state.hidden),  # 确保转换为JAX数组
            cell=jnp.asarray(new_haiku_state.cell)
        )

        last_output = lax.dynamic_index_in_dim(
            output_seq,
            index=output_seq.shape[0] - 1,  # 取最后时间步
            axis=0,
            keepdims=False  # 移除序列维度
        )

        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            output = head(last_output)
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            logits = head(last_output)
            action_mask = jnp.array(observation_seq[-1].action_mask[-1], bool)
            action_mask=jnp.reshape(action_mask,(1,5))
            output = lax.select(
                action_mask > 0,
                logits,
                jnp.full_like(logits, jnp.finfo(jnp.float32).min)
            )

        return output, new_state

    # 使用transform_with_state确保捕获所有状态
    transformed = hk.transform_with_state(network_fn)

    def init_fn(rng: chex.PRNGKey, obs_seq: List[Observation]):
        # 初始化参数和状态
        params, state = transformed.init(rng, obs_seq, None)
        return params, state

    def apply_fn(params_and_state, obs_seq: List[Observation], lstm_state: Optional[LSTMState] = None):
        # 解包参数和状态
        params, state = params_and_state

        # 应用网络并获取输出和新状态
        (output, new_lstm_state), new_state = transformed.apply(
            params,
            state,
            None,  # rng
            obs_seq,
            lstm_state
        )

        # 合并所有状态
        final_state = {
            **new_state,  # 其他网络状态（如BatchNorm）
            'lstm': new_lstm_state  # LSTM状态
        }

        return output, final_state

    return FeedForwardNetwork(init=init_fn, apply=apply_fn)'''

    def network_fn(
            observation_seq: List[Observation],
            lstm_state: Optional[LSTMState] = None
    ) -> Tuple[chex.Array, LSTMState]:
        # 获取batch大小（唯一需要修改的地方）
        batch_size = observation_seq[0].grid.shape[0] if hasattr(observation_seq[0].grid, 'shape') else 1

        processed_features = []
        for obs in observation_seq:
            rgb_obs = process_image(obs)  # 保持原有process_image不变
            conv_out = rgb_obs
            for dim in conv_n_channels:
                conv_out = hk.Conv2D(dim, (3, 3))(conv_out)
                conv_out = jax.nn.relu(conv_out)
            conv_out = hk.Flatten()(conv_out)  # [B, D_conv]

            # 修改位置特征处理以支持batch
            player_pos = jnp.stack([
                obs.player_locations.x.reshape(batch_size, ),  # [B,1]
                obs.player_locations.y.reshape(batch_size, )  # [B,1]
            ], axis=-1)  # -> [B,2]

            ghost_pos = obs.ghost_locations.reshape(batch_size, 8)  # [B,8]
            scatter_time = (obs.frightened_state_time / 60.0).reshape(batch_size, 1)  # [B,1]

            # 保持原有拼接逻辑
            features = jnp.concatenate([
                conv_out,  # [B, D_conv]
                player_pos,  # [B, 2]
                ghost_pos,  # [B, 8]
                scatter_time  # [B, 1]
            ], axis=-1)  # [B, D_total]
            processed_features.append(features)

        # 序列处理 [T, B, D]
        sequence = jnp.stack(processed_features, axis=0)

        # LSTM处理（保持原有结构）
        lstm = hk.LSTM(lstm_hidden_size)

        # 状态初始化（保持原有逻辑）
        try:
            a = lstm_state.hidden
            haiku_state = lstm_state
        except:
            haiku_state = lstm.initial_state(batch_size)

        # print(sequence.shape)
        output_seq, new_haiku_state = hk.dynamic_unroll(lstm, sequence, haiku_state)
        last_output = output_seq[-1]  # [B, D_lstm]

        # 输出头处理（保持原有逻辑）
        if critic:
            head = hk.nets.MLP((*mlp_units, 1), activate_final=False)
            output = head(last_output)  # [B, 1]
        else:
            head = hk.nets.MLP((*mlp_units, num_actions), activate_final=False)
            logits = head(last_output)  # [B, A]

            # 处理action_mask（确保维度匹配）
            action_mask = jnp.array(observation_seq[-1].action_mask, bool)
            if action_mask.ndim == 1:  # 如果是单样本
                action_mask = jnp.tile(action_mask, (batch_size, 1))  # 广播到[B, A]

            output = jnp.where(
                action_mask,
                logits,
                jnp.finfo(jnp.float32).min
            )

        return output, LSTMState.from_haiku(new_haiku_state)


    # 保持原有transform和函数定义不变
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