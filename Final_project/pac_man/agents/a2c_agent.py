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

import functools
from typing import Any, Callable, Dict, Tuple, List, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
from jax import debug

from Final_project.pac_man.env_basic.environment_basic import Environment
from .base_agent import Agent
from Final_project.pac_man.networks.actor_critic import ActorCriticNetworks
from Final_project.pac_man.train.types import (
    ActingState,
    ActorCriticParams,
    ParamsState,
    TrainingState,
    Transition,
    LSTMState,
)


class A2CAgent(Agent):
    def __init__(
            self,
            env: Environment,
            n_steps: int,
            total_batch_size: int,
            actor_critic_networks: ActorCriticNetworks,
            optimizer: optax.GradientTransformation,
            normalize_advantage: bool,
            discount_factor: float,
            bootstrapping_factor: float,
            l_pg: float,
            l_td: float,
            l_en: float,
            lstm_hidden_size: int = 128,
            sequence_length: int = 10,
    ) -> None:
        super().__init__(total_batch_size=total_batch_size)
        self.env = env
        self.observation_spec = env.observation_spec
        self.n_steps = n_steps
        self.actor_critic_networks = actor_critic_networks
        self.optimizer = optimizer
        self.normalize_advantage = normalize_advantage
        self.discount_factor = discount_factor
        self.bootstrapping_factor = bootstrapping_factor
        self.l_pg = l_pg
        self.l_td = l_td
        self.l_en = l_en
        self.lstm_hidden_size=lstm_hidden_size
        self.sequence_length = sequence_length

    def init_params(self, key: chex.PRNGKey) -> ParamsState:
        actor_key, critic_key = jax.random.split(key)
        dummy_obs = self.observation_spec.generate_value()

        dummy_obs_seq = jax.tree_map(
            lambda x: jnp.stack([x] * self.sequence_length),  # (4,H,W)
            dummy_obs
        )
        dummy_obs_seq = [
            jax.tree_map(lambda x: x[i][None], dummy_obs_seq)
            for i in range(self.sequence_length)
        ]

        params = ActorCriticParams(
            actor=self.actor_critic_networks.policy_network.init(actor_key, dummy_obs_seq),
            critic=self.actor_critic_networks.value_network.init(critic_key, dummy_obs_seq),
        )

        opt_state = self.optimizer.init(params)
        return ParamsState(
            params=params,
            opt_state=opt_state,
            update_count=jnp.array(0, float),
        )


    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        if not isinstance(training_state.params_state, ParamsState):
            raise TypeError(
                "Expected params_state to be of type ParamsState, got "
                f"type {type(training_state.params_state)}."
            )

        grad, (acting_state, metrics, new_lstm_state) = jax.grad(self.a2c_loss, has_aux=True)(
            training_state.params_state.params,
            training_state.acting_state,
            training_state.lstm_state
        )

        grad, metrics = jax.lax.pmean((grad, metrics), "devices")
        updates, opt_state = self.optimizer.update(grad, training_state.params_state.opt_state)
        params = optax.apply_updates(training_state.params_state.params, updates)

        training_state = TrainingState(
            params_state=ParamsState(
                params=params,
                opt_state=opt_state,
                update_count=training_state.params_state.update_count + 1,
            ),
            acting_state=acting_state,
            lstm_state=new_lstm_state
        )
        return training_state, metrics

    def a2c_loss(
            self,
            params: ActorCriticParams,
            acting_state: ActingState,
            lstm_state: Optional[LSTMState] = None
    ) -> Tuple[float, Tuple[ActingState, Dict, LSTMState]]:
        parametric_action_distribution = self.actor_critic_networks.parametric_action_distribution
        value_apply = self.actor_critic_networks.value_network.apply

        acting_state, data, new_lstm_state = self.rollout(
            policy_params=params.actor,
            acting_state=acting_state,
            lstm_state=lstm_state
        )
        last_observation = jax.tree_map(lambda x: x[-1], data.next_observation)
        observation = jax.tree_util.tree_map(
            lambda obs_0_tm1, obs_t: jnp.concatenate([obs_0_tm1, obs_t[None]], axis=0),
            data.observation,
            last_observation,
        )

        def compute_value(obs):
            return value_apply(params.critic, [obs], None)[0]

        value = jax.vmap(compute_value)(observation)
        value = value.squeeze(-1)

        discounts = jnp.asarray(self.discount_factor * data.discount, float)
        value_tm1 = value[:-1]
        value_t = value[1:]

        advantage = jax.vmap(
            functools.partial(
                rlax.td_lambda,
                lambda_=self.bootstrapping_factor,
                stop_target_gradients=True,
            ),
            in_axes=1,
            out_axes=1,
        )(value_tm1, data.reward, discounts, value_t)

        critic_loss = jnp.mean(advantage ** 2)

        metrics: Dict = {}
        if self.normalize_advantage:
            metrics.update(unnormalized_advantage=jnp.mean(advantage))
            advantage = jax.nn.standardize(advantage)

        policy_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * data.log_prob)
        entropy = jnp.mean(parametric_action_distribution.entropy(data.logits, acting_state.key))
        entropy_loss = -entropy

        total_loss = self.l_pg * policy_loss + self.l_td * critic_loss + self.l_en * entropy_loss

        metrics.update(
            total_loss=total_loss,
            policy_loss=policy_loss,
            critic_loss=critic_loss,
            entropy_loss=entropy_loss,
            entropy=entropy,
            advantage=jnp.mean(advantage),
            value=jnp.mean(value),
        )

        if data.extras:
            metrics.update(data.extras)

        return total_loss, (acting_state, metrics, new_lstm_state)

    def make_policy(
            self,
            policy_params: hk.Params,
            stochastic: bool = True,
    ) -> Callable[[List[Any], chex.PRNGKey, LSTMState], Tuple[chex.Array, Tuple[chex.Array, chex.Array], LSTMState]]:
        policy_network = self.actor_critic_networks.policy_network
        parametric_action_distribution = self.actor_critic_networks.parametric_action_distribution

        def policy(
                observation_seq: List[Any],
                key: chex.PRNGKey,
                lstm_state: Optional[LSTMState] = None
        ) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array], LSTMState]:
            if lstm_state is not None:
                if isinstance(lstm_state, dict):
                        try:
                            haiku_state = hk.LSTMState(
                                hidden=lstm_state['lstm'].hidden,
                                cell=lstm_state['lstm'].cell)
                        except:
                            haiku_state = hk.LSTMState(
                                hidden=lstm_state['hidden'],
                                cell=lstm_state['cell']
                            )

                elif hasattr(lstm_state, 'to_haiku'):
                    haiku_state = lstm_state.to_haiku()
                else:
                    haiku_state = None
            else:
                haiku_state = None

            processed_seq = []
            for obs in observation_seq:
                if isinstance(obs, (jnp.ndarray, jnp.ndarray)):
                    processed_seq.append(jax.tree_map(lambda x: x, obs))
                else:
                    processed_seq.append(obs)


            if not isinstance(observation_seq, list):
                observation_seq = [observation_seq]


            processed_seq = []
            for obs in observation_seq:
                if isinstance(obs, (jnp.ndarray, jnp.ndarray)):
                    processed_seq.append(jax.tree_map(lambda x: x[:1], obs))
                else:
                    processed_seq.append(obs)

            logits, new_lstm_state = policy_network.apply(
                policy_params,
                processed_seq,
                haiku_state
            )
            if new_lstm_state is not None:
                if isinstance(new_lstm_state, dict):
                    new_lstm_state = hk.LSTMState(
                        hidden=new_lstm_state['lstm'].hidden,
                        cell=new_lstm_state['lstm'].cell
                    )
                elif hasattr(new_lstm_state, 'to_haiku'):
                    new_lstm_state = lstm_state.to_haiku()
            else:
                new_lstm_state = None

            if stochastic:
                raw_action = parametric_action_distribution.sample_no_postprocessing(logits, key)
                log_prob = parametric_action_distribution.log_prob(logits, raw_action)
            else:
                raw_action = parametric_action_distribution.mode_no_postprocessing(logits)
                log_prob = jnp.zeros_like(parametric_action_distribution.log_prob(logits, raw_action))

            action = parametric_action_distribution.postprocess(raw_action)
            return action, (log_prob, logits), new_lstm_state

        return policy

    def rollout(self, policy_params, acting_state, lstm_state=None):
        policy = self.make_policy(policy_params, stochastic=True)

        batch_size = 1

        obs_buffer = [
                         jax.tree_map(lambda x: x, acting_state.timestep.observation)  # 直接使用已有batch维度
                     ] * self.sequence_length

        def run_one_step(carry, key):
            acting_state, lstm_state, obs_buffer = carry

            obs_seq = [
                jax.tree_map(lambda x: x, obs)
                for obs in obs_buffer
            ]
            if isinstance(lstm_state, LSTMState):
                lstm_state = lstm_state.to_haiku()
            else:
                lstm_state = lstm_state

            action, (log_prob, logits), new_lstm_state = policy(
                obs_seq, key, lstm_state
            )

            next_state, next_timestep = self.env.step(
                acting_state.state,
                action
            )

            new_obs = jax.tree_map(lambda x: x, next_timestep.observation)
            new_buffer = obs_buffer[1:] + [new_obs]

            transition = Transition(
                observation=jax.tree_map(lambda x: x, acting_state.timestep.observation),
                action=action,
                reward=next_timestep.reward,
                discount=next_timestep.discount,
                next_observation=jax.tree_map(lambda x: x, next_timestep.observation),
                log_prob=log_prob,
                logits=logits,
                extras=next_timestep.extras,
            )

            new_acting_state = ActingState(
                state=next_state,
                timestep=next_timestep,
                key=acting_state.key,
                episode_count=acting_state.episode_count
                              + jax.lax.psum(next_timestep.last().sum(), "devices"),
                env_step_count=acting_state.env_step_count
                               + jax.lax.psum(self.batch_size_per_device, "devices"),
            )

            if isinstance(new_lstm_state, LSTMState):
                new_lstm_state = new_lstm_state.to_haiku()
            else:
                new_lstm_state = new_lstm_state

            return (new_acting_state, new_lstm_state, new_buffer), transition

        keys = jax.random.split(acting_state.key, self.n_steps)

        (final_state, final_lstm, _), all_data = jax.lax.scan(
            run_one_step,
            (acting_state, lstm_state, obs_buffer),
            keys
        )

        return final_state, all_data, final_lstm

    def act(
            self,
            policy_params: hk.Params,
            timestep: Any,
            lstm_state: Optional[LSTMState] = None,
            key: Optional[chex.PRNGKey] = None
    ) -> Tuple[chex.Array, LSTMState]:
        parametric_action_distribution = self.actor_critic_networks.parametric_action_distribution

        if key is None:
            key = jax.random.PRNGKey(0)

        observation = timestep.observation

        obs_seq = [
                      jax.tree_map(
                          lambda x: x[None] if isinstance(x, (jnp.ndarray, jnp.ndarray)) else x,
                          observation
                      )
                  ] * self.sequence_length

        haiku_state = None
        if lstm_state is not None:
            haiku_state = hk.LSTMState(
                hidden=lstm_state.hidden,
                cell=lstm_state.cell
            )

        action, (log_prob, logits), new_haiku_state = self.make_policy(
            policy_params=policy_params,
            stochastic=False
        )(obs_seq, key, haiku_state)

        new_lstm_state = None
        if new_haiku_state is not None:
            new_lstm_state = LSTMState(
                hidden=new_haiku_state.hidden,
                cell=new_haiku_state.cell
            )

        raw_action = parametric_action_distribution.mode_no_postprocessing(logits)
        action = parametric_action_distribution.postprocess(raw_action)

        return action, new_lstm_state