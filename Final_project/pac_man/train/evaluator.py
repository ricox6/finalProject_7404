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
from typing import Any, Dict, Optional, Tuple, List

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax import lax

from Final_project.pac_man.env_basic.environment_basic import Environment
from Final_project.pac_man.agents.a2c_agent import A2CAgent
from Final_project.pac_man.agents.base_agent import Agent
from Final_project.pac_man.agents.random_agent import RandomAgent
from Final_project.pac_man.train.types import ActingState, ParamsState, LSTMState


class Evaluator:
    """Enhanced evaluator with LSTM sequence handling"""

    def __init__(
            self,
            eval_env: Environment,
            agent: Agent,
            total_batch_size: int,
            stochastic: bool,
    ):
        self.eval_env = eval_env
        self.agent = agent
        self.num_local_devices = jax.local_device_count()
        self.num_global_devices = jax.device_count()
        self.num_workers = self.num_global_devices // self.num_local_devices

        if total_batch_size % self.num_global_devices != 0:
            raise ValueError(
                "Expected eval total_batch_size to be a multiple of num_devices, "
                f"got {total_batch_size} and {self.num_global_devices}."
            )

        self.total_batch_size = total_batch_size
        self.batch_size_per_device = total_batch_size // self.num_global_devices
        self.stochastic = stochastic

        # Initialize observation buffer for sequence handling
        self._compiled_eval_fn = self._compile_eval_fn()

        self.generate_evaluations = jax.pmap(
            functools.partial(
                self._generate_evaluations,
                eval_batch_size=self.batch_size_per_device
            ),
            axis_name="devices",
        )

    def _compile_eval_fn(self):
        """Compile evaluation function to avoid tracing issues"""

        def eval_fn(params, key):
            return self._eval_one_episode(params, key)

        return jax.jit(eval_fn)

    def _eval_one_episode(
            self,
            policy_params: Optional[hk.Params],
            key: chex.PRNGKey,
    ) -> Dict[str, chex.Array]:
        reset_key, init_key = jax.random.split(key)
        state, timestep = self.eval_env.reset(reset_key)

        # Create policy function
        policy = self.agent.make_policy(
            policy_params=policy_params,
            stochastic=self.stochastic
        )

        def preserve_dtype(x):
            if isinstance(x, (bool, jnp.bool_)):
                return jnp.asarray(x, dtype=jnp.bool_)
            elif isinstance(x, (int, jnp.integer)):
                return jnp.asarray(x, dtype=jnp.int32)
            elif isinstance(x, (float, jnp.float32, jnp.float64)):
                return jnp.asarray(x, dtype=jnp.float32)
            return jnp.asarray(x)

        if isinstance(self.agent, A2CAgent):
            def init_lstm_state(hidden_size, batch_size):
                lstm = hk.LSTM(hidden_size)
                return lstm.initial_state(batch_size)


            init_lstm_state_fn = hk.transform(init_lstm_state).apply
            init_lstm_params = hk.transform(init_lstm_state).init(
                key, self.agent.lstm_hidden_size, 1  # batch_size=1
            )

            lstm_state = init_lstm_state_fn(
                init_lstm_params, None, self.agent.lstm_hidden_size, 1)  # batch_size=1

            # Initialize observation buffer
            obs_buffer = jax.tree_map(
                lambda x: jnp.stack([x] * self.agent.sequence_length),
                timestep.observation
            )
        else:
            lstm_state = None
            obs_buffer = None

        initial_carry = (
            ActingState(
                state=state,
                timestep=timestep,
                key=init_key,
                episode_count=jnp.array(0, jnp.int32),
                env_step_count=jnp.array(0, jnp.int32),
            ),
            jnp.array(0, float),  # return accumulator
            lstm_state,  # Direct LSTMState, not wrapped in dict
            obs_buffer   # observation buffer
        )

        def cond_fun(carry: Tuple) -> chex.Array:
                acting_state, *_ = carry
                return ~acting_state.timestep.last()

        def body_fun(
                carry: Tuple[ActingState, chex.Array, Optional[LSTMState], Optional[chex.Array]]
        ) -> Tuple[ActingState, chex.Array, Optional[LSTMState], Optional[chex.Array]]:
            acting_state, return_, lstm_state, obs_buffer = carry
            key, action_key = jax.random.split(acting_state.key)

            current_obs = acting_state.timestep.observation

            if isinstance(self.agent, A2CAgent) and obs_buffer is not None:
                # Build input sequence
                obs_seq = [
                    jax.tree_map(lambda x: x[i][None], obs_buffer)
                    for i in range(self.agent.sequence_length)
                ]

                # Update buffer
                new_obs_buffer = jax.tree_map(
                    lambda buf, obs: jnp.concatenate([buf[1:], obs[None]], axis=0),
                    obs_buffer,
                    current_obs
                )
            else:
                obs_seq = current_obs
                new_obs_buffer = None

            # 3. Delegate processing to agent's policy
            action, (_, _), new_lstm_state = policy(obs_seq, action_key, lstm_state)
            # jax.debug.print('action,{action}',action=action)
            # Environment step
            next_state, next_timestep = self.eval_env.step(acting_state.state, action[0])

            # Update carry
            new_acting_state = ActingState(
                state=next_state,
                timestep=next_timestep,
                key=key,
                episode_count=jnp.array(0, jnp.int32),
                env_step_count=acting_state.env_step_count + 1,
            )

            return (
                new_acting_state,  # ActingState
                return_ + next_timestep.reward,
                new_lstm_state,
                new_obs_buffer
            )

        # Run episode
        final_carry = jax.lax.while_loop(
            cond_fun,
            body_fun,
            initial_carry,
        )
        final_acting_state, return_, _, _ = final_carry

        eval_metrics = {
            "episode_return": return_,
            "episode_length": final_acting_state.env_step_count,
        }
        if final_acting_state.timestep.extras:
            eval_metrics.update(final_acting_state.timestep.extras)

        return eval_metrics

    def _generate_evaluations(
            self,
            params_state: ParamsState,
            key: chex.PRNGKey,
            eval_batch_size: int,
    ) -> Dict[str, chex.Array]:
        if isinstance(self.agent, A2CAgent):
            policy_params = params_state.params.actor
        elif isinstance(self.agent, RandomAgent):
            policy_params = None
        else:
            raise ValueError(f"Unsupported agent type: {type(self.agent)}")

        keys = jax.random.split(key, eval_batch_size)
        eval_metrics_all = jax.vmap(self._compiled_eval_fn, in_axes=(None, 0))(
            policy_params,
            keys,
        )

        eval_metrics = jax.tree_map(lambda x: x[0], eval_metrics_all)

        return eval_metrics

    def run_evaluation(
            self,
            params_state: Optional[ParamsState],
            eval_key: chex.PRNGKey
    ) -> Dict[str, chex.Array]:
        eval_keys = jax.random.split(eval_key, self.num_global_devices).reshape(
            self.num_workers, self.num_local_devices, -1
        )
        eval_keys_per_worker = eval_keys[jax.process_index()]

        return self.generate_evaluations(
            params_state,
            eval_keys_per_worker,
        )