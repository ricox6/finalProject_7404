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

from typing import Any, Dict, NamedTuple, Optional, Tuple

import chex
import haiku as hk
import optax

from Final_project.pac_man.env_basic.types_basic import TimeStep


class Transition(NamedTuple):
    """Container for a transition."""

    observation: chex.ArrayTree
    action: chex.ArrayTree
    reward: chex.ArrayTree
    discount: chex.ArrayTree
    next_observation: chex.ArrayTree
    log_prob: chex.ArrayTree
    logits: chex.ArrayTree
    extras: Optional[Dict]


class ActorCriticParams(NamedTuple):
    actor: hk.Params
    critic: hk.Params


class ParamsState(NamedTuple):
    """Container for the variables used during the training of an agent."""

    params: ActorCriticParams
    opt_state: optax.OptState
    update_count: float


class ActingState(NamedTuple):
    """Container for data used during the acting in the environment."""

    state: Any
    timestep: TimeStep
    key: chex.PRNGKey
    episode_count: chex.Array
    env_step_count: chex.Array


class LSTMState(hk.LSTMState):
    """Wrapper around Haiku's LSTMState to maintain compatibility."""

    @classmethod
    def from_haiku(cls, haiku_state: hk.LSTMState) -> 'LSTMState':
        return cls(haiku_state.hidden, haiku_state.cell)

    def to_haiku(self) -> hk.LSTMState:
        return hk.LSTMState(self.hidden, self.cell)

class TrainingState(NamedTuple):
    params_state: ParamsState
    acting_state: ActingState
    lstm_state: LSTMState

