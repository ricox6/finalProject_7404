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

import abc
from typing import Callable, Dict, Optional, Tuple, Any
import chex
import haiku as hk
import jax
from sympy import sequence

from Final_project.pac_man.train.types import ParamsState, TrainingState,LSTMState


class Agent(abc.ABC):
    """Anakin agent."""

    def __init__(self, total_batch_size: int):
        self.total_batch_size = total_batch_size
        num_devices = jax.local_device_count()
        assert total_batch_size % num_devices == 0, (
            "The total batch size must be a multiple of the number of devices, "
            f"got total_batch_size={total_batch_size} and num_devices={num_devices}."
        )
        self.batch_size_per_device = total_batch_size // num_devices
        self.sequence_length = 10
    @abc.abstractmethod
    def init_params(self, key: chex.PRNGKey) -> Optional[ParamsState]:
        pass

    @abc.abstractmethod
    def run_epoch(self, training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        pass

    @abc.abstractmethod
    def make_policy(
            self,
            policy_params: hk.Params,
            stochastic: bool = True,
    ) -> Callable[[Any, chex.PRNGKey, Optional[LSTMState]],
    Tuple[chex.Array, Any, Optional[LSTMState]]]:
        pass

    def act(self, policy_params: hk.Params, timestep: Any, lstm_state: Optional[LSTMState] = None, key: Optional[chex.PRNGKey] = None
    ) -> Tuple[chex.Array, LSTMState]:
        pass
