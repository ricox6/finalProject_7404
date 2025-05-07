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

'''
from typing import Any, Callable, NamedTuple

import chex
import haiku as hk


class FeedForwardNetwork(NamedTuple):
    """Networks are meant to take a batch of observations: shape (B, ...)."""

    init: Callable[[chex.PRNGKey, Any], hk.Params]
    apply: Callable[[hk.Params, Any], chex.Array]'''

from typing import Any, Callable, NamedTuple, Optional, Tuple
import chex
import haiku as hk


class FeedForwardNetwork(NamedTuple):
    """Networks with support for recurrent state (e.g. LSTM).

    Modified to handle both stateless and stateful networks:
    - For stateless networks: state can be ignored
    - For stateful networks: init returns (params, state) and apply handles state
    """
    init: Callable[
        [chex.PRNGKey, Any],
        Tuple[hk.Params, Any]  # Returns (params, initial_state)
    ]
    apply: Callable[
        [hk.Params, Any, Optional[Any]],  # Takes (params, inputs, state)
        Tuple[chex.Array, Any]  # Returns (outputs, new_state)
    ]
