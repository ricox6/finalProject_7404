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

from typing import NamedTuple

from .base import FeedForwardNetwork
from .parametric_distribution import ParametricDistribution


class ActorCriticNetworks(NamedTuple):
    """Defines the actor-critic networks, which outputs the logits of a policy, and a value given
    an observation. The assumption is that the networks are given a batch of observations.
    """

    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution
