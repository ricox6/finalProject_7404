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

from functools import cached_property, partial
from typing import Optional, Sequence, Tuple

import chex
import esquilax
import jax
import jax.numpy as jnp
from esquilax.transforms import spatial
from matplotlib.animation import FuncAnimation

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.swarms.common.types import AgentParams
from jumanji.environments.swarms.common.updates import update_state
from jumanji.environments.swarms.search_and_rescue import utils
from jumanji.environments.swarms.search_and_rescue.dynamics import RandomWalk, TargetDynamics
from jumanji.environments.swarms.search_and_rescue.generator import Generator, RandomGenerator
from jumanji.environments.swarms.search_and_rescue.observations import (
    AgentAndTargetObservationFn,
    ObservationFn,
)
from jumanji.environments.swarms.search_and_rescue.reward import IndividualRewardFn, RewardFn
from jumanji.environments.swarms.search_and_rescue.types import Observation, State, TargetState
from jumanji.environments.swarms.search_and_rescue.viewer import SearchAndRescueViewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class SearchAndRescue(Environment):
    """A multi-agent search environment

    Environment modelling a collection of agents collectively searching
    for a set of targets on a 2d environment. Agents are rewarded
    (individually) for coming within a fixed range of a target that has
    not already been detected. Agents visualise their local environment
    (i.e. the location of other agents and targets) via a simple segmented
    view model. The environment area is a uniform square space with wrapped
    boundaries.

    An episode will terminate if all targets have been located by the team of
    searching agents.

    - observation: `Observation`
        searcher_views: jax array (float) of shape (num_searchers, channels, num_vision)
            Individual local views of positions of other agents and targets, where
            channels can be used to differentiate between agents and targets types.
            Each entry in the view indicates the distance to another agent/target
            along a ray from the agent, and is -1.0 if nothing is in range along the ray.
            The view model can be customised by implementing the  `ObservationFn` interface.
        targets_remaining: (float) Number of targets remaining to be found from
            the total scaled to the range [0, 1] (i.e. a value of 1.0 indicates
            all the targets are still to be found).
        step: (int) current simulation step.
        positions: jax array (float) of shape (num_searchers, 2) search agent positions.

    - action: jax array (float) of shape (num_searchers, 2)
        Array of individual agent actions. Each agents actions rotate and
        accelerate/decelerate the agent as [rotation, acceleration] on the range
        [-1, 1]. These values are then scaled to update agent velocities within
        given parameters (i.e. a value of -+1 is the maximum acceleration/rotation).

    - reward: jax array (float) of shape (num_searchers,)
        Arrays of individual agent rewards. A reward of +1 is granted when an agent
        comes into contact range with a target that has not yet been found, and
        the target is within the searchers view cone. It is possible for multiple
        agents to newly find the same target within a given step, by default
        in this case the reward is split between the locating agents. By default,
        rewards granted linearly decrease over time, with zero reward granted
        at the environment time-limit. These defaults can be modified by flags
        in `IndividualRewardFn`, or further customised by  implementing the `RewardFn`
        interface.

    - state: `State`
        - searchers: `AgentState`
            - pos: jax array (float) of shape (num_searchers, 2) in the range [0, env_size].
            - heading: jax array (float) of shape (num_searcher,) in
                the range [0, 2π].
            - speed: jax array (float) of shape (num_searchers,) in the
                range [min_speed, max_speed].
        - targets: `TargetState`
            - pos: jax array (float) of shape (num_targets, 2) in the range [0, env_size].
            - vel:  jax array (float) of shape (num_targets, 2).
            - found: jax array (bool) of shape (num_targets,) flag indicating if
                target has been located by an agent.
        - key: jax array (uint32) of shape (2,)
        - step: int representing the current simulation step.

    ```python
    from jumanji.environments import SearchAndRescue

    env = SearchAndRescue()
    key = jax.random.PRNGKey(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec.generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```
    """

    def __init__(
        self,
        target_contact_range: float = 0.02,
        searcher_max_rotate: float = 0.25,
        searcher_max_accelerate: float = 0.005,
        searcher_min_speed: float = 0.005,
        searcher_max_speed: float = 0.02,
        time_limit: int = 400,
        viewer: Optional[Viewer[State]] = None,
        target_dynamics: Optional[TargetDynamics] = None,
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
        observation: Optional[ObservationFn] = None,
    ) -> None:
        """Instantiates a `SearchAndRescue` environment

        Args:
            target_contact_range: Range at which a searchers will 'find' a target.
            searcher_max_rotate: Maximum rotation searcher agents can
                turn within a step. Should be a value from [0,1]
                representing a fraction of π-radians.
            searcher_max_accelerate: Magnitude of the maximum
                acceleration/deceleration a searcher agent can apply within a step.
            searcher_min_speed: Minimum speed a searcher agent can move at.
            searcher_max_speed: Maximum speed a searcher agent can move at.
            time_limit: Maximum number of environment steps allowed for search.
            viewer: `Viewer` used for rendering. Defaults to `SearchAndRescueViewer`.
            target_dynamics: Target object dynamics model, implemented as a
                `TargetDynamics` interface. Defaults to `RandomWalk`.
            generator: Initial state `Generator` instance. Defaults to `RandomGenerator`
                with 50 targets and 2 searchers, with targets uniformly distributed
                across the environment.
            reward_fn: Reward aggregation function. Defaults to `IndividualRewardFn` where
                agents split rewards if they locate a target simultaneously, and
                rewards linearly decrease to zero over time.
            observation: Agent observation view generation function. Defaults to
                `AgentAndTargetObservationFn` where all targets (found and unfound)
                and other searching agents are included in the generated view.
        """

        self.target_contact_range = target_contact_range

        self.searcher_params = AgentParams(
            max_rotate=searcher_max_rotate,
            max_accelerate=searcher_max_accelerate,
            min_speed=searcher_min_speed,
            max_speed=searcher_max_speed,
        )
        self.time_limit = time_limit
        self._target_dynamics = target_dynamics or RandomWalk(acc_std=0.0001, vel_max=0.002)
        self.generator = generator or RandomGenerator(num_targets=40, num_searchers=2)
        self._viewer = viewer or SearchAndRescueViewer()
        self._reward_fn = reward_fn or IndividualRewardFn()
        self._observation_fn = observation or AgentAndTargetObservationFn(
            num_vision=128,
            searcher_vision_range=0.4,
            target_vision_range=0.1,
            view_angle=0.4,
            agent_radius=target_contact_range,
            env_size=self.generator.env_size,
        )
        super().__init__()

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Search & rescue multi-agent environment:",
                f" - num searchers: {self.generator.num_searchers}",
                f" - num targets: {self.generator.num_targets}",
                f" - max searcher rotation: {self.searcher_params.max_rotate}",
                f" - max searcher acceleration: {self.searcher_params.max_accelerate}",
                f" - searcher min speed: {self.searcher_params.min_speed}",
                f" - searcher max speed: {self.searcher_params.max_speed}",
                f" - search vision range: {self._observation_fn.searcher_vision_range}",
                f" - target vision range: {self._observation_fn.target_vision_range}",
                f" - search view angle: {self._observation_fn.view_angle}",
                f" - target contact range: {self.target_contact_range}",
                f" - num vision: {self._observation_fn.num_vision}",
                f" - agent radius: {self._observation_fn.agent_radius}",
                f" - time limit: {self.time_limit},"
                f" - env size: {self.generator.env_size}"
                f" - target dynamics: {self._target_dynamics.__class__.__name__}",
                f" - generator: {self.generator.__class__.__name__}",
                f" - reward fn: {self._reward_fn.__class__.__name__}",
                f" - observation fn: {self._observation_fn.__class__.__name__}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Initialise searcher and target initial states.

        Args:
            key: Random key used to reset the environment.

        Returns:
            state: Initial environment state.
            timestep: TimeStep with individual search agent views.
        """
        state = self.generator(key, self.searcher_params)
        timestep = restart(observation=self._state_to_observation(state), shape=(self.num_agents,))
        return state, timestep

    def step(self, state: State, actions: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Environment update.

        Update searcher velocities and consequently their positions,
        mark found targets, and generate rewards and local observations.

        Args:
            state: Environment state.
            actions: 2d array of searcher steering actions.

        Returns:
            state: Updated searcher and target positions and velocities.
            timestep: Transition timestep with individual agent local observations.
        """
        key, target_key = jax.random.split(state.key, num=2)
        searchers = update_state(
            self.generator.env_size, self.searcher_params, state.searchers, actions
        )
        targets = self._target_dynamics(target_key, state.targets, self.generator.env_size)

        # Searchers return an array of flags of any targets they are in range of,
        #  and that have not already been located, result shape here is (n-searcher, n-targets)
        targets_found = spatial(
            utils.searcher_detect_targets,
            reduction=esquilax.reductions.logical_or((self.generator.num_targets,)),
            i_range=self.target_contact_range,
            dims=self.generator.env_size,
        )(
            self._observation_fn.view_angle,
            searchers,
            (jnp.arange(self.generator.num_targets), targets),
            pos=searchers.pos,
            pos_b=targets.pos,
            env_size=self.generator.env_size,
            n_targets=self.generator.num_targets,
        )

        rewards = self._reward_fn(targets_found, state.step, self.time_limit)

        targets_found = jnp.any(targets_found, axis=0)
        # Targets need to remain found if they already have been
        targets_found = jnp.logical_or(targets_found, state.targets.found)

        state = State(
            searchers=searchers,
            targets=TargetState(pos=targets.pos, vel=targets.vel, found=targets_found),
            key=key,
            step=state.step + 1,
        )
        observation = self._state_to_observation(state)
        observation = jax.lax.stop_gradient(observation)
        timestep = jax.lax.cond(
            jnp.logical_or(state.step >= self.time_limit, jnp.all(targets_found)),
            partial(termination, shape=(self.num_agents,)),
            partial(transition, shape=(self.num_agents,)),
            rewards,
            observation,
        )
        return state, timestep

    def _state_to_observation(self, state: State) -> Observation:
        searcher_views = self._observation_fn(state)
        return Observation(
            searcher_views=searcher_views,
            targets_remaining=1.0 - jnp.sum(state.targets.found) / self.generator.num_targets,
            step=state.step,
            positions=state.searchers.pos / self.generator.env_size,
        )

    @cached_property
    def num_agents(self) -> int:
        return self.generator.num_searchers

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Returns the observation spec.

        Local searcher agent views representing the distance to the
        closest neighbouring agents and targets in the environment.

        Returns:
            observation_spec: Search-and-rescue observation spec
        """
        searcher_views = specs.BoundedArray(
            shape=(
                self.num_agents,
                self._observation_fn.num_channels,
                self._observation_fn.num_vision,
            ),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
            name="searcher_views",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            searcher_views=searcher_views,
            targets_remaining=specs.BoundedArray(
                shape=(), minimum=0.0, maximum=1.0, name="targets_remaining", dtype=jnp.float32
            ),
            step=specs.BoundedArray(
                shape=(), minimum=0, maximum=self.time_limit, name="step", dtype=jnp.int32
            ),
            positions=specs.BoundedArray(
                shape=(self.num_agents, 2),
                minimum=0.0,
                maximum=1.0,
                name="positions",
                dtype=jnp.float32,
            ),
        )

    @cached_property
    def action_spec(self) -> specs.BoundedArray:
        """Returns the action spec.

        2d array of individual agent actions. Each agents action is
        an array representing [rotation, acceleration] in the range
        [-1, 1].

        Returns:
            action_spec: Action array spec
        """
        return specs.BoundedArray(
            shape=(self.generator.num_searchers, 2),
            minimum=-1.0,
            maximum=1.0,
            dtype=float,
        )

    @cached_property
    def reward_spec(self) -> specs.BoundedArray:
        """Returns the reward spec.

        Array of individual rewards for each agent.

        Returns:
            reward_spec: Reward array spec.
        """
        return specs.BoundedArray(
            shape=(self.generator.num_searchers,),
            minimum=0.0,
            maximum=float(self.generator.num_targets),
            dtype=float,
        )

    def render(self, state: State) -> None:
        """Render a frame of the environment for a given state using matplotlib.

        Args:
            state: State object.
        """
        self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 100,
        save_path: Optional[str] = None,
    ) -> FuncAnimation:
        """Create an animation from a sequence of environment states.

        Args:
            states: sequence of environment states corresponding to consecutive
                timesteps.
            interval: delay between frames in milliseconds.
            save_path: the path where the animation file should be saved. If it
                is None, the plot will not be saved.

        Returns:
            Animation that can be saved as a GIF, MP4, or rendered with HTML.
        """
        return self._viewer.animate(states, interval=interval, save_path=save_path)

    def close(self) -> None:
        """Perform any necessary cleanup."""
        self._viewer.close()
