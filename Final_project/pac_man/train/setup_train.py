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

from typing import Tuple, Optional, List, Dict, Any
import chex
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig
import haiku as hk

import jumanji
from Final_project.pac_man.env_basic.environment_basic import Environment
from Final_project.pac_man.environment.env import PacMan
from jumanji.training import networks
from Final_project.pac_man.agents.a2c_agent import A2CAgent
from Final_project.pac_man.agents.base_agent import Agent
from Final_project.pac_man.agents.random_agent import RandomAgent
from evaluator import Evaluator
from loggers import (
    Logger,
    NeptuneLogger,
    NoOpLogger,
    TensorboardLogger,
    TerminalLogger,
)
from Final_project.pac_man.networks.actor_critic import ActorCriticNetworks
from Final_project.pac_man.networks.protocols import RandomPolicy
from Final_project.pac_man.train.types import ActingState, TrainingState, LSTMState
from wrappers import MultiToSingleWrapper, VmapAutoResetWrapper

from Final_project.pac_man.environment.viewer import PacManViewer
from Final_project.pac_man.environment.types import State
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def setup_logger(cfg: DictConfig) -> Logger:
    logger: Logger
    if jax.process_index() != 0:
        return NoOpLogger()
    if cfg.logger.type == "tensorboard":
        logger = TensorboardLogger(name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint)
    elif cfg.logger.type == "neptune":
        logger = NeptuneLogger(
            name=cfg.logger.name,
            project="InstaDeep/jumanji",
            cfg=cfg,
            save_checkpoint=cfg.logger.save_checkpoint,
        )
    elif cfg.logger.type == "terminal":
        logger = TerminalLogger(name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint)
    else:
        raise ValueError(
            f"logger expected in ['neptune', 'tensorboard', 'terminal'], got {cfg.logger}."
        )
    return logger


def _make_raw_env(cfg: DictConfig) -> Environment:
    env = PacMan()
    return env


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapper(env)
    return env


def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent == "random":
        random_policy = _setup_random_policy(cfg, env)
        agent = RandomAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            random_policy=random_policy,
        )
    elif cfg.agent == "a2c":
        actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        agent = A2CAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.env.a2c.normalize_advantage,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
            sequence_length=cfg.env.network.get("sequence_length", 10)
        )
    else:
        raise ValueError(f"Expected agent name to be in ['random', 'a2c'], got {cfg.agent}.")
    return agent


def _setup_random_policy(cfg: DictConfig, env: Environment) -> RandomPolicy:
    assert cfg.agent == "random"
    if cfg.env.name == "pac_man":
        assert isinstance(env.unwrapped, PacMan)
        random_policy = networks.make_random_policy_pacman()
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return random_policy


def _setup_actor_critic_neworks(cfg: DictConfig, env: Environment) -> ActorCriticNetworks:
    assert cfg.agent == "a2c"
    if cfg.env.name == "pac_man":
        assert isinstance(env.unwrapped, PacMan)
        actor_critic_networks = networks.make_actor_critic_networks_pacman(
            pac_man=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
            lstm_hidden_size=cfg.env.network.lstm_hidden_size,
            sequence_length=cfg.env.network.get("sequence_length", 10)
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return actor_critic_networks


def setup_evaluators(cfg: DictConfig, agent: Agent) -> Tuple[Evaluator, Evaluator]:
    env = _make_raw_env(cfg)
    stochastic_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.eval_total_batch_size,
        stochastic=True,
    )
    greedy_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.greedy_eval_total_batch_size,
        stochastic=False,
    )
    return stochastic_eval, greedy_eval


def setup_training_state(env: Environment, agent: Agent, key: chex.PRNGKey) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)
    params_state = agent.init_params(params_key)
    num_local_devices = jax.local_device_count()
    num_global_devices = jax.device_count()
    num_workers = num_global_devices // num_local_devices
    local_batch_size = agent.total_batch_size // num_global_devices

    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (num_workers, num_local_devices, local_batch_size, -1)
    )
    reset_keys_per_worker = reset_keys[jax.process_index()]
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys_per_worker)

    acting_key_per_device = jax.random.split(acting_key, num_global_devices).reshape(
        num_workers, num_local_devices, -1
    )
    acting_key_per_worker_device = acting_key_per_device[jax.process_index()]

    def init_lstm_state(hidden_size, batch_size):
        lstm = hk.LSTM(hidden_size)
        return lstm.initial_state(batch_size)

    init_lstm_state_fn = hk.transform(init_lstm_state).apply
    init_lstm_params = hk.transform(init_lstm_state).init(
        params_key, agent.lstm_hidden_size, 1
    )

    if hasattr(agent, 'lstm_hidden_size'):
        haiku_state = init_lstm_state_fn(
            init_lstm_params, None, agent.lstm_hidden_size, 1
        )
        lstm_state = LSTMState.from_haiku(haiku_state)
        lstm_state = jax.tree_map(
            lambda x: jnp.broadcast_to(x, (num_local_devices, *x.shape)),
            lstm_state
        )
    else:
        lstm_state = None
    if isinstance(lstm_state, LSTMState):
        lstm_state = lstm_state.to_haiku()
    else:
        lstm_state = lstm_state

    return TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=ActingState(
            state=env_state,
            timestep=timestep,
            key=acting_key_per_worker_device,
            episode_count=jnp.zeros(num_local_devices, float),
            env_step_count=jnp.zeros(num_local_devices, float)
        ),
        lstm_state=lstm_state
    )


def visualize_training(states: List[State], save_path: Optional[str] = None) -> None:
    """Visualize the training process using the PacManViewer.

    Args:
        states: List of states collected during training.
        save_path: Optional path to save the animation as a GIF.
    """

    viewer = PacManViewer(name="PacMan Training", render_mode="human")


    frames = []
    for state in states:
        frame = viewer.render(state)
        frames.append(frame)


    if save_path:
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.imshow(frame)
            return ax,

        anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=200,
            blit=True
        )
        anim.save(save_path, writer="pillow", fps=5)
        plt.close(fig)

    plt.show()


def collect_training_states(
        agent: Agent,
        env: Environment,
        num_episodes: int = 1,
        max_steps: int = 1000
) -> List[State]:
    """Collect states during training for visualization.

    Args:
        agent: The trained agent.
        env: The environment.
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.

    Returns:
        List of states collected during the episodes.
    """
    collected_states = []
    key = jax.random.PRNGKey(0)

    for _ in range(num_episodes):
        state, _ = env.reset(key)
        collected_states.append(state)

        for _ in range(max_steps):
            action = agent.act(state)
            state, _ = env.step(state, action)
            collected_states.append(state)

            if state.last():
                break

    return collected_states


def remove_batch_dim(params):

    def _squeeze_first_dim(x):
        if x.ndim > 1:
            return x.squeeze(axis=0)
        return x
    return jax.tree_map(_squeeze_first_dim, params)