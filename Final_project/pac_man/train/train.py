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
import logging
import os
from typing import Dict, Optional, Tuple, Any

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from joblib.testing import param
from matplotlib.animation import FuncAnimation
import omegaconf
from tqdm.auto import trange

import utils
from Final_project.pac_man.agents.random_agent import RandomAgent
from Final_project.pac_man.agents.a2c_agent import Agent
from loggers import TerminalLogger
from setup_train import (
    setup_agent,
    setup_env,
    setup_evaluators,
    setup_logger,
    setup_training_state,
)
from timer import Timer
from Final_project.pac_man.train.types import TrainingState
from Final_project.pac_man.environment.env import PacMan
from Final_project.pac_man.environment.viewer import PacManViewer


def run_visualization(states: Any, save_path: Optional[str] = None) -> None:
    """Run visualization of the trained agent.

    Args:
        agent: The trained agent.
        env: The PacMan environment.
        save_path: Optional path to save the animation as GIF.
    """
    viewer = PacManViewer(name="PacMan Evaluation", render_mode="human")
    # Save animation if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            frame_img = viewer.render(frame)
            ax.imshow(frame_img)
            ax.set_axis_off()
            return ax,

        anim = FuncAnimation(
            fig, update, frames=states, interval=200, blit=True
        )
        anim.save(save_path, writer="pillow", fps=5)
        plt.close(fig)
        print(f"Animation saved to {save_path}")

    # Interactive visualization
    for state in states:
        viewer.render(state)
        plt.pause(0.1)

    viewer.close()

def inference(agent: Agent, env: PacMan, training_state: TrainingState):
    key = jax.random.PRNGKey(0)
    state, timestep = env.reset(key)
    states = [state]
    lstm_state = None
    lstm_state = jax.tree_map(lambda x: jax.device_put(x, jax.devices()[0]), lstm_state)

    def remove_pmap_dim(params):
        return jax.tree_map(lambda x: x[0] if x.ndim > 1 else x, params)

    policy_params = remove_pmap_dim(training_state.params_state.params.actor)

    # Run episode
    for _ in range(200):
        with jax.default_device(jax.devices()[0]):
            action, lstm_state = agent.act(
                policy_params=policy_params,
                timestep=timestep,
                lstm_state=lstm_state,
                key=key
            )

        state, timestep = env.step(state, action[0])
        states.append(state)

        if timestep.last():
            break

    return states

@hydra.main(config_path="configs", config_name="config.yaml")
def train(cfg: omegaconf.DictConfig, log_compiles: bool = False) -> None:
    """Main training function with integrated visualization."""
    # Setup logging
    logging.info(omegaconf.OmegaConf.to_yaml(cfg))
    logging.getLogger().setLevel(logging.INFO)
    logging.info({"devices": jax.local_devices()})

    # Initialize keys
    key, init_key = jax.random.split(jax.random.PRNGKey(cfg.seed))

    # Setup components
    logger = setup_logger(cfg)
    env = setup_env(cfg)
    agent = setup_agent(cfg, env)
    stochastic_eval, greedy_eval = setup_evaluators(cfg, agent)
    training_state = setup_training_state(env, agent, init_key)

    # Calculate steps per epoch
    num_steps_per_epoch = (
            cfg.env.training.n_steps
            * cfg.env.training.total_batch_size
            * cfg.env.training.num_learner_steps_per_epoch
    )

    # Setup timers
    eval_timer = Timer(out_var_name="metrics")
    train_timer = Timer(out_var_name="metrics", num_steps_per_timing=num_steps_per_epoch)

    @functools.partial(jax.pmap, axis_name="devices")
    def epoch_fn(training_state: TrainingState) -> Tuple[TrainingState, Dict]:
        """Run one training epoch."""
        training_state, metrics = jax.lax.scan(
            lambda training_state, _: agent.run_epoch(training_state),
            training_state,
            None,
            cfg.env.training.num_learner_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, metrics

    # Main training loop
    with jax.log_compiles(log_compiles), logger:
        for i in trange(
                cfg.env.training.num_epochs,
                disable=isinstance(logger, TerminalLogger),
        ):
            env_steps = i * num_steps_per_epoch

            # Evaluation
            key, stochastic_eval_key, greedy_eval_key = jax.random.split(key, 3)

            # Stochastic evaluation
            with eval_timer:
                metrics = stochastic_eval.run_evaluation(
                    training_state.params_state, stochastic_eval_key
                )
                jax.block_until_ready(metrics)
            logger.write(
                data=utils.first_from_device(metrics),
                label="eval_stochastic",
                env_steps=env_steps,
            )

            # Greedy evaluation
            if not isinstance(agent, RandomAgent):
                with eval_timer:
                    metrics = greedy_eval.run_evaluation(
                        training_state.params_state, greedy_eval_key
                    )
                    jax.block_until_ready(metrics)
                logger.write(
                    data=utils.first_from_device(metrics),
                    label="eval_greedy",
                    env_steps=env_steps,
                )

            # Training step
            with train_timer:
                training_state, metrics = epoch_fn(training_state)
                jax.block_until_ready((training_state, metrics))
            logger.write(
                data=utils.first_from_device(metrics),
                label="train",
                env_steps=env_steps,
            )

    # Visualization after training
    if jax.process_index() == 0:
        print("\nTraining completed. Running visualization...")
        raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env

        # Safe path handling
        save_dir = getattr(getattr(cfg, "logger", None), "save_dir", None)
        save_path = os.path.join(save_dir, "training_animation.gif") if save_dir else None


        states = inference(agent, raw_env, training_state)
        run_visualization(states, save_path)


if __name__ == "__main__":
    train()