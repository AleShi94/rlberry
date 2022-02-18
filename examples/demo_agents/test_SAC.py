"""
=============================
Record reward during training
=============================

This script shows how to modify an agent to easily record reward or action
during the fit of the agent.
"""


import numpy as np

from rlberry.wrappers import WriterWrapper
# from rlberry.envs import GridWorld
from rlberry.manager import plot_writer_data, AgentManager
from rlberry.envs.benchmarks.ball_exploration import PBall2D
# from rlberry.agents.torch.a2c.a2c import A2CAgent
from rlberry.agents.torch import SACAgent
from rlberry.envs import gym_make


env_id = "CartPole-v0"
env = (gym_make, dict(id=env_id))

agent = AgentManager(SACAgent, env, fit_budget=100, n_fit=3, enable_tensorboard=True, output_dir="rlberry_data/sac_example")
agent.fit()










# We use the following preprocessing function to plot the cumulative reward.
# def compute_reward(rewards):
#     return np.cumsum(rewards)


# # Plot of the cumulative reward.
# output = plot_writer_data(agent, tag="loss_q1", title="Loss q1")

# output = plot_writer_data(agent, tag="loss_q2", title="Loss q2")

# output = plot_writer_data(agent, tag="loss_v", title="Loss critic")

# output = plot_writer_data(agent, tag="loss_act", title="Loss actor")
