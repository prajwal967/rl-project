import sys
import random
import gym
from algorithm import *
from policy import *

import matplotlib.pyplot as plt

random.seed(0)

if __name__ == "__main__":
    # gameName = "LunarLanderContinuous-v2"
    gameName = "MountainCarContinuous-v0"
    env = gym.make(gameName)
    policy = Policy(env)

    # mountain car
    policy, rewards, timesteps = DPG(
        env,
        policy,
        buffer_size=8000,
        batch_size=200,
        num_episodes=1000,
        max_steps=3000,
        gamma=0.99,
        lr_policy=0.005,
        lr_qfunction=0.03,
        lr_qvalue=0.03,
    )

    # lunar lander
    # policy, rewards, timesteps = DPG(
    #     env,
    #     policy,
    #     buffer_size=100000,
    #     batch_size=200,
    #     num_episodes=2000,
    #     max_steps=1000,
    #     gamma=0.99,
    #     lr_policy=3e-5,
    #     lr_qfunction=3e-4,
    #     lr_qvalue=3e-3,
    # )

