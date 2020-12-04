import gym
import numpy as np
import matplotlib.pyplot as plt
from REINFORCE import Reinforce

SEED = 0
MAX_EPISODES = 10
MAX_STEPS = 1000

env = gym.make("LunarLander-v2")
env.seed(SEED)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_hidden = 64

agent = Reinforce(seed=SEED, n_states=n_states, n_actions=n_actions, n_hidden=n_hidden)


def main():
    agent.load_model(fname="../lunarLander.pkl")
    for i_episode in range(MAX_EPISODES):
        state = env.reset()
        score = 0
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            env.render()
            agent.add_step_reward(reward)
            score += reward
            if done:
                print(
                    "Reward: {} | Episode: {}/{}".format(
                        int(score), i_episode, MAX_EPISODES
                    )
                )
                break


if __name__ == "__main__":
    main()

