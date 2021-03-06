import gym
import numpy as np
import matplotlib.pyplot as plt
from REINFORCE import Reinforce

SEED = 0
MAX_EPISODES = 100000
MAX_STEPS = 1000

env = gym.make("LunarLander-v2")
env.seed(SEED)
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
n_hidden = 64

agent = Reinforce(seed=SEED, n_states=n_states, n_actions=n_actions, n_hidden=n_hidden)


def main():
    avg_reward = None
    score_list = []
    plt.ion()
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

        last_reward = agent.get_rewards_sum()
        score_list.append(score)
        avg_reward = (
            last_reward if not avg_reward else avg_reward * 0.9 + last_reward * 0.1
        )
        agent.finish_episode()
        avg = np.mean(score_list[-200:])
        print("Average of last 200 episodes: {0:.2f} \n".format(avg))
        if avg > 200:
            print("Task Completed")
            print("The last episode ran for {} time steps!".format((t + 1)))
            agent.save_model(fname="lunarLander.pkl")
            break

        if i_episode % 10 == 0:
            agent.save_model(fname="lunarLander.pkl")

        plt.cla()
        plt.plot(score_list)
        plt.pause(0.0001)

    plt.ioff()
    plt.show()
    plt.savefig("lunarLander-REINFORCE.png")


if __name__ == "__main__":
    main()

