import random
import numpy as np
from ExperienceReply import *
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(0)


def DPG(
    env,
    policy,
    buffer_size=5000,
    batch_size=100,
    num_episodes=100,
    max_steps=10000,
    gamma=0.99,
    lr_policy=0.3,
    lr_qfunction=0.3,
    lr_qvalue=0.3,
):
    lr_decay = 0.99
    replay = ExperienceReply(buffer_size)
    replay_update_count = 0
    policy_update_count = 0

    rewards = []
    timesteps = []
    score_list = []
    plt.ion()

    for iter1 in range(num_episodes):

        state = env.reset()
        score = 0
        action = policy.get_action_from_policy(state, True)

        t = 0
        isTerminal = False
        nupdate_episode = 0

        rewards_episode = 0
        episode_buffer = []
        while isTerminal == False and t < max_steps:
            state1, reward1, isTerminal, info = env.step(action)
            action1 = policy.get_action_from_policy(state1, True)
            episode_buffer.append(
                [state.T, action, reward1, state1.T, action1, isTerminal]
            )

            state = state1
            action = action1
            t += 1
            env.render()

            rewards_episode += reward1
            score += reward1
            replay_update_count += 1

        print("Reward: {} | Episode: {}/{}".format(int(score), iter1 + 1, num_episodes))

        discount_reward = 0.0
        for idx in range(len(episode_buffer) - 1, -1, -1):
            episode_buffer[idx][2] += gamma * discount_reward
            discount_reward = episode_buffer[idx][2]

        for idx in range(len(episode_buffer)):
            replay.add_item(
                episode_buffer[idx][0],
                episode_buffer[idx][1],
                episode_buffer[idx][2],
                episode_buffer[idx][3],
                episode_buffer[idx][4],
                episode_buffer[idx][5],
            )

            if replay.size() > batch_size and idx % 10 == 0:
                batch = replay.sample_batch(batch_size)
                batch_reward, batch_reward2 = policy.update_policy_batch(
                    batch, gamma, lr_policy, lr_qfunction, lr_qvalue
                )

                policy_update_count += 1
                nupdate_episode += 1

                if policy_update_count % 3000 == 0:
                    if lr_policy > 0.005:
                        lr_policy *= lr_decay
                    if lr_qvalue > 0.05:
                        lr_qvalue *= lr_decay
                    if lr_qfunction > 0.05:
                        lr_qfunction *= lr_decay

        if t > 0:
            rewards_episode /= t
        else:
            rewards_episode = 0
        rewards.append(rewards_episode)
        score_list.append(score)
        timesteps.append(t)
        plt.cla()
        plt.plot(score_list)
        plt.pause(0.0001)

        avg = np.mean(score_list[-200:])
        print("Average of last 200 episodes: {0:.2f} \n".format(avg))

        if avg > 100:
            print("Task Completed")
            print("The last episode ran for {} time steps!".format((j + 1)))

    plt.ioff()
    plt.show()
    plt.savefig("lunarLander-DPG.png")
    return policy, rewards, timesteps

