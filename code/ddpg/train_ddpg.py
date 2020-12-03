import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt

from noise import OUNoise
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer


def save_model(saver, session, fname, steps=None, write_meta_graph=False):
    if steps:
        saver.save(session, fname, global_step=steps, write_meta_graph=write_meta_graph)
    else:
        saver.save(session, fname)


def train(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(buffer_size, 0)

    max_episodes = ep
    max_steps = 1000
    score_list = []
    plt.ion()

    for i in range(max_episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            next_state, reward, done, info = env.step(action[0])
            replay_buffer.add(
                np.reshape(state, (actor.s_dim,)),
                np.reshape(action, (actor.a_dim,)),
                reward,
                done,
                np.reshape(next_state, (actor.s_dim,)),
            )

            # updating the network in batch
            if replay_buffer.size() < min_batch:
                continue

            states, actions, rewards, dones, next_states = replay_buffer.sample_batch(
                min_batch
            )
            target_q = critic.predict_target(
                next_states, actor.predict_target(next_states)
            )

            y = []
            for k in range(min_batch):
                y.append(rewards[k] + critic.gamma * target_q[k] * (1 - dones[k]))

            # Update the critic given the targets
            predicted_q_value, _ = critic.train(
                states, actions, np.reshape(y, (min_batch, 1))
            )

            # Update the actor policy using the sampled gradient
            a_outs = actor.predict(states)
            grads = critic.action_gradients(states, a_outs)
            actor.train(states, grads[0])

            # Update target networks
            actor.update_target_network()
            critic.update_target_network()

            state = next_state
            score += reward
            env.render()

            if done:
                print("Reward: {} | Episode: {}/{}".format(int(score), i, max_episodes))
                break

        score_list.append(score)

        plt.cla()
        plt.plot(score_list)
        plt.pause(0.0001)

        avg = np.mean(score_list[-200:])
        print("Average of last 200 episodes: {0:.2f} \n".format(avg))

        if avg > 200:
            print("Task Completed")
            print("The last episode ran for {} time steps!".format((j + 1)))
            save_model(
                saver,
                sess,
                fname="model_checkpoints/lunarLander",
                steps=i,
                write_meta_graph=True,
            )
            break

        if i % 10 == 0:
            save_model(saver, sess, fname="model_checkpoints/lunarLander", steps=i)

    plt.ioff()
    plt.show()

    plt.savefig("lunarLander-ddpg.png")

    return score_list


if __name__ == "__main__":

    with tf.Session() as sess:

        env = gym.make("LunarLanderContinuous-v2")

        env.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        ep = 2000
        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = 0.00005
        critic_lr = 0.0005
        buffer_size = 1000000

        n_states = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(
            sess, n_states, action_dim, action_bound, actor_lr, tau, min_batch
        )
        critic = CriticNetwork(
            sess,
            n_states,
            action_dim,
            critic_lr,
            tau,
            gamma,
            actor.get_num_trainable_vars(),
        )
        scores = train(
            sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep
        )

