import gym
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import matplotlib.pyplot as plt

from noise import OUNoise
from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer


def predict(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep):

    saver = tf.train.Saver(max_to_keep=5)
    saver.restore(sess, tf.train.latest_checkpoint("model_checkpoints/"))

    max_episodes = ep
    max_steps = 3000

    for i in range(max_episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            next_state, reward, done, info = env.step(action[0])

            state = next_state
            score += reward
            env.render()

            if done:
                print("Reward: {} | Episode: {}/{}".format(int(score), i, max_episodes))
                break


if __name__ == "__main__":

    with tf.Session() as sess:

        env = gym.make("LunarLanderContinuous-v2")

        env.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        ep = 10
        tau = 0.001
        gamma = 0.99
        min_batch = 64
        actor_lr = 0.00005
        critic_lr = 0.0005
        buffer_size = 1000000

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        actor_noise = OUNoise(mu=np.zeros(action_dim))
        actor = ActorNetwork(
            sess, state_dim, action_dim, action_bound, actor_lr, tau, min_batch
        )
        critic = CriticNetwork(
            sess,
            state_dim,
            action_dim,
            critic_lr,
            tau,
            gamma,
            actor.get_num_trainable_vars(),
        )
        predict(sess, env, actor, critic, actor_noise, buffer_size, min_batch, ep)

