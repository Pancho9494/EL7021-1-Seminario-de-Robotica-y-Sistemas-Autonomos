import time
import datetime
import gym
import csv
import os
import torch

import numpy as np

import matplotlib.pyplot as plt

from deep_qnetwork import DeepQNetworkAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # no corre sin esto


def train_agent(env, eval_env, agent, nb_training_steps, nb_steps_target_replace, render=False):
    tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec = [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12, 4))

    ob_t = env.reset()
    done = False
    episode_nb = 0
    episode_reward = 0
    episode_steps = 0

    update_performance_metrics(agent, eval_env, 0, axes, tr_steps_vec,
                               avg_reward_vec, std_reward_vec, success_rate_vec)
    max_avg = 0
    max_idx = 0
    for tr_step in range(nb_training_steps):

        if (tr_step + 1) % (nb_training_steps / 20) == 0:
            update_performance_metrics(agent, eval_env, tr_step + 1, axes, tr_steps_vec,
                                       avg_reward_vec, std_reward_vec, success_rate_vec)
            if avg_reward_vec[-1] > max_avg:
                PATH = f"models/model_step_{tr_step}_reward_{avg_reward_vec[-1]}.pkl"
                torch.save(agent._deep_QNetwork.state_dict(), PATH)
                max_avg = avg_reward_vec[-1]
                max_idx = tr_step

        if (tr_step + 1) % nb_steps_target_replace == 0:
            agent.replace_target_network()

        action = agent.select_action(ob_t)

        ob_t1, reward, done, _ = env.step(action.item())

        agent.store_transition(ob_t, action, reward, ob_t1, done)

        if tr_step > agent.replay_buffer._sample_size:
            agent.update()

        ob_t = ob_t1

        if render:
            env.render()
            time.sleep(1 / 60.)

        episode_reward += reward
        episode_steps += 1

        if done:
            print(
                'Global training step %5d | Training episode %5d | Steps: %4d | Reward: %4d | Success: %5r | Epsilon: %.3f' % \
                (tr_step + 1, episode_nb + 1, episode_steps, episode_reward, episode_steps == 500, agent._epsilon))

            episode_nb += 1
            ob_t = env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0

    # load and test best model
    agent._deep_QNetwork.load_state_dict(torch.load(f"models/model_step_{max_idx}_reward_{max_avg}.pkl"))
    update_performance_metrics(agent, eval_env, tr_step + 1, axes, tr_steps_vec,
                               avg_reward_vec, std_reward_vec, success_rate_vec)

    save_metrics(agent, nb_steps_target_replace, tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec)
    fig, (axis) = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axis

    # save plots
    [ax.cla() for ax in axis]
    ax1.errorbar(tr_steps_vec, avg_reward_vec, yerr=std_reward_vec, marker='.', color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.plot(tr_steps_vec, success_rate_vec, marker='.', color='C1')
    ax2.set_ylabel('Success Rate')

    [ax.grid('on') for ax in axis]
    [ax.set_xlabel('episode') for ax in axis]
    plt.pause(0.05)
    plt.savefig('performance_metrics.png')
    plt.show()
    plt.close()


def update_performance_metrics(agent, eval_env, training_step, axes, tr_steps_vec, avg_reward_vec, std_reward_vec,
                               success_rate_vec):
    avg_reward, std_reward, success_rate = test_agent(eval_env, agent)

    tr_steps_vec.append(training_step)
    avg_reward_vec.append(avg_reward)
    std_reward_vec.append(std_reward)
    success_rate_vec.append(success_rate)

    plot_performance_metrics(axes,
                             tr_steps_vec,
                             avg_reward_vec,
                             std_reward_vec,
                             success_rate_vec)


def save_metrics(agent, nb_steps_target_replace, tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec):
    with open('metrics/metrics' + datetime.datetime.now().strftime('%H-%M-%S') + '.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['steps', 'avg_reward', 'std_reward', 'success_rate'])
        for i in range(len(tr_steps_vec)):
            csv_writer.writerow([tr_steps_vec[i], avg_reward_vec[i], std_reward_vec[i], success_rate_vec[i]])


def test_agent(env, agent, nb_episodes=30, render=True):
    ep_rewards = []
    success_rate = 0
    avg_steps = 0

    for episode in range(nb_episodes):

        ob_t = env.reset()
        done = False
        episode_reward = 0
        nb_steps = 0

        while not done:

            if render and episode == 0:
                env.render()
                time.sleep(1. / 60)

            action = agent.select_action(ob_t, greedy=True)

            ob_t1, reward, done, _ = env.step(action.item())

            ob_t = ob_t1
            episode_reward += reward

            nb_steps += 1

            if done:
                if nb_steps == 500:
                    success_rate += 1.
                avg_steps += nb_steps
                ep_rewards.append(episode_reward)
                print('Evaluation episode %3d | Steps: %4d | Reward: %4d | Success: %r' % (
                    episode + 1, nb_steps, episode_reward, nb_steps == 500))

    ep_rewards = np.array(ep_rewards)
    avg_reward = np.average(ep_rewards)
    std_reward = np.std(ep_rewards)
    success_rate /= nb_episodes
    avg_steps /= nb_episodes
    print('Average Reward: %.2f, Reward Deviation: %.2f | Average Steps: %.2f, Success Rate: %.2f' % (
        avg_reward, std_reward, avg_steps, success_rate))

    return avg_reward, std_reward, success_rate


def plot_performance_metrics(axes, tr_steps_vec, avg_reward_vec, std_reward_vec, success_rate_vec):
    ax1, ax2 = axes

    [ax.cla() for ax in axes]
    ax1.errorbar(tr_steps_vec, avg_reward_vec, yerr=std_reward_vec, marker='.', color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.plot(tr_steps_vec, success_rate_vec, marker='.', color='C1')
    ax2.set_ylabel('Success Rate')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('Training step') for ax in axes]
    plt.pause(0.05)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')

    # Actions are discrete
    # +1 or -1
    dim_actions = np.array(env.action_space.n)

    # States are continuous
    dim_states = env.observation_space.shape[0]

    print(dim_states)
    print(dim_actions)

    nb_training_steps = 20000

    deep_qlearning_agent = DeepQNetworkAgent(dim_states=dim_states,
                                             dim_actions=dim_actions,
                                             lr=0.01,
                                             gamma=0.99,
                                             epsilon=0.8,
                                             nb_training_steps=nb_training_steps,
                                             replay_buffer_size=5000,
                                             batch_size=128)

    train_agent(env=env,
                eval_env=eval_env,
                agent=deep_qlearning_agent,
                nb_training_steps=nb_training_steps,
                nb_steps_target_replace=100)
