import gym
import time
import datetime
import csv

import numpy as np

import matplotlib.pyplot as plt

from policy_gradients import PolicyGradients


def perform_single_rollout(env, agent, render=False):
    """
    Perform a rollout in an environment and save the observations, actions and rewards as arrays of shapes:
        * observations -> (nb_steps, dim_states)
        * actions -> (nb_steps, 1)
        * rewards -> (nb_steps, 1)
    Then returns the 3 arrays as a tuple

    :param env: The simulated environment
    :param agent: The agent interacting with the environment
    :param render: Boolean flag indicating whether to render the environment or not
    :return: tuple containing the observations, actions and rewards arrays
    """
    ob_t = env.reset()

    done = False
    episode_reward = 0
    nb_steps = 0

    # arrays to be filled
    observations = np.empty(ob_t.shape)
    actions = np.empty(1)
    rewards = np.empty(1)
    while not done:

        if render:
            env.render()
            time.sleep(1. / 60)

        # action must be tensor
        action_t = agent.select_action(ob_t)

        # continuous environment only receives arrays while discrete environment can recieve int
        if continuous_control:
            ob_t1, reward_t, done, _ = env.step(np.array([action_t]))
        else:
            ob_t1, reward_t, done, _ = env.step(action_t)

        # parse observations, actions and rewards into (T, dim)
        observations = np.vstack((observations, ob_t))
        actions = np.vstack((actions, action_t))
        rewards = np.vstack((rewards, reward_t))

        ob_t = np.squeeze(ob_t1)  # <-- may not be needed depending on gym version
        episode_reward += reward_t

        nb_steps += 1

        if done:
            # remove first empty values
            observations = observations[1:]
            actions = actions[1:]
            rewards = rewards[1:]
            return observations, actions, rewards


def sample_rollouts(env, agent, training_iter, min_batch_steps):
    """
    Samples N = min_batch_steps/len(sample_rollout[0]) rollouts
    Each rollout k has T_k time steps, T_k can be different to T_k'
    :param env: The simulated environment
    :param agent: The agent interacting with the environment
    :param training_iter: Iteration M of training
    :param min_batch_steps: How much data to store in each batch
    :return: list of length N containing in each element a tuple of 3 arrays containing T values
    """
    samples = []
    total_nb_steps = 0
    episode_nb = 0

    while total_nb_steps < min_batch_steps:
        episode_nb += 1
        render = training_iter % 200 == 0 and len(
            samples) == 0  # Render every 200 iterations

        # Use perform_single_rollout to get data
        sample_rollout = perform_single_rollout(env, agent, render=False)
        total_nb_steps += len(sample_rollout[0])

        samples.append(sample_rollout)
    return samples


def train_pg_agent(env, agent, training_iterations, min_batch_steps):
    tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec = [], [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12, 4))

    for tr_iter in range(training_iterations):
        # Sample rollouts using sample_rollouts
        samples = sample_rollouts(env, agent, tr_iter, min_batch_steps)

        sampled_obs = [rollout[0] for rollout in samples]
        sampled_acs = [rollout[1] for rollout in samples]
        sampled_rew = [rollout[2] for rollout in samples]

        # Return estimation
        # estimated_returns: Numpy array, shape: (performed_batch_steps, )
        estimated_returns = agent.estimate_returns(sampled_rew)

        # performance metrics
        update_performance_metrics(tr_iter, samples, axes, tr_iters_vec, avg_reward_vec, std_reward_vec,
                                   avg_steps_vec, std_steps_vec)

        agent.update(sampled_obs, sampled_acs, estimated_returns)

    save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec)
    fig, (axis) = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axis

    # save plots
    [ax.cla() for ax in axis]
    ax1.errorbar(tr_iters_vec, avg_reward_vec, yerr=std_reward_vec, marker='.', color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.plot(tr_iters_vec, avg_steps_vec, marker='.', color='C1')
    ax2.set_ylabel('Avg Steps')

    [ax.grid('on') for ax in axis]
    [ax.set_xlabel('training iteration') for ax in axis]
    plt.pause(0.05)
    plt.savefig('performance_metrics.png')
    plt.show()
    plt.close()


def update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec,
                               avg_steps_vec, std_steps_vec):
    raw_returns = np.array([np.sum(rollout[2]) for rollout in sampled_rollouts])
    rollout_steps = np.array([len(rollout[2]) for rollout in sampled_rollouts])

    avg_return = np.average(raw_returns)
    max_episode_return = np.max(raw_returns)
    min_episode_return = np.min(raw_returns)
    std_return = np.std(raw_returns)
    avg_steps = np.average(rollout_steps)
    std_steps = np.std(rollout_steps)

    # logs 
    print('-' * 32)
    print('%20s : %5d' % ('Training iter', (tr_iter + 1)))
    print('-' * 32)
    print('%20s : %5.3g' % ('Max episode return', max_episode_return))
    print('%20s : %5.3g' % ('Min episode return', min_episode_return))
    print('%20s : %5.3g' % ('Return avg', avg_return))
    print('%20s : %5.3g' % ('Return std', std_return))
    print('%20s : %5.3g' % ('Steps avg', avg_steps))
    print('%20s : %5.3g' % ('Steps std', std_steps))

    avg_reward_vec.append(avg_return)
    std_reward_vec.append(std_return)

    avg_steps_vec.append(avg_steps)
    std_steps_vec.append(std_steps)

    tr_iters_vec.append(tr_iter)

    if tr_iter % 50:
        plot_performance_metrics(axes,
                                 tr_iters_vec,
                                 avg_reward_vec,
                                 std_reward_vec,
                                 avg_steps_vec,
                                 std_steps_vec)


def plot_performance_metrics(axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec):
    ax1, ax2 = axes

    [ax.cla() for ax in axes]
    ax1.errorbar(tr_iters_vec, avg_reward_vec, yerr=std_reward_vec, marker='.', color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.errorbar(tr_iters_vec, avg_steps_vec, yerr=std_steps_vec, marker='.', color='C1')
    ax2.set_ylabel('Avg Steps')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training iteration') for ax in axes]
    plt.pause(0.05)


def save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec):
    with open('metrics/metrics' + datetime.datetime.now().strftime('%H-%M-%S') + '.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['steps', 'avg_reward', 'std_reward'])
        for i in range(len(tr_iters_vec)):
            csv_writer.writerow([tr_iters_vec[i], avg_reward_vec[i], std_reward_vec[i]])


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')

    dim_states = env.observation_space.shape[0]

    continuous_control = isinstance(env.action_space, gym.spaces.Box)
    dim_actions = env.action_space.shape[0] if continuous_control else env.action_space.n

    policy_gradients_agent = PolicyGradients(dim_states=dim_states,
                                             dim_actions=dim_actions,
                                             lr=0.005,
                                             gamma=0.99,
                                             continuous_control=continuous_control,
                                             reward_to_go=True,
                                             use_baseline=False)

    train_pg_agent(env=env,
                   agent=policy_gradients_agent,
                   training_iterations=1000,
                   min_batch_steps=5000)

    # exp 31