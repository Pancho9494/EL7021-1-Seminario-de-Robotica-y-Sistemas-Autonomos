import gym
import time
import datetime
import csv

import numpy as np

import matplotlib.pyplot as plt

from policy_gradients import PolicyGradients


def perform_single_rollout(env, agent, episode_nb, render=False):

    # Modify this function to return a tuple of numpy arrays containing (observations, actions, rewards).
    # (np.array(obs), np.array(acs), np.array(rws))
    # np.array(obs) -> shape: (time_steps, nb_obs)
    # np.array(acs) -> shape: (time_steps, nb_acs) if actions are continuous, (time_steps,) if actions are discrete
    # np.array(rws) -> shape: (time_steps,)

    ob_t = env.reset()
    
    done = False
    episode_reward = 0
    nb_steps = 0

    while not done:

        if render:
            env.render()
            time.sleep(1. / 60)

        action = agent.select_action(ob_t)

        ob_t1, reward, done, _ = env.step(action)

        ob_t = np.squeeze(ob_t1) # <-- may not be needed depending on gym version
        episode_reward += reward
        
        nb_steps += 1

        if done:
            return None


def sample_rollouts(env, agent, training_iter, min_batch_steps):

    sampled_rollouts = []
    total_nb_steps = 0
    episode_nb = 0
    
    while total_nb_steps < min_batch_steps:

        episode_nb += 1
        render = training_iter%10 == 0 and len(sampled_rollouts) == 0 # Change training_iter%10 to any number you want

        # Use perform_single_rollout to get data 
        # Uncomment once perform_single_rollout works.
        # Return sampled_rollouts
        """
        sample_rollout = perform_single_rollout(env, agent, episode_nb, render=render)
        total_nb_steps += len(sample_rollout[0])

        sampled_rollouts.append(sample_rollout)
        """
    return None


def train_pg_agent(env, agent, training_iterations, min_batch_steps):

    tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec = [], [], [], [], []
    _, (axes) = plt.subplots(1, 2, figsize=(12,4))

    
    for tr_iter in range(training_iterations):
        # Sample rollouts using sample_rollouts
        sampled_rollouts = None

        # Parse sampled observations, actions and reward into three arrays:
        # performed_batch_steps >= min_batch_steps
        # sampled_obs: Numpy array, shape: (performed_batch_steps, dim_observations)
        sampled_obs = None
        # sampled_acs: Numpy array, shape: (performed_batch_steps, dim_actions) if actions are continuous, (performed_batch_steps,) if actions are discrete
        sampled_acs = None
        # sampled_rew: standard array of length equal to the number of trayectories that were sampled.
        # You may change the shape of sampled_rew, but it is useful keeping it as is to estimate returns.
        sampled_rew = None

        # Return estimation
        # estimated_returns: Numpy array, shape: (performed_batch_steps, )
        estimated_returns = agent.estimate_returns(sampled_rew)

        # performance metrics
        update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec)

        agent.update(sampled_obs, sampled_acs, estimated_returns)
    
    save_metrics(tr_iters_vec,avg_reward_vec, std_reward_vec)
    

def update_performance_metrics(tr_iter, sampled_rollouts, axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec):

    raw_returns     = np.array([np.sum(rollout[2]) for rollout in sampled_rollouts])
    rollout_steps   = np.array([len(rollout[2]) for rollout in sampled_rollouts])

    avg_return = np.average(raw_returns)
    max_episode_return = np.max(raw_returns)
    min_episode_return = np.min(raw_returns)
    std_return = np.std(raw_returns)
    avg_steps = np.average(rollout_steps)
    std_steps = np.std(rollout_steps)

    # logs 
    print('-' * 32)
    print('%20s : %5d'   % ('Training iter'     ,(tr_iter + 1)          ))
    print('-' * 32)
    print('%20s : %5.3g' % ('Max episode return', max_episode_return    ))
    print('%20s : %5.3g' % ('Min episode return', min_episode_return    ))
    print('%20s : %5.3g' % ('Return avg'        , avg_return            ))
    print('%20s : %5.3g' % ('Return std'        , std_return            ))
    print('%20s : %5.3g' % ('Steps avg'         , avg_steps             ))
    print('%20s : %5.3g' % ('Steps std'         , std_steps             ))

    avg_reward_vec.append(avg_return)
    std_reward_vec.append(std_return)

    avg_steps_vec.append(avg_steps)
    std_steps_vec.append(std_steps)

    tr_iters_vec.append(tr_iter)

    plot_performance_metrics(axes, 
                            tr_iters_vec, 
                            avg_reward_vec, 
                            std_reward_vec, 
                            avg_steps_vec,
                            std_steps_vec)


def plot_performance_metrics(axes, tr_iters_vec, avg_reward_vec, std_reward_vec, avg_steps_vec, std_steps_vec):
    ax1, ax2 = axes
    
    [ax.cla() for ax in axes]
    ax1.errorbar(tr_iters_vec, avg_reward_vec, yerr=std_reward_vec, marker='.',color='C0')
    ax1.set_ylabel('Avg Reward')
    ax2.errorbar(tr_iters_vec, avg_steps_vec, yerr=std_steps_vec, marker='.',color='C1')
    ax2.set_ylabel('Avg Steps')

    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('training iteration') for ax in axes]
    plt.pause(0.05)


def save_metrics(tr_iters_vec, avg_reward_vec, std_reward_vec):
    with open('metrics'+datetime.datetime.now().strftime('%H:%M:%S')+'.csv', 'w') as csv_file:
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
                                             reward_to_go=False,
                                             use_baseline=False)

    train_pg_agent(env=env, 
                   agent=policy_gradients_agent, 
                   training_iterations=1000,
                   min_batch_steps=5000)
