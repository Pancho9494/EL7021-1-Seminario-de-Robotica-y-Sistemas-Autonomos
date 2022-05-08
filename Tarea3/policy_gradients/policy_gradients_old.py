import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Categorical, Normal

import numpy as np


class Policy(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Policy, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear output activation
        # dim_states -> 64 -> 64 -> dim_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)
        self.float()

        if continuous_control:
            # trainable parameter
            self._log_std = nn.parameter.Parameter(torch.tensor([1.0]))

    def forward(self, input):
        # input = torch.from_numpy(input).float()
        out = input.to(self.device)
        out = func.relu(self.fc1(out))
        out = func.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class PolicyGradients:

    def __init__(self, dim_states, dim_actions, lr, gamma,
                 continuous_control=False, reward_to_go=False, use_baseline=False):

        self._learning_rate = lr
        self._gamma = gamma

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self._continuous_control = continuous_control
        self._use_reward_to_go = reward_to_go
        self._use_baseline = use_baseline

        self._policy = Policy(self._dim_states, self._dim_actions, self._continuous_control)
        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self._learning_rate)

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete

    def select_action(self, observation):
        return self._select_action(observation)

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        categorical_distribution = Categorical(logits=self._policy(observation))
        return categorical_distribution.sample().item()

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        normal_distribution = Normal(loc=self._policy(observation), scale=self._policy._log_std)
        return normal_distribution.sample().item()

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here
        # you should use self._compute_loss
        self._optimizer.zero_grad()
        loss = self._compute_loss(observation_batch, action_batch, advantage_batch)
        loss.backward()
        self._optimizer.step()
        pass

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        log_probs = []
        for k, G in enumerate(advantage_batch):
            log_prob = 0
            for t in rollout:
                delta = int(len(observation_batch) / len(advantage_batch))
                ran = t * delta, t * delta + delta
                obs = observation_batch[ran[0]:ran[1]]
                act = action_batch[ran[0]:ran[1]]

                dist = Categorical(logits=self._policy(obs))
                log_prob += dist.log_prob(act)
            log_probs.append(log_prob)
        log_probs = np.array(log_probs)
        advntg = np.array(advantage_batch)
        loss = np.multiply(log_probs, advntg)





        dist = Categorical(logits=self._policy(observation_batch))
        log_prob = dist.log_prob(action_batch)
        # print(log_prob.size())
        loss = -torch.mean(torch.mul(log_prob, advantage_batch))

        return loss

    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        log_probs = []
        for k, G in enumerate(advantage_batch):
            log_prob = 0
            delta = int(len(observation_batch) / len(advantage_batch))
            t_1, t_T = k * delta, k * delta + delta
            obs = observation_batch[t_1:t_T]
            act = action_batch[t_1:t_T]

            dist = Normal(loc=self._policy(obs), scale=self._policy._log_std)
            p = dist.log_prob(act)
            log_probs.append(np.sum(p))
        log_probs = np.array(log_probs)
        advntg = np.array(advantage_batch)
        loss = np.multiply(log_probs, advntg)
        return loss

    def estimate_returns(self, rollouts_rew):
        # estimated_returns = torch.zeros(len(rollouts_rew), dtype=torch.float)
        estimated_returns = []
        for idx, rollout_rew in enumerate(rollouts_rew):
            for t in range(len(rollout_rew)):
                Gt = 0
            # compute powers of gamma as:
            # [gamma, gamma**2, gamma**3, ..., gamma**len(rollout_rew)]
            # gammas = np.fromiter([self._gamma ** i for i in range(len(rollout_rew))], dtype=np.float32)
            # gammas = torch.from_numpy(gammas)

            # if self._use_reward_to_go:
            if True:
                # only for part 2
                estimated_return = None
                for t_ in range(t, len(rollout_rew)):
                    Gt += self._gamma**(t_ - t) * rollout_rew[t_]
                estimated_returns.append(Gt)
            # else:
                # estimated return is just a dot product between powers of gamma and returns
                # estimated_return = torch.dot(rollout_rew, gammas)

            # estimated_returns[idx] = estimated_return
        if self._use_baseline:
            # only for part 2
            average_return_baseline = None
            # Use the baseline:
            # estimated_returns -= average_return_baseline

        return estimated_returns

    # It may be useful to discount the rewards using an auxiliary function [optional]
    def _discount_rewards(self, rewards):
        pass
