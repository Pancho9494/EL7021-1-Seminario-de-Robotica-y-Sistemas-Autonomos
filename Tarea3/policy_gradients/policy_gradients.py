import torch
import torch.nn as nn
import torch.nn.functional as func
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
            self._log_std = None

    def forward(self, input):
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
        self._optimizer = None

        self._select_action = self._select_action_continuous if self._continuous_control else self._select_action_discrete
        self._compute_loss = self._compute_loss_continuous if self._continuous_control else self._compute_loss_discrete

    def select_action(self, observation):
        return self._select_action(observation)

    def _select_action_discrete(self, observation):
        # sample from categorical distribution
        pass

    def _select_action_continuous(self, observation):
        # sample from normal distribution
        # use the log std trainable parameter
        pass

    def update(self, observation_batch, action_batch, advantage_batch):
        # update the policy here
        # you should use self._compute_loss 
        pass

    def _compute_loss_discrete(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        pass

    def _compute_loss_continuous(self, observation_batch, action_batch, advantage_batch):
        # use negative logprobs * advantages
        pass

    def estimate_returns(self, rollouts_rew):
        estimated_returns = []
        for rollout_rew in rollouts_rew:

            if self._use_reward_to_go:
                # only for part 2
                estimated_return = None
            else:
                estimated_return = None

            estimated_returns = np.concatenate([estimated_returns, estimated_return])

        if self._use_baseline:
            # only for part 2
            average_return_baseline = None
            # Use the baseline:
            # estimated_returns -= average_return_baseline

        return np.array(estimated_returns, dtype=np.float32)

    # It may be useful to discount the rewards using an auxiliary function [optional]
    def _discount_rewards(self, rewards):
        pass
