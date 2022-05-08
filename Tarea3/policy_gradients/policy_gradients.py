import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions import Categorical, Normal
import numpy as np


class Policy(nn.Module):

    def __init__(self, dim_states, dim_actions, continuous_control):
        super(Policy, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear ouput activation
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
        """
        Calls either _select_action_discrete or _select_action_continuous depending on the type of environment
        :param observation: the observation given by the environment
        :return: the selected action
        """
        obs = torch.tensor(observation)
        return self._select_action(obs)

    def _select_action_discrete(self, observation):
        """
        Samples an action from a Categorical distribution
        :param observation: the observation given by the environment
        :return: the selected action
        """
        distribution = Categorical(logits=self._policy(observation))
        return distribution.sample().item()

    def _select_action_continuous(self, observation):
        """
        Samples an action from a Normal distribution, using the observation as the mean and _log_std as the std
        :param observation: the observation given by the environment
        :return: the selected action
        """
        distribution = Normal(loc=self._policy(observation), scale=torch.exp(self._policy._log_std))
        return distribution.sample().item()

    def update(self, observation_batch, action_batch, advantage_batch):
        """
        Computes the loss for every rollout and updates the network's parameters

        For each of the N rollouts we have 3 arrays with T time steps each:
            * observations_k -> (T, 3)
            * actions_k      -> (T, 1)
            * rewards_k      -> (T, 1)

        Passing each observation s through the neural network policy we get a probability distribution P(s)
        Then we can compute the probability of getting action a in state s with the distribution P(s) -> P(a|s)
        This gives us one log_prob for each (s_t, a_t)

        The update is made with ∇_θ J_RL = -mean(log_prob * rewards)

        :param observation_batch: the observations sampled
        :param action_batch: the actions sampled
        :param advantage_batch: the rewards sampled
        :return:
        """
        # update the policy here
        # you should use self._compute_loss
        self._optimizer.zero_grad()
        losses = self._compute_loss(observation_batch, action_batch, advantage_batch)
        loss = -torch.mean(losses)
        loss.backward()
        self._optimizer.step()
        return

    def _compute_loss_discrete(self, observations, actions, advantages):
        """
        Compute the loss tensor for each rollout by log_prob * rewards

        The log_probs are sampled from a Categorical distribution, which when sampled returns an array of shape TxT:

            [log_prob(a1|s1), ... , log_prob(a1|sT)]
            [     .         , ... ,       .        ]
            [     .         , ... ,       .        ]
            [     .         , ... ,       .        ]
            [log_prob(aT|s1), ... , log_prob(aT|sT)]

        We are only interested in the probabilities in the diagonal of the matrix:

            [log_prob(a1|s1), ... , log_prob(aT|sT)]

        Once the log_probs are comptued, the loss of the rollout is given by the dot product between log_probs and rew

        :param observation_batch: the observations sampled
        :param action_batch: the actions sampled
        :param advantage_batch: the rewards sampled
        :return: a tensor containing the N losses
        """
        # use negative logprobs * advantages
        rollout_losses = []
        for roll_id in range(len(observations)):
            # make tensors
            obs = torch.tensor(observations[roll_id], requires_grad=True).float()
            act = torch.tensor(actions[roll_id], requires_grad=True).float()
            rew = torch.tensor(advantages[roll_id], requires_grad=True).float()

            # get log prob and expected rewards
            dist = Categorical(logits=self._policy(obs))
            log_prob_k = torch.diag(dist.log_prob(act))
            rew = torch.flatten(rew)

            # compute and save rollout loss
            loss = torch.dot(log_prob_k, rew)
            rollout_losses.append(loss)
        rollout_losses = torch.stack(rollout_losses)
        return rollout_losses

    def _compute_loss_continuous(self, observations, actions, advantages):
        """
        Compute the loss tensor for each rollout by log_prob * rewards

        The log_probs are sampled from a Normal distribution, which when sampled returns an array of shape Tx1:

            [log_prob(a1|s1), ... , log_prob(aT|sT)]

        Once the log_probs are comptued, the loss of the rollout is given by the dot product between log_probs and rew

        :param observation_batch: the observations sampled
        :param action_batch: the actions sampled
        :param advantage_batch: the rewards sampled
        :return: a tensor containing the N losses
        """
        # use negative logprobs * advantages
        # compute log probabilities
        rollout_losses = []
        for roll_id in range(len(observations)):
            # make tensors
            obs = torch.tensor(observations[roll_id], requires_grad=True).float()
            act = torch.tensor(actions[roll_id], requires_grad=True).float()
            rew = torch.tensor(advantages[roll_id], requires_grad=True).float()

            # get log prob and expected rewards
            dist = Normal(loc=self._policy(obs), scale=torch.exp(self._policy._log_std))
            log_prob_k = torch.flatten(dist.log_prob(act))
            rew = torch.flatten(rew)

            # compute and save rollout loss
            loss = torch.dot(log_prob_k, rew)
            rollout_losses.append(loss)
        rollout_losses = torch.stack(rollout_losses)
        return rollout_losses

    def estimate_returns(self, rollouts_rew):
        """
        Compute estimated returns for each rollout k.
        Each rollout has T time steps, so T expected rewards are computed according to:

        Without reward to go, in each time step all rewards are considered so:
                expected_R_t = Σ_(t=1)^(T) [γ^(t - 1) * r_t]
        But with reward to go, past rewards don't influence current rewards so:
                expected_R_t = Σ_(t'=t)^(T) [γ^(t' - t) * r_t']

        :param rollouts_rew: array of shape (N, T) containing the rewards of each rollout
        :return: a list of N arrays where each row contains the T expected rewards for each time step
        """
        estimated_returns = []
        for rollout_rew in rollouts_rew:
            if self._use_reward_to_go:
                # only for part 2
                estimated_return = []
                for t in range(len(rollout_rew)):
                    expected_return = 0
                    for t_ in range(t, len(rollout_rew)):
                        expected_return += (self._gamma ** (t_ - t)) * rollout_rew[t_]
                    estimated_return.append(expected_return)
                estimated_return = np.array(estimated_return)
            else:
                discounted_r = [(self._gamma ** t) * r_t for t, r_t in enumerate(rollout_rew)]
                estimated_return = np.repeat(np.sum(discounted_r), len(rollout_rew))
            estimated_returns.append(estimated_return)

        if self._use_baseline:
            # in discrete environment each rollout has a different amount of time steps so we must use weighted average
            weighted_R = [len(estimated_returns[i])*np.mean(estimated_returns[i]) for i in range(len(estimated_returns))]
            lengths_R = [len(estimated_returns[i]) for i in range(len(estimated_returns))]
            average_return_baseline = np.sum(weighted_R)/np.sum(lengths_R)

            estimated_returns = [return_k - average_return_baseline for return_k in estimated_returns]
        return estimated_returns
