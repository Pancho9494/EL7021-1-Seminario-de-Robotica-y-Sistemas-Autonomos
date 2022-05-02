import torch
import torch.nn as nn
import torch.nn.functional as func
import copy

import numpy as np

from replay_buffer import ReplayBuffer


class DeepQNetwork(nn.Module):

    def __init__(self, dim_states, dim_actions):
        super(DeepQNetwork, self).__init__()
        # MLP, fully connected layers, ReLU activations, linear output activation
        # dim_states -> 64 -> 64 -> dim_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(dim_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, dim_actions)
        self.float()

    def forward(self, input):
        out = input.to(self.device)
        out = func.relu(self.fc1(out))
        out = func.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DeepQNetworkAgent:

    def __init__(self, dim_states, dim_actions, lr, gamma, epsilon, nb_training_steps, replay_buffer_size, batch_size):

        self._learning_rate = lr
        self._gamma = gamma
        self._epsilon = epsilon

        self._epsilon_min = 0
        self._epsilon_decay = self._epsilon / (nb_training_steps / 2.)

        self._dim_states = dim_states
        self._dim_actions = dim_actions

        self.replay_buffer = ReplayBuffer(dim_states=self._dim_states,
                                          dim_actions=self._dim_actions,
                                          max_size=replay_buffer_size,
                                          sample_size=batch_size)

        # Complete
        self._deep_QNetwork = DeepQNetwork(self._dim_states, self._dim_actions)
        self._deep_QNetwork = self._deep_QNetwork.to(self._deep_QNetwork.device)

        self._target_deepQ_network = copy.deepcopy(self._deep_QNetwork)
        self._target_deepQ_network = self._target_deepQ_network.to(self._deep_QNetwork.device)

        # Adam optimizer and MSE Loss
        self._optimizer = torch.optim.Adam(self._deep_QNetwork.parameters(), lr)
        self._criterion = nn.MSELoss()

    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        """
        Save a transition, as tensors, in the replay buffer

        :param s_t: initial state of episode
        :param a_t: action chosen in episode (a_t is already a tensor)
        :param r_t: reward given by environment in episode
        :param s_t1: next state of episode
        :param done_t: flat indicating whether s_t1 is a terminal state or not
        :return:
        """
        s_t = torch.tensor(s_t, device=self._deep_QNetwork.device)
        r_t = torch.tensor(r_t, device=self._deep_QNetwork.device)
        s_t1 = torch.tensor(s_t1, device=self._deep_QNetwork.device)
        done_t = torch.tensor(done_t, device=self._deep_QNetwork.device)
        self.replay_buffer.store_transition(s_t, a_t, r_t, s_t1, done_t)

    def replace_target_network(self):
        """
        Load the weights of _deep_QNetwork onto the target network

        :return:
        """
        self._target_deepQ_network.load_state_dict(self._deep_QNetwork.state_dict())

    def select_action(self, observation, greedy=False):
        """
        Select next action given an environment observation

        With chance Ɛ, return a random action
        With chance 1 - Ɛ or if greedy is True, return the action a that maximizes the Q_values given by deep_QNetwork

        :param observation: state of the agent
        :param greedy: flag indicating whether to act greedily or not
        :return: the chosen action
        """
        if (np.random.random() > self._epsilon) or greedy:
            with torch.no_grad():
                action = self._deep_QNetwork(torch.tensor(observation, device=self._deep_QNetwork.device)).argmax()
                action = action.type(torch.int64)
        else:
            # Select random action
            action = torch.tensor(np.random.randint(0, self._dim_actions), device=self._deep_QNetwork.device,
                                  dtype=torch.int64)

        if not greedy and self._epsilon >= self._epsilon_min:
            # Implement epsilon linear decay
            self._epsilon -= self._epsilon_decay

        return action

    def update(self):
        """
        Run one step of training with a mini batch

        :return:
        """
        # get batches
        st_arr, a_arr, r_arr, st1_arr, end_arr = self.replay_buffer.sample_transitions()

        # make tensors from batches
        st_arr = torch.from_numpy(st_arr).float()
        a_arr = torch.from_numpy(a_arr).type(torch.int64)
        r_arr = torch.from_numpy(r_arr).float()
        st1_arr = torch.from_numpy(st1_arr).float()
        end_arr = torch.from_numpy(end_arr).bool()

        # compute desired labels y_j
        y_t = self.compute_y(r_arr, st1_arr, end_arr)

        # compute actual labels Q(s_j, a_j)
        self._optimizer.zero_grad(set_to_none=True)
        Q_values = self.compute_QValues(st_arr, a_arr)

        # compute and propagate loss, update optimizer
        loss = self._criterion(Q_values, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._deep_QNetwork.parameters(), 1)
        self._optimizer.step()
        return

    def compute_QValues(self, st, a):
        """
        Compute the QValues for st given by QNetwork
        Then, choose each value according to the chosen actions
        :param st: batch of states
        :param a: batch of actions
        :return: QValues of states st for actions a
        """
        QValues = self._deep_QNetwork(st)
        return torch.gather(QValues, dim=1, index=a).flatten()

    def compute_y(self, r, st1, end):
        """
        Compute the QValues for st1 given by target network
        We don't train the target network so values are computed with torch.no_grad()

        :param r: The reward of the episode
        :param st1: The state of the episode
        :param end: Flag indicating whether the state is terminal or not
        :return: Tensor with r where end is True and r + gamma * Qv where end is False
        """
        with torch.no_grad():
            QValues = torch.amax(self._target_deepQ_network(st1), dim=1)
            y_j = torch.where(end.flatten(), r.flatten(), torch.add(end.flatten(), QValues, alpha=self._gamma))
        return y_j
