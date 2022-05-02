import numpy as np


class QLearningAgent:

    def __init__(self, states_high_bound, states_low_bound, nb_actions, nb_episodes, gamma, alpha, epsilon):

        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha

        self._states_high_bound = states_high_bound
        self._states_low_bound = states_low_bound
        self._nb_actions = nb_actions

        # Define these variables (P2-2)
        self._nb_states = 285
        self._tabular_q = np.zeros((19, 15, 3))

        self._pos_range = np.around(np.arange(self._states_low_bound[0], self._states_high_bound[0], 0.1), 1)
        self._vel_range = np.around(np.arange(self._states_low_bound[1], self._states_high_bound[1], 0.01), 2)

    def get_state_indices(self, obs):
        """
        Gets the state indices from an observation

        :param obs: the observation given by the environment
        :return: the position and velocity indices
        """
        pos, vel = float(np.format_float_positional(obs[0], 1)), float(np.format_float_positional(obs[1], 2))

        pos_idx = self._pos_range.tolist().index(pos)
        vel_idx = self._vel_range.tolist().index(vel)
        return pos_idx, vel_idx

    def select_action(self, observation, greedy=False):
        """
        Epsilon-greedy policy

        With chance Ɛ, return a random action
        With chance 1 - Ɛ or if greedy is True, return the action a that maximizes the Q_values in self._tabular_q

        :return: the chosen action
        """
        # P1-3
        # get state
        pos_idx, vel_idx = self.get_state_indices(observation)

        # by default choose next action randomly
        action = np.random.randint(0, 3)

        # if greedy maximize tabular_q
        if (np.random.random() > self._epsilon) or greedy:
            action = np.argmax(self._tabular_q[pos_idx, vel_idx])
        return int(np.round(action))

    def update(self, ob_t, ob_t1, action, reward, is_done):
        """
        Q-function update

        Updates the Q_values according to equation (4) of the report

        :return:
        """
        # P1-3
        terminal_condition = ob_t1[0] > 0.5

        pos_t0, vel_t0 = self.get_state_indices(ob_t)
        pos_t1, vel_t1 = self.get_state_indices(ob_t1)

        if is_done and terminal_condition:
            return

        else:
            next_Q = self._tabular_q[pos_t1, vel_t1][np.argmax(self._tabular_q[pos_t1, vel_t1])]
            self._tabular_q[pos_t0, vel_t0][action] = self._tabular_q[pos_t0, vel_t0][action] + \
                                                      self._alpha * (reward + self._gamma * next_Q -
                                                                     self._tabular_q[pos_t0, vel_t0][action])

        # P1-5 only
        if is_done and self._epsilon > 0.0:
            self._epsilon -= 0.0001
