import numpy as np
from utils import add_if_exists


class Iterator:

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):

        self._reward_grid = reward_grid
        self._wall_value = wall_value
        self._cell_value = cell_value
        self._terminal_value = terminal_value

        self._value_function = np.zeros(self._reward_grid.shape)
        self._value_function *= self._reward_grid
        self._policy = self._value_function.copy()
        self._transition_matrix = {}

    def _get_state(self, y, x):
        """
        Converts a pair of coordinates into their corresponding state considering that states are arbitrarily numbered
        from 0 to W * H, starting at the top left corner and ending at the bottom right one
        :param y: number of column in grid
        :param x: number of row in grid
        :return: state as an integer
        """
        w = self._reward_grid.shape[1]
        return y * w + x

    def _get_coord(self, s):
        """
        Inverse of _get_state(y, x), computes coordinates given the state considering the ordered described in get_state
        :param s: state as an integer
        :return: tuple of coordinates (y, x)
        """
        w = self._reward_grid.shape[1]
        return int(np.floor(s) / w), s % w

    def construct_transition_matrix(self, p):
        """
        Computes which states are reachable from each state and their corresponding transition probability given p

        Considering the state numbering described in _get_state(y, x), the probability transition matrix would be of
        shape 224 * 224. Each element of that matrix would contain a 4-element array with the probabilities of reaching
        that state given the input [up, down, right, left].

        This matrix would be to sparse, for example, the row for state 17 would be filled with zero-vectors except
        in states 17, 18 and 33. Also, the probabilities of transition are always the same values in different orders.

        So instead of creating the probability transition matrix, we simply make a dictionary where each element
        contains the reachable successor states for each state, and their corresponding probability of transition,
        for example: {17 = {17: {u: , d: , r: , l}, 18 : {u: , d: , r: , l}, 33 : {u: , d: , r: , l}}, ...}

        These dictionaries of probabilities are always the same depending on the spatial relation between the state
        and their successor, if s' is up from s then we use the U dictionary, if it's down we use the D dictionary and
        so on and so forth. The only special case happens when the agent tries to move into a wall, so it stays in
        the same state as it started, here the probability of transition is just a sum of the dictionaries where the
        current state has walls (top left state has a  U + L probability transition dictionary)

        :param p: probability that the agent actually goes to the selected successor state, it also has a
                  (1-p)/2 probability of transition for each perpendicular state
        :return:
        """
        # grid shape
        H, W = self._reward_grid.shape
        # transition probabilities for each direction
        U = {0: p, 1: 0, 2: round((1 - p) / 2, 2), 3: round((1 - p) / 2, 2)}
        D = {0: 0, 1: p, 2: round((1 - p) / 2, 2), 3: round((1 - p) / 2, 2)}
        R = {0: round((1 - p) / 2, 2), 1: round((1 - p) / 2, 2), 2: p, 3: 0}
        L = {0: round((1 - p) / 2, 2), 1: round((1 - p) / 2, 2), 2: 0, 3: p}

        # for iterating over the directions
        UD = [U, D]
        LR = [L, R]

        # iterate states skipping external walls
        transition_matrix = {}
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                # skip terminal state and inner walls
                if (self._reward_grid[y, x] == 0) or (np.isnan(self._reward_grid[y, x])):
                    continue

                state = self._get_state(y, x)
                reachable = {}
                for dIdx in range(2):
                    delta = [-1, 1]

                    # vertical: U first, then D
                    reward = self._reward_grid[y + delta[dIdx], x]
                    if not np.isnan(reward):
                        reachable[self._get_state(y + delta[dIdx], x)] = UD[dIdx]
                    else:
                        reachable = add_if_exists(reachable, state, UD[dIdx])

                    # horizontal: L first, then R
                    reward = self._reward_grid[y, x + delta[dIdx]]
                    if not np.isnan(reward):
                        reachable[self._get_state(y, x + delta[dIdx])] = LR[dIdx]
                    else:
                        reachable = add_if_exists(reachable, state, LR[dIdx])

                transition_matrix[state] = reachable
        self._transition_matrix = transition_matrix
        return

    def compute_state_value(self, state, successor, policy, gamma):
        """
        Applies the bellman expectation equation for for a given state and its successor
        :param state: original state
        :param successor: successor state to original state
        :param policy: policy of the agent at original state
        :param gamma: discount factor
        :return: a portion of the value of the original state corresponding to its successor
        """
        sy, sx = self._get_coord(successor)
        old_value = self._value_function[sy, sx]
        trans_prob = self._transition_matrix[state][successor][policy]
        reward = self._reward_grid[sy, sx]

        # V_k+1(s) = sum[R   + gamma *   V_k(s')] * P(s'|s,a)
        return (reward + gamma * old_value) * trans_prob
