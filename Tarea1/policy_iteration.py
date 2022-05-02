import numpy as np
import matplotlib.pyplot as plt
import random
from grid_world import GridWorld
from iterator import Iterator
from utils import display_policy, display_value_function


class PolicyIterator(Iterator):

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):
        Iterator.__init__(self, reward_grid, wall_value, cell_value, terminal_value)

    def _state_value(self, state, gamma, action=np.nan):
        """
        Computes the Bellman expectation equation for a state in the grid given its coordinates and the discount factor
        Must be a valid state, not the walls nor the terminal state

        :param state: the state
        :param gamma: discount factor
        :param action: optional default action to take (useful in greedy_action)
        :return: expected value of the state
        """
        new_value = 0
        y, x = self._get_coord(state)
        policy = int(self._policy[y, x])
        # is there a custom policy?
        if not np.isnan(action):
            policy = action
        # for each reachable state from current state
        for successor in self._transition_matrix[state].keys():
            new_value += self.compute_state_value(state, successor, policy, gamma)
        return new_value

    def _policy_evaluation(self, max_iter, gamma, v_thresh):
        """
        Computes the iterative policy evaluation

        Convergence delta is calculated using:
                delta <- max(delta, |v - V(s)|),
        from sutton and barto

        :param max_iter: limit of iterations
        :param gamma: discount factor
        :param v_thresh: threshold of convergence (stops the iterations)
        :return:
        """
        value_rows, value_cols = self._value_function.shape
        for _ in range(max_iter):
            delta = 0
            new_value = np.copy(self._value_function)
            for j in range(1, value_rows - 1):
                for i in range(1, value_cols - 1):
                    # skip terminal state and inner walls
                    if (self._reward_grid[j, i] == self._terminal_value) or (np.isnan(self._reward_grid[j, i])):
                        continue

                    new_value[j, i] = self._state_value(self._get_state(j, i), gamma)
            # check convergence
            diff = self._value_function - new_value
            delta = max(delta, np.linalg.norm(diff[~np.isnan(diff)], np.inf))

            self._value_function = new_value
            if delta < v_thresh:
                break
        return

    def greedy_action(self, state, gamma):
        """
        Chooses the best action by acting greedily with respect to the value function
        :param state: the current state
        :param gamma: the discount factor
        :return: the best action
        """
        values = []
        for action in [0, 1, 2, 3]:
            values.append(self._state_value(state, gamma, action))
        # random tie breaker
        candidates = [action for action, value in enumerate(values) if value == max(values)]
        selected_action = candidates[0]
        if len(candidates) > 1:
            selection = random.randint(0, len(candidates) - 1)
            selected_action = candidates[selection]
        return selected_action

    def _policy_improvement(self, gamma):
        """
        Policy iteration algorithm, cycles between policy evaluation and policy improvement
        The policy improvement is made acting greedily with respect to the expected value of an action

        :param gamma: discount factor
        :return: bool indicating if the policy converged
        """
        value_rows, value_cols = self._value_function.shape
        old_policy = self._policy.copy()
        stable_policy = True
        # for each state
        for j in range(1, value_rows - 1):
            for i in range(1, value_cols - 1):
                # skip terminal states and inner walls
                if (self._reward_grid[j, i] == 0) or (np.isnan(self._reward_grid[j, i])):
                    continue

                self._policy[j, i] = self.greedy_action(self._get_state(j, i), gamma)
                if old_policy[j, i] != self._policy[j, i]:
                    stable_policy = False
        return stable_policy

    def run_policy_iteration(self, p_dir, max_iter, gamma, v_thresh):
        stable_policy = False

        current_iter = 0
        self.construct_transition_matrix(p_dir)
        while (not stable_policy) and (current_iter <= max_iter):
            print(f"Iteration {current_iter}")
            self._policy_evaluation(max_iter, gamma, v_thresh)
            stable_policy = self._policy_improvement(gamma)
            current_iter += 1
        return


if __name__ == '__main__':
    # world = GridWorld(height=14, width=16)
    world = GridWorld(height=14, width=16)
    policy_iterator = PolicyIterator(reward_grid=world._rewards,
                                     wall_value=None,
                                     cell_value=-1,
                                     terminal_value=0)

    # Default parameters for P1-3 (change them for P2-3)
    policy_iterator.run_policy_iteration(p_dir=0.8,
                                         max_iter=1000,
                                         gamma=0.9,
                                         v_thresh=0.0001)
    # world.display()

    display_value_function(policy_iterator._value_function)

    display_policy(world._grid,
                   policy_iterator._reward_grid,
                   policy_iterator._policy)

    plt.show()
