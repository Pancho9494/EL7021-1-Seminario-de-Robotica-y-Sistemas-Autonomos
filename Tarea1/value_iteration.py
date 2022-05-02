import numpy as np
import matplotlib.pyplot as plt
import random

from grid_world import GridWorld
from iterator import Iterator
from utils import display_policy, display_value_function


class ValueIterator(Iterator):

    def __init__(self, reward_grid, wall_value, cell_value, terminal_value):
        Iterator.__init__(self, reward_grid, wall_value, cell_value, terminal_value)

    def greedy_state_value(self, state, gamma):
        """
        Updates the state value by choosing the best possible action
        :param state: current state
        :param gamma: discount factor
        :return: the best possible action
        """
        values = []
        for action in [0, 1, 2, 3]:
            action_value = 0
            for successor in self._transition_matrix[state].keys():
                action_value += self.compute_state_value(state, successor, action, gamma)
            values.append(action_value)
        # tie breaker for actions and values
        candidate_actions = [action for action, value in enumerate(values) if value == max(values)]
        selected_action = candidate_actions[0]

        candidate_values = [value for value in values if value == max(values)]
        selected_value = candidate_values[0]

        if len(candidate_values) > 1:
            selection = random.randint(0, len(candidate_values) - 1)
            selected_value = candidate_values[selection]
            selected_action = candidate_actions[selection]

        # update policy
        self._policy[self._get_coord(state)] = selected_action
        return selected_value

    def run_value_iteration(self, p_dir, max_iter, gamma, v_thresh):
        value_rows, value_cols = self._value_function.shape
        self.construct_transition_matrix(p_dir)
        for _ in range(max_iter):
            print(f"Iteration {_}")
            new_value = np.copy(self._value_function)
            for j in range(1, value_rows - 1):
                for i in range(1, value_cols - 1):
                    # skip terminal state and inner walls
                    if (self._reward_grid[j, i] == 0) or (np.isnan(self._reward_grid[j, i])):
                        continue

                    new_value[j, i] = self.greedy_state_value(self._get_state(j, i), gamma)
            # check convergence
            diff = self._value_function - new_value
            delta = max(0, np.linalg.norm(diff[~np.isnan(diff)], np.inf))

            self._value_function = new_value
            if delta < v_thresh:
                break
        return


if __name__ == '__main__':
    world = GridWorld(height=14, width=16)

    value_iterator = ValueIterator(reward_grid=world._rewards,
                                   wall_value=None,
                                   cell_value=-1,
                                   terminal_value=0)

    # Default parameters for P2-2 (change them for P2-3 & P2-4 & P2-5)
    value_iterator.run_value_iteration(p_dir=0.8,
                                       max_iter=1000,
                                       gamma=0.9,
                                       v_thresh=0.0001)

    world.display()

    display_value_function(value_iterator._value_function)

    display_policy(world._grid,
                   value_iterator._reward_grid,
                   value_iterator._policy)

    plt.show()
