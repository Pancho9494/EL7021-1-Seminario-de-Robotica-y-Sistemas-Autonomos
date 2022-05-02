import numpy as np
import matplotlib.pyplot as plt


class GridWorld:

    def __init__(self, height, width):

        # World parameters
        self._height = height
        self._width = width
        self._grid = np.ones((self._height, self._width))
        self._rewards = - np.ones((self._height, self._width))

        self._construct_world(self._grid, 0, 1, 0.5)
        self._construct_world(self._rewards, np.nan, -1, 0)

    # Private methods
    def _construct_world(self, grid, wall_value, cell_value, terminal_value):

        # External Walls
        grid[0, 0:] = wall_value
        grid[self._height - 1, 0:] = wall_value
        grid[0:, self._width - 1] = wall_value
        grid[0:, 0] = wall_value

        # Internal Walls
        grid[5, 0:] = wall_value
        grid[5, 5] = cell_value
        grid[8:, 4] = wall_value

        # Terminal state
        grid[self._height - 2, self._width - 2] = terminal_value

    def _construct_world2(self, grid, wall_value, cell_value, terminal_value):

        # External Walls
        grid[0, 0:] = wall_value
        grid[self._height - 1, 0:] = wall_value
        grid[0:, self._width - 1] = wall_value
        grid[0:, 0] = wall_value

        # Terminal states
        grid[1, 1] = terminal_value
        grid[self._height - 2, self._width - 2] = terminal_value

    # Public methods
    def display(self):
        fig, ax = plt.subplots()

        for j in range(self._height):
            for i in range(self._width):
                if self._rewards[j, i] == 0:
                    ax.text(i, j, 'G', ha='center', va='center')
                elif self._rewards[j, i] is not None:
                    ax.text(i, j, self._rewards[j, i], ha='center', va='center')

        ax.imshow(self._grid, cmap='gray')

        plt.title('Grid World')
        plt.axis('off')
        fig.tight_layout()

        plt.savefig('world.pdf')
