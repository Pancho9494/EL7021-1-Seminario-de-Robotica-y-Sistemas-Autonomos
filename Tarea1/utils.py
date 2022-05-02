import numpy as np
import matplotlib.pyplot as plt


def add_if_exists(a_dict, key, new_dict):
    """
    Utility function

    If a_dict already has key then sums (element-wise) the values of new_dict to the ones in a_dict[key].
    If not, then just add the elements of new_dict to a_dict under key

    :param a_dict: original dictionary
    :param key: key to be checked
    :param new_dict: elements to be added
    :return:
    """
    out = a_dict.copy()
    new_dict = new_dict.copy()
    if key not in out.keys():
        out[key] = new_dict
    else:
        assert len(out[key]) == len(new_dict)
        for idx in range(len(out[key])):
            out[key][idx] += new_dict[idx]
    return out


def display_value_function(value_function):
    fig, ax = plt.subplots()
    value_rows, value_cols = value_function.shape
    value_function_display = value_function.copy()
    value_function_display = np.nan_to_num(value_function_display)
    value_function_display[np.isnan(value_function)] = np.min(value_function_display)
    threshold = (np.max(value_function_display) - np.min(value_function_display)) / 2

    for j in range(value_rows):
        for i in range(value_cols):
            if not np.isnan(value_function[j, i]):
                ax.text(i, j, format(value_function[j, i], '.1f'), ha='center', va='center',
                        color='white' if abs(value_function[j, i]) > threshold else 'black', fontsize='small')

    ax.imshow(value_function_display, cmap='gray')

    plt.title('Value Function')
    plt.axis('off')
    fig.tight_layout()

    plt.savefig('./figs/value_function.png')


def display_policy(world_grid, reward_grid, policy):
    fig, ax = plt.subplots()
    rows, cols = reward_grid.shape

    arrow_symbols = [u'\u2191', u'\u2193', u'\u2192', u'\u2190']

    for j in range(rows):
        for i in range(cols):
            if reward_grid[j, i] == 0.0:
                ax.text(i, j, 'G', ha='center', va='center')

            elif not np.isnan(policy[j, i]):
                ax.text(i, j, arrow_symbols[int(policy[j, i])], ha='center', va='center')

    ax.imshow(world_grid, cmap='gray')

    plt.title('Policy')
    plt.axis('off')
    fig.tight_layout()

    plt.savefig('./figs/policy.png')
