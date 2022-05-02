import numpy as np
import random


class ReplayBuffer:

    def __init__(self, dim_states, dim_actions, max_size, sample_size):
        assert sample_size < max_size, "Sample size cannot be greater than buffer size"

        self._buffer_idx = 0
        self._exps_stored = 0
        self._buffer_size = max_size
        self._sample_size = sample_size

        self._s_t_array = np.zeros((self._buffer_size, dim_states))
        self._a_t_array = np.zeros((self._buffer_size, 1))
        self._r_t_array = np.zeros((self._buffer_size, 1))
        self._s_t1_array = np.zeros((self._buffer_size, dim_states))
        self._term_t_array = np.zeros((self._buffer_size, 1))

    def store_transition(self, s_t, a_t, r_t, s_t1, done_t):
        """
        Add transitions to replay buffer according to self._buffer_idx

        Every time self._buffer_idx reaches its maximum value possible it gets reset back to 0

        :param s_t: state of the episode stored
        :param a_t: action of the episode stored
        :param r_t: reward of the episode stored
        :param s_t1: next state of the episode stored
        :param done_t: True if s_t1 is terminal state, False if else
        :return:
        """
        # Add transition to replay buffer according to self._buffer_idx
        self._s_t_array[self._buffer_idx] = s_t
        self._a_t_array[self._buffer_idx] = a_t
        self._r_t_array[self._buffer_idx] = r_t
        self._s_t1_array[self._buffer_idx] = s_t1
        self._term_t_array[self._buffer_idx] = done_t

        # Update replay buffer index
        self._buffer_idx = (self._buffer_idx + 1) % self._buffer_size
        self._exps_stored += 1

    def sample_transitions(self):
        """
        Get random transitions from buffer

        This is donde by selecting random indices between 0 and _buffer_idx if the buffer isn't full and between 0
        and _buffer_size if the buffer is full.

        Then return the transitions from the buffer according to the chosen indices

        :return: a tuple containing #self._sample_size transitions
        """
        assert self._exps_stored + 1 > self._sample_size, "Not enough samples have been stored to start sampling"

        if self._exps_stored < self._sample_size:
            sample_ids = random.sample(range(0, self._buffer_idx), self._sample_size)
        else:
            sample_ids = random.sample(range(0, self._buffer_size), self._sample_size)

        return (self._s_t_array[sample_ids],
                self._a_t_array[sample_ids].astype(int),
                self._r_t_array[sample_ids],
                self._s_t1_array[sample_ids],
                self._term_t_array[sample_ids])
