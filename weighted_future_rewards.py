import numba as nb
import numpy as np

@nb.njit(cache=True, fastmath=True)
def weighted_future_rewards(unweighted_vector, gamma):
		# Apply discount factor
		discounted_rewards = unweighted_vector * (gamma ** np.arange(len(unweighted_vector)))
		return np.sum(discounted_rewards)