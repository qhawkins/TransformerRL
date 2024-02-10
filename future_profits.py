import numpy as np
import numba as nb
from numba import prange
from find_fill_price import find_fill_price

@nb.njit(cache=True, fastmath=True, parallel=True)
def future_profits(prices_v, cash, buffer_len, position, current_tick):
		# Simplified future profits calculation
		fut_profit = np.zeros(buffer_len)
		initial_basis = find_fill_price(prices_v, position, -position, current_tick)+cash
		#initial_basis = (position * self.prices_v[current_tick]) + self.cash
		for i in prange(buffer_len):
			fut_profit[i] = (find_fill_price(prices_v, position, -position, current_tick + i)+cash)/initial_basis
			#5000 * (((position * self.prices_v[current_tick + i]) + self.cash) - initial_basis) / initial_basis
		return fut_profit
