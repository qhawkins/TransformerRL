import numpy as np
import numba as nb
from numba import prange
from find_fill_price import find_fill_price


@nb.njit(cache=True, fastmath=True, parallel=True)
def future_profits(prices_v, buffer_len, position, current_tick, action_taken):
    # Simplified future profits calculation
    fut_profit = np.zeros(buffer_len)
    #initial_basis = find_fill_price(prices_v, position, -position, current_tick)+cash
    #initial_basis = (position * self.prices_v[current_tick]) + self.cash
    for i in prange(buffer_len):
        pos_closing_cost, _ = find_fill_price(prices_v, -position, current_tick)
        fut_profit[i], _ = find_fill_price(prices_v, -(position+action_taken), current_tick + i)-pos_closing_cost
        
        #fut_profit[i] = ((find_fill_price(prices_v, position, -position, current_tick + i)+cash)-initial_basis)/initial_basis
        #5000 * (((position * self.prices_v[current_tick + i]) + self.cash) - initial_basis) / initial_basis
    return fut_profit
