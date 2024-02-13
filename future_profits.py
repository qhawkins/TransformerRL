import numpy as np
import numba as nb
from numba import prange
from find_fill_price import find_fill_price
from execute_trade import execute_trade

@nb.njit(cache=True, fastmath=True, parallel=True)
def future_profits(prices_v, buffer_len, position, current_tick, action_taken):
    # Simplified future profits calculation
    fut_profit = np.zeros(buffer_len)
    #initial_basis = find_fill_price(prices_v, position, -position, current_tick)+cash
    #initial_basis = (position * self.prices_v[current_tick]) + self.cash
    for i in prange(buffer_len):
        with_trade = execute_trade(prices_v, position+action_taken, current_tick + i)
        wo_trade = execute_trade(prices_v, position, current_tick + i)
        fut_profit[i] = with_trade/wo_trade
        
        #fut_profit[i] = ((find_fill_price(prices_v, position, -position, current_tick + i)+cash)-initial_basis)/initial_basis
        #5000 * (((position * self.prices_v[current_tick + i]) + self.cash) - initial_basis) / initial_basis
    return fut_profit
