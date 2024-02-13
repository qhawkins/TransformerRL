import numpy as np
import numba as nb
from numba import prange
from execute_trade import execute_trade

@nb.njit(cache=True, fastmath=True, parallel=True)
def future_profits(prices_v, buffer_len, position, current_tick, action_taken, pre_cash, post_cash):
    # Simplified future profits calculation
    fut_profit = np.zeros(buffer_len)
    #initial_basis = find_fill_price(prices_v, position, -position, current_tick)+cash
    #initial_basis = (position * self.prices_v[current_tick]) + self.cash
    for i in prange(buffer_len):
        with_trade = execute_trade(prices_v, -(position+action_taken), current_tick + i)+post_cash
        wo_trade = execute_trade(prices_v, -position, current_tick + i)+pre_cash
        #print(f'with_trade: {with_trade}, wo_trade: {wo_trade}')
        if wo_trade != 0:
            fut_profit[i] = (with_trade-wo_trade)/wo_trade
        else:
            fut_profit[i] = 0
        #fut_profit[i] = ((find_fill_price(prices_v, position, -position, current_tick + i)+cash)-initial_basis)/initial_basis
        #5000 * (((position * self.prices_v[current_tick + i]) + self.cash) - initial_basis) / initial_basis
    return fut_profit
