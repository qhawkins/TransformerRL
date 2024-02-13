from find_fill_price import find_fill_price
import numba as nb

@nb.njit(cache=True, fastmath=True)
def execute_trade(prices_v, action, current_tick):
    cash = 0
    remaining_liquidity = action
    while remaining_liquidity != 0:
        action_cost, remaining_liquidity = find_fill_price(prices_v, action, current_tick)
        cash += action_cost
        action = remaining_liquidity
        current_tick += 1

    cash -= .0035 * action
    return cash