from find_fill_price import find_fill_price
import numba as nb

@nb.njit(cache=True, fastmath=True)
def execute_trade(prices_v, action, current_tick):
    cash = 0
    while True:
        action_cost, remaining_liquidity = find_fill_price(prices_v, action, current_tick) - (.0035 * abs(action))
        cash += action_cost
        action = remaining_liquidity
        current_tick += 1

        if remaining_liquidity == 0:
            return cash