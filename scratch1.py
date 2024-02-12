
#@nb.njit(cache=True, fastmath=True)
def find_fill_price(prices_v, action, timestep=None):
    '''quantity is positive for buy and negative for sell'''
    current_price_slice = prices_v[timestep]
    liquidity_used = 0
    
    if action > 0:
        index = 50
        while abs(action) > 0:
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]

            if liquidity >= abs(action):
                liquidity_used -= price * abs(action)
                return liquidity_used
            else:
                liquidity_used -= price * liquidity
                action -= liquidity
                index += 1
        
    elif action < 0:
        index = 49
        while abs(action) > 0:
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]

            if liquidity >= abs(action):
                liquidity_used += price * abs(action)
                return liquidity_used
            else:
                liquidity_used += price * liquidity
                action -= liquidity
                index -= 1

    return liquidity_used
