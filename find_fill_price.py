import numba as nb

@nb.njit(cache=True, fastmath=True)
def find_fill_price(prices_v, action, timestep=None) -> float:
    '''quantity is positive for buy and negative for sell'''
    current_price_slice = prices_v[timestep]
    liquidity_used = 0
    if action > 0:
        index = 50
        while action > 0:
            if index > 99:
                return liquidity_used, action
            
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]
            #print(f'price: {price}, liquidity: {liquidity}, index: {index}')
            if liquidity >= action:
                liquidity_used -= price * action
                action = 0
                return liquidity_used, action
            else:
                liquidity_used -= price * liquidity
                action -= liquidity
                index += 1
        
    elif action < 0:
        index = 49
        while action < 0:
            if index < 0:
                return liquidity_used, action
            
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]

            if liquidity >= abs(action):
                liquidity_used += price * abs(action)
                action = 0
                return liquidity_used, action
            else:
                liquidity_used += price * liquidity
                action += liquidity
                index -= 1
    else:
        return liquidity_used, action