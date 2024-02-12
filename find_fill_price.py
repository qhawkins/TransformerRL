import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def find_fill_price(prices_v, quantity, action, timestep=None):
    '''quantity is positive for buy and negative for sell'''
    current_price_slice = prices_v[timestep]

    liquidity_used = 0
    if action > 0:  # Buying
        index = 50
        while abs(quantity) > 0:
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]
            #print(f'price: {price}, liquidity: {liquidity}, index: {index}, quantity: {quantity}')
            if liquidity >= abs(quantity):
                liquidity_used -= price * abs(quantity)
                return liquidity_used
            
            else:
                liquidity_used -= price * liquidity
                
                if quantity < 0:
                    quantity += liquidity
                else:
                    quantity -= liquidity
                
                index += 1

    elif action < 0:  # Selling
        index = 49
        while abs(quantity) > 0:
            price = current_price_slice[index, 0]
            liquidity = current_price_slice[index, 1]
            
            if liquidity >= abs(quantity):
                liquidity_used += price * abs(quantity)

                return liquidity_used
            else:
                liquidity_used += price * liquidity
                
                if quantity < 0:
                    quantity += liquidity
                else:
                    quantity -= liquidity

                index -= 1

    return liquidity_used
