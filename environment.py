import numpy as np
import numba as nb

@nb.njit(cache=True)
def jit_z_score(x):
	mean_x = np.nanmean(x)
	std_x = np.nanstd(x)
	diff_x = np.subtract(x, mean_x)
	if (np.isnan(std_x) == False and np.isinf(std_x) == False) or (np.isnan(mean_x) == False and np.isinf(mean_x) == False):
		if std_x != 0:
			result = np.divide(diff_x, std_x)
		else:
			result = np.zeros_like(diff_x)
	else:
		print('nans or infs in z_score')
		result = np.empty_like(diff_x)
	
	return result


class Environment:
	def __init__(self, prices=None, offset_init=0, gamma_init=0.0, time=256, margin_requirement_percentage=.25, leverage_factor=4):
		if prices is None:
			print("Error: prices cannot be None")
		self.margin_requirement_percentage = margin_requirement_percentage
		self.leverage_factor = leverage_factor
		self.prices_v = prices
		self.position_history = np.zeros(len(self.prices_v))
		self.cash_history = np.zeros(len(self.prices_v))
		self.total_profit_history = np.zeros(len(self.prices_v))
		self.st_profit_history = np.zeros(len(self.prices_v))
		self.action_history = np.zeros(len(self.prices_v))
		self.w_short = 0.5
		self.w_long = 0.5
		self.offset = offset_init
		self.gamma = gamma_init
		self.time_dim = time
		self.timestep_offset = offset_init
		self.state = np.zeros((self.time_dim, 3))
		

	def reset(self, prices, cash, position, account_value):
		self.prices_v = prices
		# Initialize or reset other attributes as needed
		self.prediction = 0
		self.vec_mean = 0
		self.vec_std = 0
		self.done = False
		self.last_trade_tick = self.time_dim
		self.current_tick = 0
		self.cash = cash
		self.start_cash = cash
		self.position = position
		self.account_value = account_value
		self.profit_list = []
		self.step_reward = 0
		self.total_reward = 0
		self.total_profit = 0
		self.action_taken = 0
		self.past_profit = 0
		self.st_profit = 0
		self.trade = False
		# More attributes initialization as per original C++ code
		self.state = np.zeros((self.time_dim, 3))  # Example, adjust dimensions as needed
		self.price = 0
		self.timestep_offset = 0
		# Initialize histories
		self.sharpe_history = 0
		self.position_penalty_history = 0
		self.weighted_profit_history = 0
		self.total_profit_reward = 0
		self.omega_ratio_history = 0
		self.previous_action_reward = 0

	# Include other methods from the C++ class as Python methods here
	def get_step_reward(self):
		return self.step_reward

	def step(self, action, timestep):
		self.current_tick = timestep + self.timestep_offset
		current_position = self.position
		'''needs to take into account the "true" price from the order book'''
		self.past_profit = self.total_profit
		self.total_profit = (self.find_fill_price(current_position, -current_position, self.current_tick) + self.cash) / self.start_cash
		self.st_profit = self.total_profit - self.past_profit
		self.st_profit_history[self.current_tick] = self.st_profit
		
		

		if action > 0:  # Buying
			potential_trade_cost = self.find_fill_price(action, action)
			potential_cash_after_trade = self.cash - potential_trade_cost
			# Calculate total account value considering leverage and potential trade
			potential_account_value = (self.cash + (self.leverage_factor * potential_trade_cost))
			# Check if account value after trade meets margin requirements
			if potential_trade_cost < self.cash and potential_cash_after_trade >= (potential_account_value * self.margin_requirement_percentage):
				#print('executing trade')
				self.execute_trade(action)
				#print(self.position)
	

		elif action < 0:  # Selling or Negative Action
			adjustment_cost = self.find_fill_price(current_position, action) + self.find_fill_price(action, action)
			# Here, you may also want to consider how selling affects margin usage and account equity
			# For simplicity, ensure net effect of the action doesn't exceed a certain loss threshold, considering margin
			if adjustment_cost > -100000 and (self.cash + adjustment_cost) >= (self.cash * self.margin_requirement_percentage):
				#print('executing trade')
				self.execute_trade(action)
				#print(self.position)
				
		self.position_history[self.current_tick] = self.position
		if self.position > 0:
			self.account_value = self.cash + self.find_fill_price(current_position, current_position)
		elif self.position < 0:
			self.account_value = self.cash - self.find_fill_price(current_position, current_position)
		else:
			self.account_value = self.cash
		
		self.total_profit = self.account_value / self.start_cash
		self.cash_history[self.current_tick] = self.cash
		self.total_profit_history[self.current_tick] = self.total_profit
		self.action_history[self.current_tick] = action
		
		# Compute reward
		self.step_reward = self.calculate_reward()
		self.previous_action = action
		#print(f'current_tick: {self.current_tick}, action: {action}, position: {self.position}, cash: {self.cash}, account_value: {self.account_value}, total_profit: {self.total_profit}, step_reward: {self.step_reward}')


	def execute_trade(self, action):
		# Simplified trading logic
		if action > 0:  # Buy
			self.cash -= (self.find_fill_price(action, action, self.current_tick) + (.0035 * abs(action)))
			self.position += action
		elif action < 0:  # Sell
			self.cash += (self.find_fill_price(action, action, self.current_tick) - (.0035 * abs(action)))
			self.position += action

		
	def calculate_reward(self):
		step_reward = 0.0
		current_position = self.position
		# Example reward calculation
		profit_vec = self.future_profits(self.offset, current_position, self.current_tick)
		step_reward += self.weighted_future_rewards(profit_vec, self.gamma)
		
		if abs(self.position) > 50:
			step_reward -= abs(float(self.position)) / 1000
		
		self.total_reward += step_reward
		return step_reward
	
	'''function to find the fill price of the order'''
	def find_fill_price(self, quantity, action, timestep=None):
		'''quantity is positive for buy and negative for sell'''
		if timestep is None:
			timestep = self.current_tick

		current_price_slice = self.prices_v[timestep]
		liquidity_used = 0

		if action > 0:  # Buying
			index = 50
			while abs(quantity) > 0:
				price = current_price_slice[index, 0]
				liquidity = current_price_slice[index, 1]
				#print(f'price: {price}, liquidity: {liquidity}, index: {index}, quantity: {quantity}')
				if liquidity >= abs(quantity):
					if quantity < 0:
						liquidity_used -= price * abs(quantity)
					else:
						liquidity_used += price * abs(quantity)

					break  # Exit the loop as the entire quantity has been filledq
				else:
					liquidity_used += price * liquidity
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
				#print(f'price: {price}, liquidity: {liquidity}, index: {index}, quantity: {quantity}')
				
				if liquidity >= abs(quantity):
					if quantity < 0:
						liquidity_used += price * abs(quantity)
					else:
						liquidity_used -= price * abs(quantity)

					break  # Exit the loop as the entire quantity has been filled
				else:
					liquidity_used += price * liquidity
					if quantity < 0:
						quantity += liquidity  # Correctly increase towards zero for negative quantities
					else:
						quantity -= liquidity
					index -= 1

		return liquidity_used

	def future_profits(self, buffer_len, position, current_tick):
		# Simplified future profits calculation
		fut_profit = np.zeros(buffer_len)
		initial_basis = self.find_fill_price(position, -position, current_tick)+self.cash
		#initial_basis = (position * self.prices_v[current_tick]) + self.cash
		for i in range(buffer_len):
			fut_profit[i] = (self.find_fill_price(position, -position, current_tick + i)+self.cash)/initial_basis
			#5000 * (((position * self.prices_v[current_tick + i]) + self.cash) - initial_basis) / initial_basis
		return fut_profit

	def weighted_future_rewards(self, unweighted_vector, gamma):
		# Apply discount factor
		discounted_rewards = unweighted_vector * (gamma ** np.arange(len(unweighted_vector)))
		return np.sum(discounted_rewards)

	def get_state(self, tick):
		# Assuming the state includes the last `time_dim` timesteps of prices, predictions, and other features
		start_index = max(0, tick - self.time_dim)
		end_index = tick
		'''need to replace with the "true" price from the order book'''
		self.state[:, 0] = jit_z_score(self.position_history[start_index:end_index])
		self.state[:, 1] = jit_z_score(self.cash_history[start_index:end_index])
		self.state[:, 2] = jit_z_score(self.action_history[start_index:end_index])  # Example scaling
		# Combine all features into a single state array
		return self.state
	
	def get_total_profit(self) -> float:
		return self.total_profit
	
	def get_step_reward(self) -> float:
		return self.step_reward
	
	def get_position(self) -> int:
		return self.position
