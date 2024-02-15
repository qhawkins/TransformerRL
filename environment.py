import numpy as np
import numba as nb
from weighted_future_rewards import weighted_future_rewards
from future_profits import future_profits
from execute_trade import execute_trade
import time

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
	def __init__(self, prices=None, offset_init=0, gamma_init=0.95, time=256, margin_requirement_percentage=.25, leverage_factor=4):
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
		self.portfolio_leverage_history = np.zeros(len(self.prices_v))
		self.cash_leverage_history = np.zeros(len(self.prices_v))
		self.offset = offset_init
		self.gamma = gamma_init
		self.time_dim = time
		self.timestep_offset = offset_init
		self.state = np.zeros((self.time_dim, 3))
		self.running_reward = []
		self.buy_hold_position = 0
		self.sell_hold_position = 0
		self.bh_profit = 0
		self.sh_profit = 0
		self.bh_paid = 0
		self.sh_paid = 0
		self.counter = 0
		self.bh_cash = 0
		self.sh_cash = 0
		self.sharpe_ratio = 0
		self.portfolio_leverage = 0

	def reset(self, prices, cash, position, account_value):
		self.running_reward = []
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
		# Initialize histories
		self.sharpe_history = 0
		self.position_penalty_history = 0
		self.weighted_profit_history = 0
		self.total_profit_reward = 0
		self.omega_ratio_history = 0
		self.previous_action_reward = 0
		self.buy_hold_position = 0
		self.sell_hold_position = 0
		self.bh_profit = 0
		self.sh_profit = 0
		self.bh_paid = 0
		self.sh_paid = 0
		self.counter = 0
		self.bh_cash = 0
		self.sh_cash = 0
		self.sharpe_ratio = 0
		self.position_history = np.zeros(len(self.prices_v))
		self.cash_history = np.zeros(len(self.prices_v))
		self.total_profit_history = np.zeros(len(self.prices_v))
		self.st_profit_history = np.zeros(len(self.prices_v))
		self.action_history = np.zeros(len(self.prices_v))
		self.portfolio_leverage_history = np.zeros(len(self.prices_v))
		self.cash_leverage_history = np.zeros(len(self.prices_v))
		self.portfolio_leverage = 0
	
	def get_step_reward(self):
		return self.step_reward

	def step(self, action, timestep):
		#self.current_tick = timestep + self.timestep_offset
		self.current_tick = timestep
		self.past_profit = self.total_profit
		
		self.account_value = execute_trade(self.prices_v, -self.position, timestep) + self.cash
		self.total_profit = self.account_value / self.start_cash
		
		
		self.st_profit = self.total_profit - self.past_profit
		self.st_profit_history[self.current_tick] = self.st_profit
		if timestep == 255:
			self.counter = 0
			while True:
				self.counter+=1

				self.bh_paid = execute_trade(self.prices_v, self.counter, timestep)
				
				self.bh_cash = self.cash+self.bh_paid
				if self.bh_cash < 0:
					self.buy_hold_position = self.counter-1
					self.bh_paid = execute_trade(self.prices_v, self.buy_hold_position, timestep)
					self.bh_cash = self.cash+self.bh_paid
					break


			self.counter = 0
			while True:
				self.counter-=1
				self.sh_paid = execute_trade(self.prices_v, self.counter, timestep)
				self.sh_cash = self.cash+self.sh_paid
				if self.sh_cash > 2*self.start_cash:
					self.sell_hold_position = self.counter+1
					self.sh_paid = execute_trade(self.prices_v, self.sell_hold_position, timestep)
					self.sh_cash = self.cash+self.sh_paid
					break

			self.counter=0
		

		self.pre_cash = self.cash
		self.action_taken = 0
		if action > 0:  # Buying
			# Calculate potential trade cost and remaining liquidity
			'''cash position action'''
			pot_cash = execute_trade(self.prices_v, action, timestep)
			
			if pot_cash+self.cash > 0:
				self.cash+=pot_cash
				self.position+=action
				self.action_taken = action

		elif action < 0:  # Selling or Negative Action
			pot_cash = execute_trade(self.prices_v, action, timestep)
			
			if pot_cash+self.cash < 2* self.start_cash:
				self.cash+=pot_cash
				self.position+=action
				self.action_taken = action
		self.post_cash = self.cash



			#if pot_cash < 100000 and (self.cash + adjustment_cost) >= (self.cash * self.margin_requirement_percentage):
				#print('executing trade')
			#self.execute_trade(action)
				#print(self.position)
		self.position_history[self.current_tick] = self.position
		
		#pot_cash = execute_trade(self.prices_v, -self.position, timestep)


		# Calculate buy and hold and sell and hold profits
		
		self.bh_profit = (self.bh_cash + execute_trade(self.prices_v, -self.buy_hold_position, timestep))/self.start_cash
		self.sh_profit = (self.sh_cash + execute_trade(self.prices_v, -self.sell_hold_position, timestep))/self.start_cash

		# Compute reward
		self.step_reward = self.calculate_reward()
		self.running_reward.append(self.step_reward)
		#self.step_reward = (self.step_reward-np.mean(self.running_reward[timestep-self.offset:timestep]))/(np.std(self.running_reward[timestep-self.offset:timestep])+.000001)
		self.previous_action = action
		#print(f'current_tick: {self.current_tick}, action: {action}, position: {self.position}, cash: {self.cash}, account_value: {self.account_value}, total_profit: {self.total_profit}, step_reward: {self.step_reward}')
		self.position_value = execute_trade(self.prices_v, -self.position, timestep)
		self.account_value = self.position_value+self.cash
		self.portfolio_leverage = self.position_value / self.account_value
		self.cash_leverage = self.cash / self.account_value


		self.cash_history[self.current_tick] = self.cash
		self.total_profit_history[self.current_tick] = self.total_profit
		self.action_history[self.current_tick] = action
		self.portfolio_leverage_history[self.current_tick] = self.portfolio_leverage
		self.cash_leverage_history[self.current_tick] = self.cash_leverage

	def calculate_reward(self):
		step_reward = 0.0
		current_position = self.position
		# Example reward calculation
		profit_vec = future_profits(self.prices_v, self.offset, current_position, self.current_tick, self.action_taken, self.pre_cash, self.post_cash)
		step_reward += weighted_future_rewards(profit_vec, self.gamma)*1000000
		if abs(self.portfolio_leverage) > .2:
			step_reward -= abs(self.portfolio_leverage)
		
		'''sharpe ratio calculation'''
		if np.std(profit_vec) != 0:
			self.sharpe_ratio = np.mean(profit_vec) / np.std(profit_vec)
			step_reward += self.sharpe_ratio
			

		'''
		if abs(self.position) > 50:
			step_reward -= abs(float(self.position)) / 1000
		'''
		
		self.total_reward += step_reward
		return step_reward
	

	def get_state(self, tick):
		# Assuming the state includes the last `time_dim` timesteps of prices, predictions, and other features
		start_index = max(0, tick - self.time_dim)
		end_index = tick
		'''need to replace with the "true" price from the order book'''
		#self.state[:, 0] = jit_z_score(self.position_history[start_index:end_index])
		
		self.state[:, 0] = self.portfolio_leverage_history[start_index:end_index]

		#self.state[:, 1] = jit_z_score(self.cash_history[start_index:end_index])
		self.state[:, 1] = self.cash_leverage_history[start_index:end_index]

		self.state[:, 2] = jit_z_score(self.action_history[start_index:end_index])  # Example scaling
		# Combine all features into a single state array
		return self.state
	
	def get_total_profit(self) -> float:
		return self.total_profit
	
	def get_step_reward(self) -> float:
		return self.step_reward
	
	def get_position(self) -> int:
		return self.position
	
	def get_action_taken(self) -> int:
		return self.action_taken

	def get_bh_profit(self) -> float:
		return self.bh_profit
	
	def get_sh_profit(self) -> float:
		return self.sh_profit

	def get_sharpe_ratio(self) -> float:
		return self.sharpe_ratio

	def get_portfolio_leverage(self) -> float:
		return self.portfolio_leverage