from environment import Environment
import numpy as np    
import numba as nb
import glob
import os
from numba import prange
import time
import torch
import transformer_engine.pytorch as te
from actor_model import ActorModel
from transformer_engine.common.recipe import Format, DelayedScaling
import random
import multiprocessing as mp
from multiprocessing import shared_memory
import asyncio

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
		result = np.zeros_like(diff_x)
	
	return result

@nb.njit(cache=True, fastmath=True)
def parse_file(file):
	start_offset = 256
	consolidated_order_book = np.zeros((24000, start_offset, 100, 2), dtype=np.float32)
	order_book_i = 0
	arr_0 = np.zeros((start_offset, 100, 1), dtype=np.float32)
	arr_1 = np.zeros((start_offset, 100, 1), dtype=np.float32)
	try:
		#file = np.reshape(file, (file.shape[2], 100, 2)).astype(np.float32)
		file = np.swapaxes(file, 0, 1)
		file = np.swapaxes(file, 0, 2)
		raw_ob = file
		for i in prange(start_offset, file.shape[0]):
			#if i % 1 == 0:
			arr_0 = jit_z_score(file[i-start_offset:i, :, 0])
			arr_1 = jit_z_score(file[i-start_offset:i, :, 1])
			arr_2 = np.stack((arr_0, arr_1), -1)
			consolidated_order_book[i, :, :, :] = arr_2
			order_book_i += 1
			

		consolidated_order_book = consolidated_order_book[:order_book_i-1, :, :, :]
	except:
		print('error with file')
		return (None, None)
	print(f'file has {np.sum(np.isnan(file))} nans')
	print(f'file has {np.sum(np.isinf(file))} infs')
	print(f'parsed order book shape: {len(consolidated_order_book.shape)}')
	print(f'raw order book shape: {len(raw_ob)}')
	return (consolidated_order_book, raw_ob)

def mask_tokens(data, mask_probability):
	# Create a mask of the same shape as the data
	mask = torch.rand(data.shape) < mask_probability
	# Masking the data
	return mask

def load_model():
	net = ActorModel(transformer_size=1024, transformer_attention_size=64, dropout=.1)
	net_state_dict = net.state_dict()
	raw_state_dict = torch.load('/media/qhawkins/Archive/MLM Models/mlm_model_cluster_2_ddp/model_1_0.004021_normal_1024_64_40.pth')
	
	# Remove the module prefix from the keys
	parsed_state_dict = {key.replace('module.', ''): value for key, value in raw_state_dict.items()}
	print(parsed_state_dict.keys())
	print(net_state_dict.keys())
	[print(f'key inside: {key}') if key in parsed_state_dict else print(key) for key, value in net_state_dict.items()]
	
	exit()
	# replace all items in net state dict that have correesponding keys in parsed state dict with their matching values
	
	net_state_dict = {key: parsed_state_dict[key] if key in parsed_state_dict else value for key, value in net_state_dict.items()}
	
	net.load_state_dict(net_state_dict)
	return net


def get_action(ob_state, env_state, model, device, recipe):
	mask = mask_tokens(ob_state, 0)
	mask = mask.to(device, non_blocking=True)
	env_state = torch.tensor(env_state)
	env_state = env_state.to(device, non_blocking=True)
	ob_state = torch.tensor(ob_state)
	ob_state = ob_state.to(device, non_blocking=True)
	with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
		action = model(mask, ob_state, env_state)

	return action
	# Initialize transformer


def random_index_selection(index_start, num_steps, vector_size, time_dim):
	random_index = random.randint(index_start, vector_size - num_steps - time_dim)
	# Uncomment the next line to print the random index
	# print(f"random index: {random_index}")
	return random_index


def act(env_state, env_num, epsilon, shared_ob_state, model_1, recipe):
	action_probs, state_val = get_action(shared_ob_state, env_state, model=model_1, device='cuda:1', recipe=recipe)
	
	random_value = random.random()
	
	if random_value > epsilon:
		action = torch.argmax(action_probs, dim=1)
	else:
		action = torch.multinomial(action_probs, 1, replacement=True)
	
	action_logprobs = torch.zeros(env_num, device='cuda')
	
	for i in range(action_probs.size(0)):
		action_logprobs[i] = (action_probs[i][action[i].item()] + 1e-8).log()
	
	return action, action_logprobs, state_val

def environment_step(environment, timestep, shared_ob_state, model, recipe):
	mse_loss = torch.nn.MSELoss()
	batched_env_state = np.zeros((len(environment), 256, 3), dtype=np.float32) 
	batched_returns = np.zeros(len(environment), dtype=np.float32)
	for idx, env in enumerate(environment):
		batched_env_state[idx] = env.get_state(timestep)
	batched_actions, action_logprobs, state_val = act(batched_env_state, len(environment), .1, shared_ob_state, model, recipe)
	
	for idx, env in enumerate(environment):
		env.step(batched_actions[idx], timestep)
		batched_returns[idx] = env.get_step_reward()
	
	advantages: torch.Tensor = batched_returns - state_val
	actor_loss = torch.mean(-action_logprobs * advantages.detach())
	actor_loss.backward()

	critic_loss = mse_loss(state_val, torch.tensor(batched_returns, device='cuda'))
	critic_loss.backward()

	return actor_loss, critic_loss

	

async def main():
	fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
	recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
	
	raw_data_path = "/mnt/drive2/raw_data/"
	for folder in glob.glob(raw_data_path + "*/"):
		for idx, filename in enumerate(glob.glob(folder + "*")):
			start_time = time.time()
			file = np.load(filename)
			print(f"File load time: {time.time() - start_time}")
			start_time = time.time()
			file = file[:, :2, :]
			print(f"File trim time: {time.time() - start_time}")
			start_time = time.time()
			file = file.astype(np.float32)
			print(f"File dtype time: {time.time() - start_time}")
			start_time = time.time()
			parsed_file, raw_ob = parse_file(file)
			print(f"File parse time: {time.time() - start_time}")
			if file is None:
				continue
			
			num_threads = 4
			env_per_thread = 32
			pool = mp.Pool(num_threads)
			thread_env = [Environment(prices=raw_ob, offset_init = 256, gamma_init=.09, time=256) for i in range(env_per_thread)]
			thread_env = [thread_env[i].reset() for i in range(env_per_thread)]
			environment_arr = [thread_env for i in range(num_threads)]
			
			model_0 = load_model().to('cuda:0')
			model_1 = load_model().to('cuda:1')

			#create thread pool


			for timestep in range(parsed_file.shape[0]):
				if timestep > 256:
					ob_state = parsed_file[timestep, :, :, :]
					
					for thread_idx in range(num_threads):
						actor_loss, critic_loss = await pool.map_async(environment_step, (environment_arr[thread_idx], timestep, ob_state, model_0, model_1, recipe))
						print(actor_loss.item())
						print(critic_loss.item())
						


				else:
					for thread_idx in range(num_threads):
						for env in environment_arr[thread_idx]:
							env.step(0, timestep)
			
			
			
			print(100*'=')
			"""
			if idx >10:
				exit()
			"""
			

	# Initialize environment
	
	return 0

if __name__ == '__main__':
	asyncio.run(main())