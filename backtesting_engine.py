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
import torch.distributed as dist
import math

def path_function(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print(f'{path} created successfully')
	else:
		print(f'{path} already exists')


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
def parse_file(file, start_offset):
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
	return (consolidated_order_book, raw_ob[start_offset:, :, :])

def mask_tokens(data, mask_probability):
	# Create a mask of the same shape as the data
	mask = torch.rand(data.shape) > mask_probability
	# Masking the data
	return mask

def load_model(rank, tensor_parallel_group, config):
	net = ActorModel(config['batch_size'], rank, tensor_parallel_group, config['transformer_size'], config['transformer_attention_size'], config['dropout'], config['fuse_qkv'])
	
	net_state_dict = net.state_dict()
	raw_state_dict = torch.load('/media/qhawkins/Archive/MLM RL Models/rl_model_day_5_rank_0.pth')
	
	# Remove the module prefix from the keys
	parsed_state_dict = {key.replace('module.', ''): value for key, value in raw_state_dict.items()}
	# replace all items in net state dict that have correesponding keys in parsed state dict with their matching values
	
	net_state_dict = {key: parsed_state_dict[key] if key in parsed_state_dict else value for key, value in net_state_dict.items()}
	
	net.load_state_dict(net_state_dict)
	return net



def act_calcs(action_probs, state_val):
	actions = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
	
	action = torch.argmax(action_probs)
	action = actions[action]
	
	return action, state_val

def env_state_retr(environment, timestep):
	batched_env_state = np.zeros((len(environment), 256, 3), dtype=np.float32) 
	for idx, env in enumerate(environment):
		batched_env_state[idx] = env.get_state(timestep)
	return batched_env_state

def env_step(environment, timestep, actions):
	batched_actions = actions.cpu().numpy()
	
	batched_rewards = np.zeros(len(environment), dtype=np.float32)
	for idx, env in enumerate(environment): 	
		env.step(batched_actions[idx], timestep)
		batched_rewards[idx] = env.get_step_reward()
		
	return (batched_rewards, environment)
		

def create_torch_group(rank, tensor_parallel_group, data_parallel_group, config):
	
	torch.cuda.set_device(rank)
	
	model = load_model(rank, tensor_parallel_group, config).to('cuda')

	
	fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
	recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
	
	raw_data_path = "/mnt/drive2/raw_data/"
	batched_ob = torch.zeros((config['batch_size'], 256, 100, 2), dtype=torch.float32)

	logging_path = f'/media/qhawkins/Archive/MLM RL Model Backtesting Logs'
	path_function(logging_path)
	filenames = []

	for folder in glob.glob(raw_data_path + "*/"):
		for idx, filename in enumerate(glob.glob(folder + "*")):
			filenames.append(filename)


	random.shuffle(filenames)
	idx = 0
	for filename in filenames:
		idx += 1
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
		parsed_file, raw_ob = parse_file(file, 256)
		print(f"File parse time: {time.time() - start_time}")
		if file is None:
			continue
		
		env = Environment(prices=raw_ob, offset_init = config['end_buffer'], gamma_init=.99, time=256)
		env.reset(prices=raw_ob, cash=config['start_cash'], position=0, account_value=config['start_cash'])
		
		mask = mask_tokens(batched_ob, 0)
		mask = mask.cuda(non_blocking=True)
		
		with open(f'{logging_path}/rl_model_day_{idx}_rank_{rank}.txt', 'w') as f:
			for timestep in range(parsed_file.shape[0]-config['end_buffer']):
				timestep_time_start = time.time()
				if timestep > 256:
					ob_state = torch.tensor(parsed_file[timestep, :, :, :]).cuda()
					
					batched_env_state = torch.tensor(env.get_state(timestep)).cuda()
					batched_env_state
					batched_ob = batched_ob.cuda(non_blocking=True)
					#forward_time_start = time.time()
					with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
						action_probs, state_val = model(mask, ob_state, batched_env_state)
					#forward_time_end = time.time()
					#print(f'forward time: {forward_time_end - forward_time_start}')

					action, state_val = act_calcs(action_probs, state_val)
					env_step(batched_env_state, timestep, action)
					#action_logprobs = torch.reshape(action_logprobs, (config['num_threads'], config['envs_per_thread']))
					#state_val = torch.reshape(state_val, (config['num_threads'], config['envs_per_thread']))
					#pool_time_start = time.time()
				
					accumulated_profit = 0
					accumulated_step_reward = 0
					accumulated_position = 0
					accumulated_bh_profit = 0
					accumulated_sh_profit = 0
					accumulated_action_taken = 0
					accumulated_sharpe_ratio = 0
					accumulated_portfolio_leverage = 0

					accumulated_profit += env.get_total_profit()
					accumulated_step_reward += env.get_step_reward()
					accumulated_position += env.get_position()
					accumulated_bh_profit += env.get_bh_profit()
					accumulated_sh_profit += env.get_sh_profit()
					accumulated_action_taken += env.get_action_taken()
					accumulated_sharpe_ratio += env.get_sharpe_ratio()
					accumulated_portfolio_leverage += env.get_portfolio_leverage()
			
					timestep_time_end = time.time()

					print(f'rank: {rank}, day: {idx}, step: {timestep}, accumulated profit: {accumulated_profit}, accumulated step reward: {accumulated_step_reward}, accumulated position: {accumulated_position}, step time: {timestep_time_end - timestep_time_start}, accumulated bh profit: {accumulated_bh_profit}, accumulated sh profit: {accumulated_sh_profit}, accumulated action taken: {accumulated_action_taken}, accumulated sharpe ratio: {accumulated_sharpe_ratio}, accumulated portfolio leverage: {accumulated_portfolio_leverage}')
					print(100*'=')

					f.write(f'rank: {rank}, day: {idx}, step: {timestep}, accumulated profit: {accumulated_profit}, accumulated step reward: {accumulated_step_reward}, accumulated position: {accumulated_position}, step time: {timestep_time_end - timestep_time_start}, accumulated bh profit: {accumulated_bh_profit}, accumulated sh profit: {accumulated_sh_profit}, accumulated action taken: {accumulated_action_taken}, accumulated sharpe ratio: {accumulated_sharpe_ratio}, accumulated portfolio leverage: {accumulated_portfolio_leverage}\n')

					
					

				else:
					env.step(0, timestep)
	

		


def init_process(rank, config, world_size):
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '8000'
	print(f"Running basic DDP example on rank {rank}.")
	dist.init_process_group(backend=config['backend'], rank=rank, world_size=world_size)
	data_parallel_group = torch.distributed.new_group(ranks=[rank], backend=config['backend'])
	tensor_parallel_group = torch.distributed.new_group(ranks=[rank], backend=config['backend'])
	print('creating dataset')
	print('loaders created')
	create_torch_group(rank, tensor_parallel_group, data_parallel_group, config)
	dist.destroy_process_group()



if __name__ == '__main__':
	size=2
	max_num_epochs = 100
	config = {
		#'embedding_size': tune.choice([2, 4]),
		#'transformer_size': tune.choice([2048, 4096, 8192]),
		'transformer_size': 768,
		'use_streaming': True,
		'transformer_attention_size': 64,
		"epochs": max_num_epochs,
		"learning_rate": 2.5e-6,
		#"lr": tune.choice([5e-4]),
		"batch_size": 36,
		'prefetch': 1024,
		'num_workers': 6,
		'use_scheduler': False,
		#'model_depth': tune.choice(['shallow']),
		'model_depth': 'normal',
		'dropout': 0.1,
		#'transformer_init_method': tune.choice([initialize_normal, initialize_kaiming_uniform, initialize_kaiming_normal, initialize_xavier_uniform, initialize_xavier_normal]),
		'optimizer': 'AdamW',
		'accumulation_iter': 4,
		#'optimizer': tune.choice(['SGD', 'Adam', 'AdamW'])
		'backend': 'nccl',
		'fuse_qkv': False,
		'num_threads': 3,
		'envs_per_thread': 12,
		'epsilon': .9,
		'end_buffer': 64,
		'start_cash': 50000,

	}
	
	#os.environ['CUDA_LAUNCH_BLOCKING']='1'
	
	print('dataset loaded')
	mp.set_start_method("spawn")
	processes = []
	for rank in range(size):
		print(f'rank: {rank}')
		p = mp.Process(target=init_process, args=(rank, config, size))
		print('pre start')
		p.start()
		print('started')
		processes.append(p)
		print('appended')

	for p in processes:
		print('joining')
		p.join()
