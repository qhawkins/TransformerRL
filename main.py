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
	net = ActorModel()
	net_state_dict = net.state_dict()
	raw_state_dict = torch.load('/media/qhawkins/Archive/MLM Models/mlm_model_cluster_2_ddp/model_1_0.004021_normal_1024_64_40.pth')
	
	# Remove the module prefix from the keys
	parsed_state_dict = {key.replace('module.', ''): value for key, value in raw_state_dict.items()}

	# replace all items in net state dict that have correesponding keys in parsed state dict with their matching values
	net_state_dict = {key: parsed_state_dict[key] if key in parsed_state_dict else value for key, value in net_state_dict.items()}
	net.load_state_dict(net_state_dict)
	return net

def get_action(ob_state, env_state, model, device):
	mask = mask_tokens(ob_state, 0).to(device, non_blocking=True)
	action = model(mask, ob_state, env_state)
	return action
	# Initialize transformer


def main():
	model = load_model()
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
			
			environment = Environment(prices=raw_ob, offset_init = 256, gamma_init=.09, time=0)
			environment.reset(raw_ob, 100000, 0, 100000)
			for timestep in range(parsed_file.shape[0]):
				if timestep > 256:
					ob_state = parsed_file[timestep-256:timestep, :, :]
					env_state = environment.get_state(timestep)
					action = get_action(ob_state, env_state, model, device='cuda:0')
					environment.step(action, timestep)
				else:
					environment.step(0, timestep)
				
				print(f'timestep: {timestep}')
				print(f'cash: {environment.cash}')
				print(f'position: {environment.position}')
				print(f'account value: {environment.account_value}')
				print(75*'=')
			
			
			
			
			print(100*'=')
			"""
			if idx >10:
				exit()
			"""
			

	# Initialize environment
	
	return 0

if __name__ == '__main__':
	model = 
	main()