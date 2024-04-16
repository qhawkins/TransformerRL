# Reinforcement Learning Trading Model

This repository contains an implementation of a reinforcement learning (RL) model for trading based on historical limit-order book data. The model utilizes the Actor-Critic algorithm and is trained using PyTorch and Transformer Engine for efficient computation and distributed training.

## Features

- RL-based trading model that learns from historical order book data
- Utilizes the Actor-Critic algorithm for training
- Supports distributed training using PyTorch's distributed data parallel (DDP)
- Efficient data preprocessing and parsing using Numba
- Backtesting engine for evaluating the trained model's performance
- Customizable trading environment with various performance metrics

## File Structure

- `backtesting_engine.py`: Code for backtesting the trained RL model
- `environment.py`: Defines the trading environment class `Environment`
- `execute_trade.py`: Contains the `execute_trade` function for executing trades
- `find_fill_price.py`: Defines the `find_fill_price` function for determining fill prices
- `main.py`: Main training loop for the RL model
- 'actor_model.py': Actor model utilizing transformers
- 'critic_model.py': Critic model utilizing transformers
- 'result_viewer.py': Code to interpret results
- 'weighted_future_rewards.py': Code to calculate the weighted future rewards

## Dependencies

- Python 3.x
- PyTorch
- Numba
- NumPy
- Transformer Engine

## Usage

1. Install the required dependencies:
```
pip install torch numba numpy transformer-engine
```

2. Prepare your historical order book data in the appropriate format.

3. Modify the `config` dictionary in `main.py` to adjust the training hyperparameters and model configuration.

4. Run the training script:
```
python main.py
```

5. The trained model will be saved periodically during the training process.

6. To backtest the trained model, use the `backtesting_engine.py` script:
```
python backtesting_engine.py
```

7. The backtesting results will be logged in the specified logging directory.
