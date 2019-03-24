import numpy as np
import random
import os
import sys
import gym
import tensorflow as tf
import stable_baselines.common.policies as policies

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.results_plotter import load_results, ts2xy
from models.ppo2 import ppo2
from models.a2c import a2c

log_dir = "./scores"
best_mean_reward, n_steps = -np.inf, 0


# score logging function copied from docs
def callback(_locals, _globals):
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

# Setting log levels to cut out minor errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# Testing ability to run multiple games
print('Available envs: 1 - cartpole(default), 2 - pong')
env_num = input("choose env: ")
print(env_num)
# Create and wrap the environment
if env_num == '2':
    print('starting pong')
    env = make_atari_env(env_id='PongNoFrameskip-v4', num_env=1, seed = 0)
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env)
else:
    env = gym.make('CartPole-v1')
    env.seed(0)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env)

# Setting all random seeds
random.seed(0)
np.random.seed(0)
set_global_seeds(0)
tf.random.set_random_seed(0)

# Setting up model of your choice
print("Available models: 1: A2C (default), 2: PPO2, 3: OUR CUSTOM MODEL")
model_num = input("choose model: ")
if model_num == '2':
    model = ppo2.PPO2(policies.MlpPolicy, env, verbose=1)
elif model_num == '3':
    print('this has not been added, sorry')
    sys.exit()
else:
    model = a2c.A2C(policies.MlpPolicy, env, verbose=1)

try:
    ts = int(input("timesteps: "))
except:
    ts = 1000

model.learn(total_timesteps=ts, callback=callback)
env.save_running_average("./")

model.save('test1')