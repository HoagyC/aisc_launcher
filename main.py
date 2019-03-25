# TODO put on group github and add how to run docs

import numpy as np
import random
import os
import tensorflow as tf
import stable_baselines.common.policies as policies

from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.results_plotter import load_results, ts2xy
from models import a2c, acer, acktr, deepq, ddpg, ppo1, ppo2, sac, trpo

FUNC_DICT = {
    'a2c': lambda e: a2c.A2C(policy="MlpPolicy", env=e),
    'acer': lambda e: acer.ACER(policy="MlpPolicy", env=e),
    'acktr': lambda e: acktr.ACKTR(policy="MlpPolicy", env=e),
    'dqn': lambda e: deepq.DQN(policy="MlpPolicy", env=e),
    'ddpg': lambda e: ddpg.DDPG(policy="MlpPolicy", env=e),
    'ppo1': lambda e: ppo1.PPO1(policy="MlpPolicy", env=e),
    'ppo2': lambda e: ppo2.PPO2(policy="MlpPolicy", env=e),
    'sac': lambda e: sac.SAC(policy="MlpPolicy", env=e),
    'trpo': lambda e: trpo.TRPO(policy="MlpPolicy", env=e)
}


# Score logging function copied from docs
def callback(_locals, _globals):
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 100 == 0:
        # Evaluate policy
        # Monitor broken
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                'Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}'.format(best_mean_reward, mean_reward))

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

# Parameters for logging scores
log_dir = "./scores"
best_mean_reward, n_steps = -np.inf, 0

# Choose atari environment
env_name = 'PongNoFrameskip-v4'
time_steps = 1000

# Create and wrap the environment
print('Starting', env_name)
env = make_atari_env(env_id=env_name, num_env=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env = VecNormalize(env)

# Setting all known random seeds
random.seed(0)
np.random.seed(0)
set_global_seeds(0)
tf.random.set_random_seed(0)

# Setting up model of your choice
while True:
    try:
        print("Available models: a2c, acer, acktr, dqn, ddpg, ppo1, ppo2, sac, trpo")
        model_name = input("choose model: ")
        model = FUNC_DICT[model_name](env)
        print('Running ', model_name)
        break
    except KeyError:
        print('Not a recognised model')


model.learn(total_timesteps=time_steps, callback=callback)
# env.save_running_average("./")

model.save('test1')