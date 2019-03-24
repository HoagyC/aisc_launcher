import numpy as np
import random
import os
import gym
import tensorflow as tf
import stable_baselines.common.policies as policies

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2, ACER
from models.a2c import a2c as A2C

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

print("Available models: 1: A2C (default), 2: PPO2, 3: ACER")
model_num = input("choose model: ")
env_num = input("choose env: ")
# Create and wrap the environment
env = gym.make('CartPole-v1')
env.seed(0)
random.seed(0)
np.random.seed(0)
set_global_seeds(0)
tf.random.set_random_seed(0)

env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

if model_num == 2:
    model = PPO2(policies.MlpPolicy, env, verbose=1)
elif model_num == 3:
    model = ACER(policies.MlpPolicy, env, verbose=1)
else:
    model = A2C.A2C(policies.MlpPolicy, env, verbose=1)


print(env.reset())
print(VecNormalize.__mro__)
env.unwrapped.envs[0].seed(0)
obs = env.reset()
print(obs, 'obs')

print(dir(model))

model.learn(total_timesteps=100)
env.save_running_average("./")

model.save('test1')

print(hash(PPO2.load('test1')))

for i in range(20):
    print(obs)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
