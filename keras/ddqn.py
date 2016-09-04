import gym
import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam

# Init environment
#game = "MountainCar-v0"
game = "CartPole-v0"
env = gym.make(game)

# Hyperparams
epsilon = 0.99 #random chance
epsilon_decay = 0.99 #random chance decay
learning_rate = 0.005
gamma = 0.99 #discount factor
M = 3000 #number of episodes
T = 200 #number of timesteps
Nb = 50 #train batch size
Nr = 500 #replay buffer max size
# Nmb = 50 #min size the buffer must have to train
N_bar = 10 #number of steps before updating the target optimizer
Mt = 20 #number of training steps for the optimizer
activation_function='tanh' # 'relu'


def create_net(hidden_layer_sizes):
    i = Input(shape=env.observation_space.shape)
    h = BatchNormalization()(i)
    for index, layer_size in enumerate(hidden_layer_sizes):
        h = Dense(layer_size, activation=activation_function)(h)
        if index != len(hidden_layer_sizes) - 1:
            h = BatchNormalization()(h)
    h = Dense(env.action_space.n + 1)(h)
    o = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(h)
    return i, o

hidden_layer_sizes = [50,10]

i, o = create_net(hidden_layer_sizes)
theta = Model(input=i, output=o)
theta.compile(optimizer='adam', loss='mse')

i, o = create_net(hidden_layer_sizes)
theta_bar = Model(input=i, output=o)
theta_bar.set_weights(theta.get_weights())

memory = []

for ep in range(M):
    obs = env.reset()
    score = 0
    for t in range(T):
        if ep % 50 == 0:
            env.render()
        
        # sample action
        if np.random.random() < epsilon:
          action = env.action_space.sample()
        else:
          q = theta.predict(np.array([obs]))
          action = np.argmax(q[0])

        new_obs, reward, done, _ = env.step(action)

        score += reward
        memory.append([obs, action, reward, new_obs, done])
        if len(memory) > Nr:
            memory.pop(0)

        obs = new_obs

        for k in range(Mt):
            if len(memory) < Nb:
                break
                
            selection = random.sample(memory, Nb)
            obs_list = np.array([r[0] for r in selection])
            newobs_list = np.array([r[3] for r in selection])
            target_q = theta.predict(obs_list)
            next_q_values = theta_bar.predict(newobs_list)
            for i, run in enumerate(selection):
                _, a, r, _, d = run
                target_q[i, a] = r
                if not d:
                    target_q[i,a] += gamma * next_q_values[i,a]
            theta.train_on_batch(obs_list, target_q)

        if done:
            break
        

    if ep % N_bar == 0:
        theta_bar.set_weights(theta.get_weights())

    epsilon *= epsilon_decay

    if ep % 1 == 0:
        print "Episode {} score: {}".format(ep, score)
