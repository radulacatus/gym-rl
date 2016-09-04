import gym
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD, Nadam

# Init environment
game = "MountainCar-v0"
game = "CartPole-v0"
env = gym.make(game)

action_space = env.action_space.n
observation_space = env.observation_space.high.size

# create multi-layered perceptron
net = Sequential()
net.add(Dense(20, input_shape=(observation_space,)))
net.add(Activation('relu'))
#net.add(Dropout(0.2))
net.add(Dense(10))
net.add(Activation('relu'))
#net.add(Dropout(0.2))
net.add(Dense(action_space))
net.add(Activation('softmax'))

net.summary()

# Hyperparams
random_chance = 0.999
random_chance_decay = 0.999
learning_rate = 0.005
discount_factor = 0.99 #gamma
num_episodes = 3000
num_timesteps = 200
nb_epoch = 50

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=learning_rate)
nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
net.compile(loss='mse', #loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

def sample(epsilon, obs):
    vals = np.array(net.predict(obs, batch_size=len(obs), verbose=0))
    if np.random.random() < epsilon:
        action = env.action_space.sample()
        vals[0,action] = 1
    return (np.argmax(vals, axis=1)[0], vals)

def train(obs, target):
    net.train_on_batch(obs, target)

# SARSA loop
for ep in range(num_episodes):
    obs = np.array([env.reset()])
    action, action_value = sample(random_chance, obs)
    score = 0
    for t in range(num_timesteps):
        if ep % 50 == -1:
            env.render()
        new_obs, reward, done, info = env.step(action)
        new_obs = np.array([new_obs])
        new_action, new_action_value = sample(random_chance, new_obs)
        target = action_value.copy()
        target[0,action] = reward - action_value[0,action]
        if done == False:
            target[0,action] += discount_factor * new_action_value[0,action]

        train(obs, target)

        score += reward

        obs = new_obs
        action, action_value = sample(random_chance, obs)
        if done:
            break

    random_chance *= random_chance_decay
    if ep % 500 == 0:
        print "Episode {} score: {}".format(ep, score)