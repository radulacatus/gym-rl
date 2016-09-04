import atexit
import argparse
import gym
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam

parser = argparse.ArgumentParser()
parser.add_argument('--random_chance', type=float, default=0.99)
parser.add_argument('--random_decay', type=float, default=0.99)
parser.add_argument('--discount_factor', type=float, default=0.99)
parser.add_argument('--nr_episodes', type=int, default=3000)
parser.add_argument('--nr_timesteps', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--memory_size', type=int, default=5000)
parser.add_argument('--training_steps', type=int, default=10)
parser.add_argument('--target_replace_freq', type=int, default=10)
parser.add_argument('--activation_function', type=str ,default='relu') # tanh
parser.add_argument('--environment', type=str ,default='CartPole-v0') # MountainCar-v0
parser.add_argument('--mode', choices=['train', 'simulate'], default='train')
parser.add_argument('--model_file', type=str ,default=None)
args = parser.parse_args()

base_folder = '../local/'
best_weights = []
hidden_layer_sizes = [8,4]

def create_net(hidden_layer_sizes, env):
    i = Input(shape=env.observation_space.shape)
    h = BatchNormalization()(i)
    for index, layer_size in enumerate(hidden_layer_sizes):
        h = Dense(layer_size, activation=args.activation_function)(h)
        if index != len(hidden_layer_sizes) - 1:
            h = BatchNormalization()(h)
    h = Dense(env.action_space.n + 1)(h)
    o = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(h)
    return i, o

def train():
    global best_weights
    env = gym.make(args.environment)

    i, o = create_net(hidden_layer_sizes, env)
    theta = Model(input=i, output=o)
    theta.compile(optimizer='adam', loss='mse')

    i, o = create_net(hidden_layer_sizes, env)
    theta_bar = Model(input=i, output=o)
    theta_bar.set_weights(theta.get_weights())

    memory = []
    max_score = 0

    for ep in range(args.nr_episodes):
        obs = env.reset()
        score = 0
        random_hits = 0
        for t in range(args.nr_timesteps):        
            if np.random.random() < args.random_chance:
                action = env.action_space.sample()
                random_hits += 1
            else:
                q = theta.predict(np.array([obs]))
                action = np.argmax(q[0])

            new_obs, reward, done, _ = env.step(action)

            score += reward

            memory.append([obs, action, reward, new_obs, done])

            if len(memory) > args.memory_size:
                memory.pop(0)

            obs = new_obs

            for k in range(args.training_steps):
                if len(memory) < args.batch_size:
                    break
                    
                selection = random.sample(memory, args.batch_size)
                obs_list = np.array([r[0] for r in selection])
                newobs_list = np.array([r[3] for r in selection])
                target_q = theta.predict(obs_list)
                next_q_values = theta_bar.predict(newobs_list)
                for i, run in enumerate(selection):
                    _, a, r, _, d = run
                    target_q[i, a] = r
                    if not d:
                        target_q[i, a] += args.discount_factor * next_q_values[i,a]
                theta.train_on_batch(obs_list, target_q)

            if done:
                break
            
        if score >= max_score:
            max_score = score
            best_weights = theta.get_weights()
            print 'save for score ' + str(score)

        if ep % args.target_replace_freq == 0:
            theta_bar.set_weights(theta.get_weights())

        args.random_chance *= args.random_decay

        if ep % 1 == 0:
            print "Episode {} score: {}".format(ep, score)
            # print "random hits: {} total memory frames: {}".format(random_hits, len(memory))

def simulate():
    env = gym.make(args.environment)

    best_weights = np.load(base_folder + args.model_file + '.npy')

    i, o = create_net(hidden_layer_sizes, env)
    theta = Model(input=i, output=o)
    theta.set_weights(best_weights)
    
    obs = env.reset()
    while True:
        env.render()
        action = np.argmax(theta.predict(np.array([obs])))
        obs, reward, done, info = env.step(action)
        if done:
            print 'done:', reward
            break

def save_best_weights():
    np.save(base_folder + args.model_file, best_weights)

if __name__ == "__main__":
    if args.mode == 'train':
        atexit.register(save_best_weights)
        train()
    if args.mode == 'simulate':
        assert len(args.model_file) > 0, "Provide a model file to simulate"
        simulate()