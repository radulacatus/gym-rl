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
parser.add_argument('--mode', choices=['train', 'simulate','train_continue'], default='train')
parser.add_argument('--model_file', type=str ,default=None)
args = parser.parse_args()

base_folder = '../local/'
hidden_layer_sizes = [100, 100]

best_weights = []
max_score = 0

def save_weights():
    global best_weights
    global max_score
    np.savez(base_folder + args.model_file, max_score=max_score, best_weights=best_weights)

def load_weights():
    global best_weights
    global max_score
    saved = np.load(base_folder + args.model_file + '.npz')
    max_score = saved['max_score']
    best_weights = saved['best_weights']

def create_net(hidden_layer_sizes, env, use_saved_weights = False):
    global best_weights
    i = Input(shape=env.observation_space.shape)
    h = BatchNormalization()(i)
    for index, layer_size in enumerate(hidden_layer_sizes):
        h = Dense(layer_size, activation=args.activation_function)(h)
        if index != len(hidden_layer_sizes) - 1:
            h = BatchNormalization()(h)
    h = Dense(env.action_space.n + 1)(h)
    o = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(h)
    model = Model(input=i, output=o)
    if use_saved_weights == True:
        model.set_weights(best_weights)
    
    return model

def train(load_from_file = False):
    global best_weights
    global max_score

    if load_from_file:
        load_weights()

    env = gym.make(args.environment)

    theta = create_net(hidden_layer_sizes, env, load_from_file)
    theta.compile(optimizer='adam', loss='mse')

    theta_bar = create_net(hidden_layer_sizes, env)
    theta_bar.set_weights(theta.get_weights())

    memory = []

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
    save_weights()

def simulate():
    env = gym.make(args.environment)
    load_weights()
    theta = create_net(hidden_layer_sizes, env, True)
    
    obs = env.reset()
    while True:
        env.render()
        action = np.argmax(theta.predict(np.array([obs])))
        obs, reward, done, info = env.step(action)
        if done:
            print 'done:', reward
            break

if __name__ == "__main__":
    load_from_file = False
    if args.mode == 'train_continue':
        assert len(args.model_file) > 0, "Provide a model file to continue training"
        load_from_file = True

    if args.mode == 'train' or args.mode == 'train_continue':
        atexit.register(save_weights)
        train(load_from_file)
    
    if args.mode == 'simulate':
        assert len(args.model_file) > 0, "Provide a model file to simulate"
        simulate()