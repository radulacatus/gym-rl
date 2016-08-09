import random
import numpy as np
import math

class GymQLearner:
    
    hyperparams = {
        'initial_random_chance': 0.9,
        'random_decay': 0.9,
        'replay_memory': 15,
        'batch_size': 20,
        'mini_batch_size': 4,
        'discount_factor': 0.9
        }

    options = {
        'save_model': False,
        'model_filename': '',
        'display_steps': 0,
        'normalize_reward': False,
        'reward_min': -1,
        'reward_max': -1,
        'reward_newmin': -1,
        'reward_newmax': -1
        }

    def __init__(self, net, env, action_space, hyperparams = None, options = None):
        if hyperparams:
            self.hyperparams = hyperparams

        if options:
            self.options = options

        self.net = net
        self.env = env
        self.action_space = action_space

    def train(self, 
            max_episode = 1000,
            max_step = 200):
        score = 0
        max_score = 0

        self.random_chance = self.hyperparams['initial_random_chance']
        memory = []
        for i_episode in range(max_episode):
            obs = self.env.reset()
            score = 0
            for t in range(max_step):
                if self.options['display_steps'] != 0 and i_episode % self.options['display_steps'] == 0:
                    self.env.render()

                would_take_action = self.predict_actions([obs])[0]
                if random.random() > self.random_chance:
                    action = would_take_action
                else: 
                    action = self.env.action_space.sample()
                
                newobs, reward, done, info = self.env.step(action)

                if self.options['normalize_reward']:
                    reward = self.normalize_reward(reward)

                score += reward
                memory.append([obs, action, reward, newobs, done])
                if len(memory) > self.hyperparams['replay_memory']:
                    memory.pop(0)

                if done:
                    break
                
                obs = newobs
            
            if score != self.options['reward_newmin'] and self.options['save_model'] and max_score <= score:
                self.net.save(self.options['model_filename'])
                max_score = score
            
            self.random_chance = self.hyperparams["initial_random_chance"]*(1 - score / (max_step * self.options["reward_newmax"]))
            selection = self.pseudo_shuffle(memory)
            inputs = np.array([r[0] for r in selection])
            target_q = self.calculate_target_q(selection)
            self.update_q(inputs, target_q, 1)
            
            if i_episode % 1 == 0:
                # how did we do?
                print "Episode ", i_episode, "\tScore ", score

    def normalize_reward(self, reward):
        reward = (reward - self.options['reward_min']) / (self.options['reward_max'] - self.options['reward_min'])
        reward *= (self.options['reward_newmax'] - self.options['reward_newmin'])
        reward += self.options['reward_newmin']
        return reward

    def predict_actions(self, obs):
        return np.argmax(self.net.predict(obs), axis=1)

    def predict_rewards(self, obs):
        return self.net.predict(obs).max(axis=1)

    def update_q(self, inputs, targets, training_epochs=1):
        self.net.train(inputs,targets, training_epochs)

    def pseudo_shuffle(self, memory):
        batch_size = min(len(memory),self.hyperparams['batch_size'])
        mini_batch_size = min(batch_size,self.hyperparams['mini_batch_size'])
        #split in minibatches
        selection = [memory[i:i+mini_batch_size] for i in xrange(0, batch_size, mini_batch_size)]
        #shuffle minibatches
        selection = random.sample(selection,len(selection))
        #join minibatches
        selection = [x for y in selection for x in y]
        return selection

    def calculate_target_q(self,selection):
        target_q = np.zeros((len(selection), self.action_space))
        newobs_list = np.array([r[3] for r in selection])
        next_q_value = self.predict_rewards(newobs_list)
        for i, run in enumerate(selection):
            obs, action, reward, newobs, done = run
            target_q[i,action] = reward
            if not done:
                # no future Q if action was terminal
                # target_q[i,action] -= self.hyperparams['discount_factor'] * target_q[i,action]
                target_q[i,action] += self.hyperparams['discount_factor'] * next_q_value[i]
        return target_q