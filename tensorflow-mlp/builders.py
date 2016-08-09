from neural import NeuralNetwork
from qlearners import GymQLearner
from gym import spaces

def size_of_space(space):
    if isbox(space):
        return space.shape[0]
    if isdiscrete(space):
        return space.n

def isbox(space):
    return type(space) is spaces.Box

def isdiscrete(space):
    return type(space) is spaces.Discrete

class DeepQlearnerBuilder:

    def __init__(self, env):
        self.env = env
        self.network_hyperparams = {
            'learning_rate': 1e-2,
            'training_epochs': 5,
            'batch_size': 5,
            'display_step': 1
            }
        self.learner_hyperparams = {
            'initial_random_chance': 0.9,
            'random_decay': 0.9,
            'replay_memory': 5000,
            'batch_size': 2000,
            'mini_batch_size':5,
            'discount_factor': 0.95
            }
        self.ql_opt = {
            'save_model': False,
            'model_filename': '',
            'display_steps': 0,
            'normalize_reward': False,
            'reward_min': -1,
            'reward_max': -1,
            'reward_newmin': -1,
            'reward_newmax': -1
            }
        self.hidden_layer_sizes = []
        self.cost_function = 'mse'
    
    def nn_hyperparams(self, params):
        self.network_hyperparams = params
        return self

    def ql_hyperparams(self, params):
        self.learner_hyperparams = params
        return self

    def nn_hidden_layers(self, layer_sizes):
        self.hidden_layer_sizes = layer_sizes
        return self

    def nn_cost_function(self, cost_function):
        self.cost_function = cost_function
        return self

    def ql_opt_save_model(self, filename):
        self.ql_opt['save_model'] = True
        self.ql_opt['model_filename'] = filename
        return self

    def ql_opt_display_steps(self, nr_steps):
        self.ql_opt['display_steps'] = nr_steps
        return self
    
    def ql_opt_normalize_score(self, min, max, newmin, newmax):
        self.ql_opt['normalize_reward'] = True
        self.ql_opt['reward_min'] = min
        self.ql_opt['reward_max'] = max
        self.ql_opt['reward_newmin'] = newmin
        self.ql_opt['reward_newmax'] = newmax
        return self

    def build_network(self):
        shape = [size_of_space(self.env.observation_space)] + self.hidden_layer_sizes + [size_of_space(self.env.action_space)]
        return NeuralNetwork(shape, self.network_hyperparams, self.cost_function)

    def build_qlearner(self):
        nn = self.build_network()
        action_space_size = size_of_space(self.env.action_space)
        return GymQLearner(nn, self.env, action_space_size, self.learner_hyperparams, self.ql_opt)



def create_deep_qlearner(env, hidden_layer_sizes, model_filename):
    builder = DeepQlearnerBuilder(env)
    builder.nn_cost_function('top1mse')\
        .ql_opt_save_model(model_filename)\
        .nn_hidden_layers(hidden_layer_sizes)\
        .ql_opt_normalize_score(-1,0,0, 1)

    return builder.build_qlearner()

def create_neural_network(env, hidden_layer_sizes):
    builder = DeepQlearnerBuilder(env)
    builder.nn_cost_function('top1mse')\
        .nn_hidden_layers(hidden_layer_sizes)

    return builder.build_network()