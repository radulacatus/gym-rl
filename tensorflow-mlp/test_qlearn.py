import builders
import gym
import sys
import numpy as np

game = "CartPole-v0"

hidden_shape = [10,10]

def train(model_filename):
    env = gym.make(game)
    qlearner = builders.create_deep_qlearner(env, hidden_shape, model_filename)
    qlearner.train()

def simulate(saved_model_filename):
    env = gym.make(game)
    net = builders.create_neural_network(env, hidden_shape)
    net.load(saved_model_filename)

    observation = env.reset()

    while True:
        env.render()
        action = np.argmax(net.predict([observation])[0])
        observation, reward, done, info = env.step(action)
        if done:
            print 'done:', reward
            break

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Please only provide 3 args"
    command = sys.argv[1]
    model_filename = sys.argv[2]
    if command == 'train':
        train(model_filename)
    if command == 'simulate':
        simulate(model_filename)