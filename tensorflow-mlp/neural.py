import tensorflow as tf
import numpy as np

class NeuralNetwork:
    def init_default_params(self):
        self.hyperparams = {
            'learning_rate': 1e-2,
            'training_epochs': 15,
            'batch_size': 100,
            'display_step': 1
            }

        self.cost_function_dict = {
            'mse': self.mean_squared_error,
            'top1mse': self.top1_mean_squared_error,
            'cross_entropy': self.cross_entropy
        }

    def __init__(self, layer_sizes, hyperparams = None, cost_function = None):
        self.init_default_params()

        if hyperparams:
            self.hyperparams = hyperparams
        
        self.configure_cost_function(cost_function)

        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]

        self.input_layer = tf.placeholder(tf.float32, [None, self.input_size], name='input_placeholder')
        self.targets = tf.placeholder(tf.float32, [None, self.output_size], name='target_placeholder')
        self.layer_list = [self.input_layer]
        self.weight_list = []
        self.bias_list = []
        self.sess = None

        for layer_index, layer_size in enumerate(layer_sizes[1:],1):
            with tf.name_scope('layer_'+ str(layer_index)):
                previous_size = layer_sizes[layer_index - 1]
                
                weight = tf.Variable(tf.truncated_normal([previous_size, layer_size]), 
                    name=("weights_" + str(layer_index)))
                    
                bias = tf.Variable(tf.constant(1e-3,shape=[layer_size]), 
                    name=("bias" + str(layer_index)))
                
                if layer_index != len(layer_sizes) - 1:
                    layer = self.relu_layer(self.layer_list[-1], weight, bias)
                    self.layer_list.append(layer)
                else:
                    layer = tf.matmul(self.layer_list[-1], weight) + bias
                    self.layer_list.append(layer)

                self.weight_list.append(weight)
                self.bias_list.append(bias)
                
        # self.param_mean = tf.reduce_mean(self.weight_list[-1])
        # self.bias_mean = tf.reduce_mean(self.bias_list)
        # tf.scalar_summary('param_mean', self.param_mean)
        # tf.scalar_summary('bias_mean', self.bias_mean)
        # self.summary = tf.merge_all_summaries()
        # self.summary_writer = tf.train.SummaryWriter(".", self.get_session().graph)

        self.cost = self.calculate_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hyperparams['learning_rate']).minimize(self.cost)

    def configure_cost_function(self, cost_function = None):
        if cost_function == None or cost_function not in self.cost_function_dict:
            cost_function = self.cost_function_dict.keys()[0]
        
        self.calculate_cost = self.cost_function_dict[cost_function]

    def top1_mean_squared_error(self):
        val, index = tf.nn.top_k(self.targets,k=1,sorted=False)
        val = tf.squeeze(val)
        mask = tf.one_hot(index[:,0], self.output_size, on_value=1., off_value=0.,axis=None,dtype=tf.float32)
        top1 = tf.reduce_sum(tf.mul(self.layer_list[-1], mask), reduction_indices=[1])
        return tf.reduce_mean(tf.square(val - top1))

    def mean_squared_error(self):
        return tf.reduce_mean(tf.square(tf.reduce_sum(self.layer_list[-1] - self.targets, reduction_indices=[1])))

    def cross_entropy(self):
        return tf.nn.sigmoid_cross_entropy_with_logits(self.layer_list[-1], self.targets, name=None)
    
    def train(self, inputs, targets,training_epochs=None):
        if training_epochs==None:
            training_epochs=self.hyperparams['training_epochs']
        input_size = np.ma.size(inputs,0)
        batch_size = self.hyperparams['batch_size']
        batch_size = batch_size if batch_size != -1 else input_size
        total_batch = int(input_size/batch_size)
        sess = self.get_session()

        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                batch_x = inputs[i*batch_size:(i+1)*batch_size,:]
                batch_y = targets[i*batch_size:(i+1)*batch_size]
                sess.run([self.optimizer, self.cost], feed_dict={self.input_layer: batch_x, self.targets: batch_y})
    
    def predict(self, input):
        return self.get_session().run(self.layer_list[-1], feed_dict={self.input_layer: input})

    def predict_argmax(self, input):
        return np.argmax(self.predict(input))

    def relu_layer(self, input, weights, biases):
        layer = tf.add(tf.matmul(input, weights), biases)
        layer = tf.nn.relu(layer)
        return layer

    def get_session(self):
        if not self.sess:
            self.sess = tf.Session()
            self.sess.run(tf.initialize_all_variables())
        return self.sess

    def save(self, filename):
        sess = self.get_session()
        saver = tf.train.Saver()
        save_path = saver.save(sess, "/tmp/" + filename)
        return save_path

    def load(self, filename):
        sess = self.get_session()
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/" + filename)