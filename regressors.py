import numpy as np 
import tensorflow as tf 

class Regressor():
    def __init__(self, inputs):
        self.inputs = inputs
        self.kernel_initializer = tf.truncated_normal_initializer(
            mean=0.0, stddev=0.01)
        self.network = self.create_network(self.inputs)
        self.output_layer = self.network['layer3_conv3']
        # TODO
        #self.train_output
        #self.test_output
        #self.parameters

    def layer(self, inputs, filters, kernel_size):
        return tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            activation=tf.nn.relu
            )

    def pool(self, inputs):
        return tf.layers.max_pooling2d(
            inputs=inputs,
            pool_size=[2,2],
            strides=2
            )

    def create_network(self, inputs):
        pass

class NineByNine(Regressor):
    def create_network(self, inputs):
        network = {}
        network['inputs'] = tf.reshape(-1, None, None, 1)# this probably won't work, but need to figure out what to do about variable sized images
        network['layer1_conv1'] = self.layer(network['inputs'], 16, 9)
        network['pool1'] = self.pool(network['layer1_conv1'])
        network['layer2_conv1'] = self.layer(network['pool1'], 32, 7)
        network['pool2'] = self.pool(network['layer2_conv1'])
        network['layer3_conv1'] = self.layer(network['pool2'], 16, 7)
        network['layer3_conv2'] = self.layer(network['layer3_conv1'], 8, 7)
        network['layer3_conv3'] = self.layer(network['layer3_conv2'], 1, 1)
        return network

class SevenBySeven(Regressor):
    def create_network(self, inputs):
        network = {}
        network['inputs'] = tf.reshape(-1, None, None, 1)# this probably won't work, but need to figure out what to do about variable sized images
        network['layer1_conv1'] = self.layer(network['inputs'], 20, 7)
        network['pool1'] = self.pool(network['layer1_conv1'])
        network['layer2_conv1'] = self.layer(network['pool1'], 40, 5)
        network['pool2'] = self.pool(network['layer2_conv1'])
        network['layer3_conv1'] = self.layer(network['pool2'], 20, 5)
        network['layer3_conv2'] = self.layer(network['layer3_conv1'], 10, 5)
        network['layer3_conv3'] = self.layer(network['layer3_conv2'], 1, 1)
        return network   

class FiveByFive(Regressor):
    def create_network(self, inputs):
        network = {}
        network['inputs'] = tf.reshape(-1, None, None, 1)# this probably won't work, but need to figure out what to do about variable sized images
        network['layer1_conv1'] = self.layer(network['inputs'], 24, 5)
        network['pool1'] = self.pool(network['layer1_conv1'])
        network['layer2_conv1'] = self.layer(network['pool1'], 48, 3)
        network['pool2'] = self.pool(network['layer2_conv1'])
        network['layer3_conv1'] = self.layer(network['pool2'], 24, 3)
        network['layer3_conv2'] = self.layer(network['layer3_conv1'], 12, 3)
        network['layer3_conv3'] = self.layer(network['layer3_conv2'], 1, 1)
        return network           

