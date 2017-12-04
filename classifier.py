import numpy as np 
import tensorflow as tf 

class SwitchClassifier():
	def __init__(self, inputs):
		self.inputs = self.convert_to_bgr(inputs)
		self.kernel_initializer = tf.truncated_normal_initializer(
			mean=0.0, stddev=0.01)
		self.network = self.create_network(self.inputs)
		self.output_layer = self.network['softmax']
		# TODO
		#self.train_output
		#self.test_output
		#self.parameters


	def convert_to_bgr(self, inputs):
		# https://github.com/Lasagne/Recipes/issues/20
		MEAN_VALUE = np.array([103.939, 116.779, 123.68])
		inputs = inputs[::-1] # I think this is wrong - in data_reader they convert to gray
		inputs -= MEAN_VALUE
		return inputs

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
		network = {}
		network['inputs'] = tf.reshape(-1, None, None, 1)# this probably won't work, but need to figure out what to do about variable sized images
		network['layer1_conv1'] = self.layer(inputs, 64, 3)
		network['layer1_conv2'] = self.layer(network['layer1_conv1'], 64, 3)
		network['pool1'] = self.pool(network['layer1_conv2'])
		network['layer2_conv1'] = self.layer(network['pool1'], 128, 3)
		network['layer2_conv2'] = self.layer(network['layer2_conv1'], 128, 3)
		network['pool2'] = self.pool(network['layer2_conv2'])
		network['layer3_conv1'] = self.layer(network['pool2'], 256, 3)
		network['layer3_conv2'] = self.layer(network['layer3_conv1'], 256, 3)
		network['layer3_conv3'] = self.layer(network['layer3_conv2'], 256, 3)
		network['pool3'] = self.pool(network['layer3_conv3'])
		network['layer4_conv1'] = self.layer(network['pool3'], 512, 3)
		network['layer4_conv2'] = self.layer(network['layer4_conv1'], 512, 3)
		network['layer4_conv3'] = self.layer(network['layer4_conv2'], 512, 3)
		network['pool4'] = self.pool(network['layer4_conv3'])
		network['layer5_conv1'] = self.layer(network['pool4'], 512, 3)
		network['layer5_conv2'] = self.layer(network['layer5_conv1'], 512, 3)
		network['layer5_conv3'] = self.layer(network['layer5_conv2'], 512, 3)
		#https://www.tensorflow.org/api_docs/python/tf/reduce_mean
		network['global_mean_pooling'] = tf.reduce_mean(network['layer5_conv3'])
		network['dense'] = tf.layers.dense(network['global_mean_pooling'], 512) # check that the defaults on lasagne and tensorflow are the same
		network['softmax'] = tf.layers.dense(network['dense1'], activation=tf.nn.softmax)

		return network



			

