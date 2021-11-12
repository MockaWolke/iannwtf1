"""
Class Layer

Defines a Layer for our Model
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class Dense_Layer(tf.keras.layers.Layer):
    """DenseLayer represents a Layer in our Network, inherits from keras.layer.Layer."""
    
    
    def __init__(self, units=256, activation=tf.nn.sigmoid):
        """
        Constructor, calls super Constructor, initializes layer with 256 units and activation function sigmoid.
        ## Params
            - units = units in hidden layer (standard=256)
            - activation = activation function (standard=sigmoid)
        """
        super(Dense_Layer, self).__init__()
        self.units = units
        self.activation = activation
    
    
    def build(self, input_shape): 
        """
        Sets the weights of our network to random, sets an input shape and makes weights trainable.
        ## Params
            - input_shape = input shape for network
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)
  

    def call(self, inputs):
        """
        Update weights/biases of each neuron
        ## Params
            - inputs = inputs for neurons
        """
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x