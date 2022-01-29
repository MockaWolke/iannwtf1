"""
Class Model

Defines a Model
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from layer import Dense_Layer


class MyModel(tf.keras.Model):
    """MyModel represents a Tensorflow Model, inherits from keras.Model."""
    
    
    def __init__(self):
        """Constructor, calls super Constructor, initializes 2 hidden layers and an output layer."""
        super(MyModel, self).__init__()
        self.hidden1 = Dense_Layer()
        self.hidden2 = Dense_Layer()
        self.out = Dense_Layer(10, tf.nn.softmax)

        
    @tf.function
    def call(self, inputs):
        """
        Update weights/biases of each layer (calls call of each neuron).
        ## Params
            - inputs = inputs for network
        """
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.out(x)
        return x