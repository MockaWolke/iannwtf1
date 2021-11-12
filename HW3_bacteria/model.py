"""
Class Model
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class MyModel(tf.keras.Model):
    """"""
    
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden1 = Dense_Layer()
        self.hidden2 = Dense_Layer()
        self.out = Dense_Layer(10, tf.nn.softmax)

        
    @tf.function
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.out(x)
        return x