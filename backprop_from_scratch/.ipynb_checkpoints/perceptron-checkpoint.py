"""
Class Perceptron

Represents one Perceptron in our MLP framework. It is initialized with an input layer width, a learning rate,
an activateion function (sigmoid) and its derivative (sigmoidprime).
Within this class are also methods to calculate a forward step and upgrade the Perceptrons parameters.
"""

import numpy as np
from eval import sigmoid, sigmoidprime

class Perceptron():
    """ Initializes a Perceptron.
        Passed to MLP.
        Methods:
            -   _forward_step     Calculates activation.
            -   _update           Updates neurons params.
    """

    def __init__(self, input_units, learning_rate = 1, activ_func = sigmoid, activ_func_derivate= sigmoidprime):
        """ Constructor
            ## Parameters:
        
            -   input_units = input layer width
            -   learning_rate = param update rhythm
            -   activ_func = activation function
        """

        # assign random weights and bias on initialisation
        self.weights = np.random.randn(input_units)
        self.bias = np.random.randn(1)              
        self.drive = np.nan
        self.alpha = learning_rate
        self.activation = activ_func
        self.activation_func_der = activ_func_derivate
        self.output = np.nan
        self.delta = np.nan
        

    def _forward_step(self, inputs):
        """ Calculates activation through the inputs.
            
            ## Parameters:

            inputs = activations from earlier layer neurons
        """
        
        self.incoming_activations = inputs
        self.drive = np.dot(inputs, self.weights) + self.bias
        self.output = float(self.activation(self.drive))
        
        return self.output


    def _update(self, delta):
        """ Updates neurons parameters with delta.

            ## Parameters:

            delta = error signal
        """
        
        self.delta = delta
        # gradient calc
        gradient = self.delta * self.incoming_activations
        
        # param update
        self.weights -=  self.alpha * gradient
        self.bias -= self.alpha * delta