""" 
Eval skript

Skript provides needed functions for our MLP
"""

import numpy as np


def sigmoid(drive):
    """Sigmoid activation function. """
    return 1 / (1 + np.exp(-drive))


def sigmoidprime(drive):  
    """ Derivative of the sigmoid activation function. """
    exp = np.exp(drive)
    return exp / (exp + 1)**2


def squared_error(t, y):
    """ Loss function (qualitative performance measure). """
    return (t - y)**2


def accuracy(t, y, threshold = 0.5):                        
    """ Accuracy (Quantitative performance measure). """
    return np.abs(t - y) <= threshold # if the differenec of y and the target is balow 0.5, return true, else false


if __name__ == "__main__":
    print(sigmoid(7))
    print(sigmoidprime(7))