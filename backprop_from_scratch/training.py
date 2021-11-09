"""
Training skript

Provides functions for training our MLP and plotting results.
"""

import numpy as np
from eval import  accuracy, squared_error
import matplotlib.pyplot as plt


def epoch(mlp, data):
    """
    Function to pass data forward and backward through our MLP.
    ## Params
        -   mlp = object of class MLP
        -   data = data from dataset.py ("and", "or", "nand", "nor, "xor)
    """
    input_values, targets = data # split up the date into inputs and targets

    accuracy_of_loss = 0 # this variable will count the acuracy
    
    for i, input_value in enumerate(input_values): 
        
        output = mlp.forward_step(input_value) # do the forard_step
        
        accuracy_of_loss+= accuracy(output,targets[i]) # measure the acuracy
        mlp.backprob_step(targets[i]) # adjust weights
        
    return accuracy_of_loss / len(input_values) # divide by the length of inputs to get the average
    

def train(mlp,data, times= 10000):
    """
    Prints and plots our MLP's predictions.
    ## Params
        -   mlp = object of class MLP
        -   data = data from dataset.py ("and", "or", "nand", "nor, "xor)
        -   times = iterations
    """
    erg=[] # here we will store the acuracys of the epochs
    x_axes= range(times) # our x_axes
    for i in x_axes: # for every time step
        erg.append(epoch(mlp,data)) # do one epoch, and append the value to erg
    fig = plt.figure(figsize=(10,4)) # add the end plot the result
    plt.plot(x_axes,erg)
    plt.show()
    
    for i,logic_input in enumerate(data[0]):
        print(f"Our data is: {logic_input}, the target is {data[1][i]}, our mlp returns {float(mlp.forward_step(logic_input)[0])}") # and print our values compared to the targets


def live_epoch(mlp,data):
    """
    This is basically just the same as our epoch(), but is also returns the sqared_error distance to the targets additionally to the acurracy
    ## Params
        -   mlp = object of class MLP
        -   data = data from dataset.py ("and", "or", "nand", "nor, "xor)
    """
    input_values, targets = data

    accuracy_of_loss = 0
    
    sq_error = 0
    
    for i, input_value in enumerate(input_values):
        
        output = mlp.forward_step(input_value)
        
        accuracy_of_loss+= accuracy(output,targets[i])
        sq_error += squared_error(output,targets[i])
        mlp.backprob_step(targets[i])
        
    return accuracy_of_loss / len(input_values) , sq_error /len(input_values)