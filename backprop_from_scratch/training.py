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
    input_values, targets = data

    accuracy_of_loss = 0
    
    for i, input_value in enumerate(input_values):
        
        output = mlp.forward_step(input_value)
        
        accuracy_of_loss+= accuracy(output,targets[i])
        mlp.backprob_step(targets[i])
        
    return accuracy_of_loss / len(input_values)
    

def train(mlp,data, times= 10000):
    """
    Prints and plots our MLP's predictions.
    ## Params
        -   mlp = object of class MLP
        -   data = data from dataset.py ("and", "or", "nand", "nor, "xor)
        -   times = iterations
    """
    erg=[]
    x_axes= np.linspace(0,10000,num=times)
    for i in range(0,len(x_axes)):
        erg.append(epoch(mlp,data))
    fig = plt.figure(figsize=(10,4))
    plt.plot(x_axes,erg)
    plt.show()
    
    for i,logic_input in enumerate(data[0]):
        print(f"Our data is: {logic_input}, the target is {data[1][i]}, our mlp returns {float(mlp.forward_step(logic_input)[0])}")


def live_epoch(mlp,data):
    """
    Function to pass data forward and backward through our MLP, but for live visualization.
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