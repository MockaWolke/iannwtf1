import numpy as np
from eval import  accuracy, squared_error

def epoch(mlp, data):
    
    input_values, targets = data

    accuracy_of_loss = 0
    
    for i, input_value in enumerate(input_values):
        
        output = mlp.forward_step(input_value)
        
        accuracy_of_loss+= accuracy(output,targets[i])
        mlp.backprob_step(targets[i])
        
    return accuracy_of_loss / len(input_values)
    
def live_epoch(mlp,data):
    
    input_values, targets = data

    accuracy_of_loss = 0
    
    sq_error = 0
    
    for i, input_value in enumerate(input_values):
        
        output = mlp.forward_step(input_value)
        
        accuracy_of_loss+= accuracy(output,targets[i])
        sq_error += suared_error(output,targets[i])
        mlp.backprob_step(targets[i])
        
    return accuracy_of_loss / len(input_values) , sq_error /len(input_values)


