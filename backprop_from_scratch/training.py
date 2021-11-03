import numpy as np
from mlp import MLP
from dataset import random_data
from eval import squared_error, accuracy

def train(MLP = MLP(), kind = "xor"):
    
    inputs, t = random_data(kind)

    MLP.forward_step(inputs[0])
    MLP.backprob_step(t[0])
    
    return True

print(train())