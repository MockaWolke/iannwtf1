import numpy as np

def sigmoid(di):                            # sigmoid activation func
    return 1 / (1 + np.exp(-di))

def sigmoidprime(di):                       # derivative of the sigmoid activation func
    return sigmoid(di) * (1 - sigmoid(di))