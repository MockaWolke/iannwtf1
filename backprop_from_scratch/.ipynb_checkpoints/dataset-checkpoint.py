"""" 
Dataset

Provides data with labels: and, or, nand, nor, xor
"""
import numpy as np

# all resonable logic gate inputs
inputs = np.array([(1,0),(1,1),(0,0),(0,1)]) 

# and labels
and_lables = np.array([i[0] & i[1] for i in inputs])

# or labels 
or_lables = np.array([i[0] | i[1] for i in inputs])


def random_data(kind='xor'):
    """ random_data returns labeled data dependent on input parameter
        ## Params
        
        -  kind = "and" or "or" or "nand" or "nor" or "xor" (standard = "xor")
        
    """
    
    if kind == "and":
        return inputs, and_lables

    if kind == "or":
        return inputs, or_lables

    if kind == "nand":
        return inputs,1- and_lables
    
    if kind == "nor":
        return inputs, 1- or_lables

    if kind == "xor":
        return inputs, 1- (and_lables & or_lables)