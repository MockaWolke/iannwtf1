import numpy as np

inputs = np.array([(1,0),(1,1),(0,0),(0,1)], dtype=np.bool_) # all reasonable logic gate inputs

and_lables = np.array([i[0] & i[1] for i in inputs])
or_lables = np.array([i[0] | i[1] for i in inputs])


def random_data(kind='xor'):
    
    if kind == "and":
        return inputs, and_lables

    if kind == "or":
        return inputs, or_lables

    if kind == "nand":
        return inputs, ~ and_lables

    if kind == "nor":
        return inputs, ~ or_lables

    if kind == "xor":
        return inputs, ~ and_lables & or_lables