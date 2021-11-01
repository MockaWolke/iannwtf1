import numpy as np



INPUTS = np.array([(1,0),(1,1),(0,0),(0,1)], dtype=np.bool_)   # all reasonable inputs for our logic gates



and_lables = np.array([i[0] & i[1] for i in INPUTS])           # labels for inputs evaluated with AND gate
or_lables = np.array([i[0] | i[1] for i in INPUTS])

nand_lables = ~ and_lables                                     # NOT AND/OR labels are nothing more
nor_lables = ~ or_lables                                       # than the complement of AND/OR labels

xor_lables = nand_lables & or_lables                           # XOR labels are the conjunct of NOT AND and OR