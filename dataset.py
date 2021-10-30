import numpy as np

inputs = np.array([(1,0),(1,1),(0,0),(0,1)], dtype=np.bool_)

and_lables = np.array([i[0] & i[1] for i in inputs])
or_lables = np.array([i[0] | i[1] for i in inputs])
nand_lables = ~ and_lables
nor_lables = ~ or_lables
xor_lables = nand_lables & or_lables