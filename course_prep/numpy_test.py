"""
This file is a test of my own python skillz.
@author: louiskhub
"""

import numpy as np

matrix = np.random.normal(loc=0, scale=1, size=(5,5))
matrix = np.where(matrix > 0.09, matrix**2, 42)
print(matrix[:,3])