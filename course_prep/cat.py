"""
This file is a test of my own python skillz.
@author: louiskhub
"""

import string

class CAT:
    def __init__(self, name):
        self._name = name
    def _greet(self, other_cat):
        print("Hi, I am {0}. Nice to meed you {1}!".format(self._name, other_cat._name))
    def _generate(self, limit):
        string = "Meow "
        for i in range(limit):
            yield string
            string += string