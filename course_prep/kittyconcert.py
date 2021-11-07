"""
This file is a test of my own python skillz.
@author: louiskhub
"""

from cat import CAT

cat1 = CAT("Felix")
cat2 = CAT("Louis")

cat1._greet(cat2)
print("-------------")
cat2._greet(cat1)
print("-------------")
for i in cat1._generate(10):
    print(i)