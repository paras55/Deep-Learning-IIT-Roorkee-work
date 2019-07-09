# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:23:03 2019

@author: Dell
"""

import numpy 
def NN(m1,w1,b):
    z=m1*w1+b
    if sigmoid(z)<0.5:
        return 0
    else:
        return 1
def sigmoid(x):
    return 1/(1+numpy.exp(-x))
w1=-20
b=10
NN(0,w1,b)