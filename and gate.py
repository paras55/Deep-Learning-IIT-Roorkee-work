# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:56:20 2019

@author: Dell
"""

import numpy 
def NN(m1,m2,w1,w2,b):
    z=m1*w1+m2*w2+b
    if sigmoid(z)<0.5:
        return 0
    else:
        return 1
def sigmoid(x):
    return 1/(1+numpy.exp(-x))
w1=20
w2=20
b=-30
NN(1,0,w1,w2,b)