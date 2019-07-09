# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:00:20 2019

@author: Dell
"""
# select function and press Ctrl+I for information
# no of input nodes=2
#no of nodes in hiddes layer=4
#no of nodes in output layer =1
#!/usr/bin/python

import numpy as np

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs) #4 no of inputs given at same time, it is not the number of input nodes
        self.li=len(self.inputs[0])  #2 no of input nodes 

        self.wi=np.random.random((self.li, self.l)) #generates a matrix of 2*4 random weights for input layer to hidden layer
        self.wh=np.random.random((self.l, 1))# generates a 4*1 matrix of random weights for hindden layer to output layer

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi)) # matrix multiplication of 4*2(inp) and 2*4(self.wi) matrix ,outputs a 4*4 matrix
        #see last page of copy 
        s2=sigmoid(np.dot(s1, self.wh)) 
        return s2
# backpropagation algorithm
    def train(self, inputs,outputs, it):
        for i in range(it):
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err=outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1))
            #updating weights
            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)

inputs=np.array([[0,0], [0,1], [1,0], [1,1] ])# they consider it as a 4*2 matrix
outputs=np.array([ [0], [1],[1],[0] ])

n=NN(inputs)
print(n.think(inputs))
n.train(inputs, outputs, 10000)
print(n.think(inputs))