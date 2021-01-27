#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:56:10 2019
@author: Sid
"""

## Purpose: 
##  1) Build one layer perceptron
##  2) Predict whether 0 or 1 should be output of each 3x3 matrix row
##  3) Learn basics of machine learning

#import numpy package 
import numpy as np 

# define sigmoid function for feedforward layer 
# Use sigmoid because allows for gradient of values between 0 and 1 
def sigmoid(x):
    return (1 / (1+ np.exp(-x)))
# define derivative of sigmoid func for backpropogation layer 
# 1st derivative of sigmoid needed to calc partial derivs of weights
def sigmoid_derivative(x): 
    return (x * (1 - x))
## define neural network class
## This is where the "thinking" happens
class neuralNetwork:
    #Input layer
    #Purpose: define input, weight, and output vars
    def __init__(self, x, y): 
        self.input = x #input value
        self.w1 = np.random.rand(self.input.shape[1],4) # randomly assign init weight
        self.w2 = np.random.rand(4,1) # " " 
        self.y = y 
        self.output = np.zeros(self.y.shape) #output value 
    #Feedforward layer
    #Purpose: calculate predicted output 
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1)) # dot prod (layer 1) x (weight 1) 
        self.output = sigmoid(np.dot(self.layer1, self.w2)) # dot prod (output line 39) x (weight 2) 
    #Backpropogation layer 
    #Purpose: Provide feedback to modify w1, w2 for next iteration
    def backprop(self):
        #apply Chain Rule to Mean Squared Error w.r.t. weight 1 and weight 2
        #Note: we define MSE as equivalent to the "loss" or "cost" func
        d_w2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * 
            sigmoid_derivative(self.output)) )
        d_w1 = np.dot(self.input.T, np.dot((2 * (self.y - self.output) * 
            sigmoid_derivative(self.output)), self.w2.T) * sigmoid_derivative(self.layer1))
        #update w1, w2 w/ MSE deriv
        self.w1 += d_w1
        self.w2 += d_w2
        
if __name__ == "__main__":
    # input array we will feed into neuralNetwork, can be any size (3x4 in this case)
    # Note: User should have "rule" in mind for what output of each row will be
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [0,0,0]])
    # output array neuralNetwork will attempt to predict 
    # Rule: If there is a 1 within a row, then output should be 1
    y = np.array([[1],[1],[1],[0]]) 
    ann = neuralNetwork(x,y)
    ## number of iterations neuralNetwork has to "learn" the rule 
    for i in range(1000000):
        ann.feedforward()
        ann.backprop()
    ## print output 
    print(ann.output)
