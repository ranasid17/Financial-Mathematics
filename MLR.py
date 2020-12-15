#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 22:29:42 2020

@author: Sid
"""

import numpy as np 
import pandas as pd 
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

# Goal: Confirm assumptions of MLR model 
# 1) Linear relationship between IVs and DV 
# 2) IVs are not highly correlated w each other (test via VIFs)
# 3) Residuals variance constant across IVs
# 4) Residuals are normally distributed 

# dict containing inputs 
data = {}
# Building MLR 
class MLR(): 
    # constructor 
    def __init__(self, rawData): 
        self.input = rawData
    def setVars(rawData): 
        df = pd.DataFrame(data=rawData) # convert input to pd.DataFrame
        X = df[['_____', '_______']] # change this for respective models
        X = sm.add_constant(X) # add Y intercept  
        Y = df[['deltaFM']] 
        return X, Y 
    def multipleRegression(X,Y): 
        model = sm.OLS(Y, X).fit() #
        predictions = model.predict(X) # predictions of the model
        model.summary()
        return predictions 