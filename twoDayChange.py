#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:54:37 2020
@author: Sid
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

# import csv file (include r prior to path to adjust for '/')
# modify this path to wherever your data set is located on your computer
spyRaw_Data = pd.read_csv (r'/Users/Sid/Documents/Other/SPY.csv') 
# save relevant columns from csv file 
# KEY NOTE: This assumes data is structured from Yahoo Finance publical dBase
spyDates = pd.DataFrame(spyRaw_Data, columns = ['Date'])
spyOpenPrice = pd.DataFrame(spyRaw_Data, columns = ['Open'])
spyClosePrice = pd.DataFrame(spyRaw_Data, columns = ['Close'])
spyVolume = pd.DataFrame(spyRaw_Data, columns = ['Volume'])
# convert pandas dataframe to numpy arrays 
dates = spyDates.to_numpy()
openPrices = spyOpenPrice.to_numpy()
closePrices = spyClosePrice.to_numpy()
tradeVolume = spyVolume.to_numpy()

# Goal:
#   1) Extract open and close prices of input financial instrument
#   2) Open prices should be for each Monday (Oct 19, 2019 - Oct 19, 2020) 
#   3) Close prices should be for each Wednesday (48h after open price) 
# Args:
#   1) openPrices: array containing market open prices
#   2) closePrices: array containign market close prices 
# Output: 
#   1) arrays containing mkt open and close instrument values 
def getPrices(openPrices = [], closePrices = []): 
    mondayOpen = np.zeros((52, 1)) # declare empty array to hold monday prices 
    wednesdayClose = np.zeros((52,1)) # declare empty array to hold 48h close 
    count = 0
    # iterate thru first input array 
    for i in range(len(openPrices)):
        if ( i % 5  == 0): # every 7 days 
            mondayOpen[count] = openPrices[i] # save open price 
            count = count+1 # increment counter 
    # iterate thru second input array 
    count = 0 # reset count var
    for i in range(len(closePrices)):
        if (i % 5 == 2): # every 7 days (48h after monday)
            wednesdayClose[count] = closePrices[i]
            count = count+1
    
    return mondayOpen, wednesdayClose
# Goal: 
#   1) Calculate relative change from mkt open (Monday) to mkt close (48h later) 
# Args: 
#   1) initial: array containing mkt open values 
#   2) final: array containing mkt close values 
# Output: 
#   1) relative change in instrument price for each 48h trading period
def relativeChange(initial = [], final =[]): 
    change = np.zeros((len(final), 1)) 
    for i in range(len(initial)): 
        if initial[i] == 0 or final[i] == 0:
            continue 
        else: 
            change[i] = ( final[i] - initial[i] ) / initial[i] * 100 
            
    return change 
