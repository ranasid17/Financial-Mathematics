#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:54:37 2020

@author: Sid
"""

import pandas as pd
import numpy as np 

# import csv file (include r prior to path to adjust for '/')
spyRaw_Data = pd.read_csv (r'/Users/Sid/Documents/Other/SPY.csv') 

class twoDayChange: 
    # constructor function 
    def __init__ (self, rawData):
        self.input = rawData # should be raw YahooFinance csv 
    # Goals: 
        # 1) Extract relevant columns from csv file containing trading info 
        # 2) Convert dataframe columsn into numpy arrays 
    # Inputs: 
        # 1) rawData: unmodified csv file (YahooFinance format)
    # Outputs: 
        # 1) dates: Dates of each trading day (Oct 2019 - Oct 2020)
        # 2) openPrices: mkt open value of financial instrument 
        # 3) closePrices: mkt close value of financial instrument (48h later)
        # 4) tradeVaolume: amount of trades conducted on each date 
    def extractCols(rawData): 
        # extract columns from full dataframe
        yearDates = pd.DataFrame(rawData, columns = ['Date'])
        yearOpens = pd.DataFrame(rawData, columns = ['Open'])
        yearClose = pd.DataFrame(rawData, columns = ['Close'])
        yearVolume = pd.DataFrame(rawData, columns = ['Volume'])
        # convert df to numpy arrays 
        dates = yearDates.to_numpy()
        openPrices = yearOpens.to_numpy()
        closePrices = yearClose.to_numpy()
        tradeVolume = yearVolume.to_numpy()
        
        return dates, openPrices, closePrices, tradeVolume 
    # Goals: 
        # 1) Return market open values for each Monday (Oct 2019 - Oct 2020)
        # 2) Return market close values for each Wednesday in same time frame 
    # Inputs: 
        # 1) arrOpen: array containing mkt open prices
        # 2) arrClose: array containign mkt close prices (48h later)
    # Outputs: 
        # 1) mondayOpen: array containing mkt open prices for given timeframe 
        # 2) wednesdayClose: array containing mkt close prices (48h later)
    def getPrices(rawData): 
        # Extract open and close prices from raw data set 
        yearOpens = pd.DataFrame(rawData, columns = ['Open'])
        yearClose = pd.DataFrame(rawData, columns = ['Close'])
        # convert pandas dataframe to numpy array
        openPrices = yearOpens.to_numpy()
        closePrices = yearClose.to_numpy()
        # declare empty array to hold monday/wednesday open/close prices 
        mondayOpen = np.zeros((52, 1)) 
        wednesdayClose = np.zeros((52,1)) 
        # declare counter
        counter = 0
        # iterate thru first array 
        for i in range(len(openPrices)):
            if ( i % 5  == 0): # every 7 days 
                mondayOpen[counter] = openPrices[i] # save open price 
                counter = counter+1 # increment counter 
        # iterate thru second array, reset counter 
        counter = 0 
        for i in range(len(closePrices)):
            if (i % 5 == 2): # every 7 days (48h after monday)
                wednesdayClose[counter] = closePrices[i] # save close price
                counter = counter+1
        
        return mondayOpen, wednesdayClose
    # Goal: 
        # 1) Calc relative price change from Monday open to Wednesday Close
    # Input: 
        # 1) initial: array containing mkt open prices 
        # 2) final: array containing 48h later mkt close prices 
    # Output: 
        # 1) change: array of relative price changes for each 48h period 
        # 2) Note: Returns % change (already multiplied by 100)
    def relativeChange(initial = [], final =[]): 
        change = np.zeros((len(final), 1))
        for i in range(len(initial)): 
            if initial[i] == 0 or final[i] == 0:
                continue 
            else: 
                change[i] = ( final[i] - initial[i] ) / initial[i] * 100 
                
        return change 

