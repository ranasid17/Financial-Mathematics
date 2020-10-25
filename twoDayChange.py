#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:54:37 2020
@author: Sid
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors
from scipy import stats
import matplotlib.ticker as plticker

# import csv file (include r prior to path to adjust for '/')
# adjust this path to wherever file is stored on your computer 
# SPY (ETF) and NFLX used here as example 
spyRaw_Data = pd.read_csv (r'/Users/Sid/Documents/Other/SPY.csv') 
nflxRaw_Data = pd.read_csv (r'/Users/Sid/Documents/Other/NFLX.csv') 

# Goals: 
    # 1) Determine relative price change of an input over 48h period (M-W)
    # 2) Better inform short term options investment by calculating 
    #       historic relative price changes for 1 year period 
    # Note: 48h price change period can be modified to any desired period by 
    #       changing line 82 (i % 5 should = whatever number of days desired)
# Inputs: 
    # 1) Functions that take in "rawData" should be feed csv file containing 
    #       trading dates, market open and close prices, and trade volume
    # 2) Methods were created with Yahoo Finance csv structure so likely YF
    #       will be easiest option for application 
class twoDayChange: 
    # constructor  
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
        # 1) rawData: unmodified csv file (YahooFinance format)
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
            if ( i % 5  == 4): # every 7 days 
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
        # declare empty array to hold calculated relative changes 
        change = np.zeros((len(final), 1))
        # iterate thru first array (both should have same size)
        for i in range(len(initial)): 
            # ignore days when no trades happened
            if initial[i] == 0 or final[i] == 0:
                continue 
            # relative change formula 
            else: 
                change[i] = ( final[i] - initial[i] ) / initial[i] * 100 
                
        return change 
    # Goal: 
        # 1) Process data for input into plotPDF function 
    # Input: 
        # 1) arr: array holding relative price changes of asset 
    # Output
        # 1) cleaned array containing no NaNs or INFs
    def clean(arr):
        output = arr[(np.isnan(arr) == False) & (np.isinf(arr) == False)]
        return output
    # Goal: 
        # 1) Calculate probabilities of given relative price change for period
        # 2) Plot histogram of observed probabilities 
        # #) Calculate PDF of relative price change via Gaussian KDE 
        # 4) Plot PDF on top of histogram 
    # Input: 
        # 1) arr: array of relative price changes for desired time period 
        # 2) Note: arr MUST be 'cleaned' via clean method
    # Output: 
        # 1) histogram of relative price changes of asset
        # 2) Estimated PDF of asset price change 
    def plotPDF(arr):     
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # plot histogram
        n, bins, patches = plt.hist(arr, bins = 30, density = True) 
        # color code histogram 
        fracs = n / arr.max() # normalize olor code by height (likelihood)
        norm = colors.Normalize(fracs.min(), fracs.max()) # norm fracts [0,1]
        # loop thru each bin and set colors (using viridis map)
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)
            
        # calculate kernel density estimator (with 2 separate techniques)
        kde1 = stats.gaussian_kde(arr) 
        kde2 = stats.gaussian_kde(arr, bw_method='silverman')
        # plot KDE
        arr_eval = np.linspace(-10, 10, num=200)
        ax.plot(arr_eval, kde1(arr_eval), 'k-', label="Scott's Rule")
        ax.plot(arr_eval, kde2(arr_eval), 'r-', label="Silverman's Rule")
        # set tick frequency 
        loc = plticker.MultipleLocator(base=2.0) 
        ax.xaxis.set_major_locator(loc)
        # x, y, and plot titles
        plt.xlabel('Percent (Relative) Price Change')
        plt.ylabel('Probability')
        plt.title('48h Relative Price Change of Asset Oct 2019 - Oct 2020')
        # display plot 
        plt.show 
