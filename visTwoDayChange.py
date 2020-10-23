#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:31:20 2020

@author: Sid
"""
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import colors

def plotHistogram(arr): 
    # actual histogram command
    n, bins, patches = plt.hist(arr, bins = 20, density = True) 
    # ncolor code by height (AKA: likelihood)
    fracs = n / arr.max()
    # normalize fracts from [0,1]
    norm = colors.Normalize(fracs.min(), fracs.max())
    # loop thru each bin and set colors (using viridis map)
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    # set x ticks to be +/- 1 from input min/max
    plt.xticks(np.arange(min(arr), max(arr), 2.0))
    # x, y, and plot titles
    plt.xlabel('Relative Percent Price Change')
    plt.ylabel('Observed Probability')
    plt.title('48h Relative Price Change of SPY Oct 2019 - Oct 2020')
    # display plot 
    plt.show 
    
