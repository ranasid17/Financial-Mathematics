#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:00:42 2020

@author: Sid
"""
# Necessary for importing SP500 from wiki
import bs4 as bs
import pickle
import requests
# Necessary for importing full SP500 data
import datetime as dt
import os
from pandas_datareader import data as pdr
# Necessary for visualiation 
import numpy as np 
# Necessary for MAs
import yfinance as yf
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def getSP500_Tickers():
    # store SP500 companies in list (from Wikipedia)
    lst = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    # convert list to Python BeautifulSoup object (for easier parsing)
    soup = bs.BeautifulSoup(lst.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]: # iteratre thru each row of table
        ticker = row.findAll('td')[0].text # grab text of 'table data' (ticker)
        ticker = ticker[:-1]
        tickers.append(ticker) # add current ticker to list 
    # save list 
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    
    return tickers

def getData(reload_sp500=False):
    if reload_sp500: # check if list of SP500 tickers is saved 
        tickers = getSP500_Tickers()
    else: # otherwise re-call prev function to re-obtain list 
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    start = dt.datetime(2019, 6, 8)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
            
def compileData():
    # load (saved) list of SP500 tickers
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    # initialize empty DF to hold values 
    df = pd.DataFrame()
    # iter thru list of tickers
    for count, ticker in enumerate(tickers):
        # load ticker csv
        tempDF = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        # set index to date 
        tempDF.set_index('Date', inplace=True)
        # extract only Adjusted Close prices from full csv 
        tempDF.rename(columns={'Adj Close': ticker}, inplace=True)
        # remove all other cols 
        tempDF.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        
        if df.empty: # set empty DF to temp iff program not run prior
            df = tempDF
        else: # otherwise concatenate 
            df = df.join(tempDF, how='outer')
        # progress meter 
        if count % 10 == 0:
            print(count) # confirm after every 10th stock 
    # convert df back to csv
    df.to_csv('sp500_adjClose.csv')

def analyze_visualize():
    ## Analysis 
    # read and store csv of full SP500
    df = pd.read_csv('sp500_adjClose.csv')
    
    df_corr = df.corr() # correlate all sp500 prices to e/o
    correlations = df_corr.values # store correlations 
    
    ## Visualizatoin
    fig,ax = plt.subplots() # create a fig object
    ax = fig.add_subplots(111) # build axis on figure 
    
    # LINE BELOW IS THE ACTUAL PLOT COMMAND
    heatmap = ax.pcolor(correlations, cmap=plt.cm.vlag) # plot heatmap
    # Figure set up 
    fig.colorbar(heatmap) # add color bar for scale 
    ax.set_xticks(np.arange(correlations.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(correlations.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis() # inverting Y promotes readability (how it "should" be)
    ax.xaxis.tick_top() # "                             " 
    
    column_labels = df_corr.columns # create array of Y labels 
    row_labels = df_corr.index # "                          "
    ax.set_yticklabels(row_labels) # set Y axis labels 
    ax.set_xticklabels(column_labels) # "                   " 
    
    #plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

def process_data_for_labels(ticker):
    # Goal: 
        # 1) Predict price 7 days into future 
        # 2) Do this by training each model on one company 
    # Input
        # 1) Stock desired for prediction
    hm_days = 7 
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0) # load SP500 csv 
    tickers = df.columns.values.tolist() # load list of SP500 tickers
    df.fillna(0, inplace=True) # handle NaNs
    
    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0, inplace=True)
    return tickers, df
