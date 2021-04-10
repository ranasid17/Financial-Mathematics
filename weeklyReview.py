#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:14:40 2021

@author: Sid
"""
import pandas as pd 
import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt 


class preprocessing: 
    def clean_df(): 
        # import weekly fund valuation csv 
        dataframe = pd.read_csv('/Users/Sid/Documents/Finance/Fund Returns/Money Tracking-Grid view.csv') 
        # remove '$' and '%' from all columns
        dataframe['Gain/Loss'] = dataframe['Gain/Loss'].str.replace('$', '').astype(float)
        dataframe['Starting Balance'] = dataframe['Starting Balance'].str.replace('$', '').astype(float)
        dataframe['Funds in Play'] = dataframe['Funds in Play'].str.replace('$', '').astype(float)
        dataframe['Friday Return'] = dataframe['Friday Return'].str.replace('$', '').astype(float)
        dataframe['EOW Total'] = dataframe['EOW Total'].str.replace('$', '').astype(float)
        dataframe['50% Commission'] = dataframe['50% Commission'].str.replace('$', '').astype(float)
        dataframe['ROI'] = dataframe['ROI'].str.replace('%', '').astype(float)
        
        return dataframe
        
    
    def extract_trading_weeks(dataframe): 
        # extract trading weeks from df as pd.Series
        trading_weeks = dataframe['Trading Week']
        # remove duplicate trading weeks (want only 1 of each incidence)
        trading_weeks = trading_weeks.drop_duplicates()
        # reset series index to [0, n]
        trading_weeks = trading_weeks.reset_index(drop=True)
        
        return trading_weeks 
    
    
    def pull_weekdays(year, trading_weeks):
        # pull all sundays, mondays, fridays in 2021 and convert to string
        mondays = pd.date_range(start=str(year), end=str(year+1), freq='W-MON').strftime("%-m/%-d/%Y")
        fridays = pd.date_range(start=str('1/04/2021'), end=str(year+1), freq='W-FRI').strftime("%-m/%-d/%Y")
        # truncate list to only YTD mondays and convert to np array
        mondays = np.array(mondays[0:len(trading_weeks)], dtype=str)
        fridays = np.array(fridays[0:len(trading_weeks)], dtype=str) 

        # iter thru Mondays to check for no-trading days 
        for i in range(len(trading_weeks)): 
            if (mondays[i] == '1/18/2021'): # check for MLK Jr Day 
                mondays[i] = '1/19/2021' # replace w following Tuesday 
            if (mondays[i] == '2/15/2021'): # check for Pres' Day 
                mondays[i] = '2/16/2021'# replace w following Tuesday 
            # reformat to remove 0 padding (%-d does not work earlier in func)
            mondays[i].replace("/0", "/")
            fridays[i].replace("/0", "/")
       
        return mondays, fridays 
    

class trading_analysis: 
    def revenue(dataframe, trading_weeks): 
        # create mask for each trading week, find EOW P/L for each week
        weekly_revenue = np.zeros((len(trading_weeks),1))
        # iterate thru all trading weeks 
        for i in range(len(trading_weeks)): 
            weekly_revenue[i] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']==trading_weeks[i],0).sum()            
        
        # sum running P/L 
        cumulative_revenue = np.cumsum(weekly_revenue)
        adj_cumulative_revenue = cumulative_revenue / 2 
        
        # return running revenue, 
        return weekly_revenue, cumulative_revenue, adj_cumulative_revenue

    
    def ROI(dataframe, trading_weeks): 
        # create mask for each trading week, find EOW ROI (as percentage) 
        weekly_ROI = np.zeros((len(trading_weeks),1))
        # iterate thru all trading weeks
        for i in range(len(trading_weeks)): 
            weekly_ROI[i] = dataframe.loc[dataframe['Trading Week']==trading_weeks[i],'ROI'].mean()
        
        # sum running P/L 
        cumulative_ROI = pd.Series(np.cumsum(weekly_ROI))
        adj_cumulative_ROI = cumulative_ROI / 2 
        
        # return running revenue, 
        return weekly_ROI, cumulative_ROI, adj_cumulative_ROI
    
    
    def balance(dataframe, trading_weeks): 
        # create mask for each trading week, find SOW balance 
        weekly_balance = np.zeros((len(trading_weeks),1))
        # iterate thru all trading weeks
        for i in range(len(trading_weeks)): 
            weekly_balance[i] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']==trading_weeks[i],0).sum()

        return weekly_balance
    

class spy_analysis: 
    def extract_prices(input_ticker): 
        # convert input ticker to yf.Ticker object 
        ticker = yf.Ticker(input_ticker)
        # pull YTD ticker data and store in dataframe 
        df = ticker.history(period='ytd')
        # store open and close prices 
        open_price = df['Open']
        close_price = df['Close']

        return open_price, close_price
    
    
    def get_monday_friday_prices(open_prices, close_prices, mondays, fridays):
        # extract Monday/Friday open/close prices from full lists 
        monday_opens = np.asarray(open_prices.loc[mondays])
        friday_closes = np.asarray(close_prices.loc[fridays])
        
        return monday_opens, friday_closes
    
    
    def weekly_value(monday_opens, friday_closes, trading_weeks, mondays, fridays): 
        # initialize array to hold weekly valuation of SPY 
        weekly_change = np.zeros((len(trading_weeks),1))
        
        # calculate weekly change in valuation 
        weekly_change = friday_closes - monday_opens
        # find weekly YTD valuation change                 
        cumulative_value = np.cumsum(weekly_change)
        
        # find weekly ROI by dividing weekly value by M open price of that week
        # must transpose monday_opens to [1 x n] array to make division work 
        percent_weekly_ROI = weekly_change / monday_opens[0]
        percent_weekly_ROI = percent_weekly_ROI * 100 
        percent_weekly_ROI = percent_weekly_ROI.reshape(len(trading_weeks),1)
        # sum running percent weekly chagne to find cumulative percent change
        cumulative_ROI = np.cumsum(percent_weekly_ROI)
        
        # return running revenue, 
        return weekly_change, cumulative_value, percent_weekly_ROI, cumulative_ROI

    def create_dataframe(trading_weeks, monday_opens, close_prices, weekly_ROI, cumulative_ROI): 
        df = pd.DataFrame(trading_weeks)
        df['Monday Open Price'] = pd.Series(monday_opens)
        df['Friday Close Price'] = pd.Series(close_prices)
        df['Weekly ROI (%)'] = weekly_ROI
        df['Cumulative ROI (%)'] = pd.Series(cumulative_ROI)
        return df 
        
