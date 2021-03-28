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
import datetime
from datetime import date, timedelta


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
    

class weekly_fund_analysis: 
    def weekly_revenue(dataframe, trading_weeks): 
        # create mask for each trading week, find EOW P/L for each week, store in variable
        # must add column for each new week
        weekly_revenue = np.zeros((len(trading_weeks),1))
        weekly_revenue[0] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='1/3/2021', 0).sum()
        weekly_revenue[1] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='1/10/2021', 0).sum()
        weekly_revenue[2] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='1/17/2021', 0).sum()
        weekly_revenue[3] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='1/24/2021', 0).sum()
        weekly_revenue[4] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='1/31/2021', 0).sum()
        weekly_revenue[5] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='2/7/2021', 0).sum()
        weekly_revenue[6] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='2/14/2021', 0).sum()
        weekly_revenue[7] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='2/21/2021', 0).sum()
        weekly_revenue[8] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='2/28/2021', 0).sum()
        weekly_revenue[9] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='3/7/2021', 0).sum()
        weekly_revenue[10] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='3/14/2021', 0).sum()
        weekly_revenue[11] = dataframe[['Gain/Loss']].sum(axis=1).where(dataframe['Trading Week']=='3/21/2021', 0).sum()
        # sum running P/L 
        cumulative_revenue = np.cumsum(weekly_revenue)
        adj_cumulative_revenue = cumulative_revenue / 2 
        
        # return running revenue, 
        return weekly_revenue, cumulative_revenue, adj_cumulative_revenue

    
    def weekly_ROI(dataframe, trading_weeks): 
        # repeat lines 34-45 for ROI column of df (Note: These are percentages)
        # must add column for each new week
        weekly_ROI = np.zeros((len(trading_weeks),1))
        weekly_ROI[0] = dataframe.loc[dataframe['Trading Week'] == '1/3/2021','ROI'].mean() 
        weekly_ROI[1] = dataframe.loc[dataframe['Trading Week'] == '1/10/2021','ROI'].mean()
        weekly_ROI[2] = dataframe.loc[dataframe['Trading Week'] == '1/17/2021','ROI'].mean()
        weekly_ROI[3] = dataframe.loc[dataframe['Trading Week'] == '1/24/2021','ROI'].mean()
        weekly_ROI[4] = dataframe.loc[dataframe['Trading Week'] == '1/31/2021','ROI'].mean()
        weekly_ROI[5] = dataframe.loc[dataframe['Trading Week'] == '2/7/2021','ROI'].mean()
        weekly_ROI[6] = dataframe.loc[dataframe['Trading Week'] == '2/14/2021','ROI'].mean()
        weekly_ROI[7] = dataframe.loc[dataframe['Trading Week'] == '2/21/2021','ROI'].mean()
        weekly_ROI[8] = dataframe.loc[dataframe['Trading Week'] == '2/28/2021','ROI'].mean()
        weekly_ROI[9] = dataframe.loc[dataframe['Trading Week'] == '3/7/2021','ROI'].mean()
        weekly_ROI[10] = dataframe.loc[dataframe['Trading Week'] == '3/14/2021','ROI'].mean()
        weekly_ROI[11] = dataframe.loc[dataframe['Trading Week'] == '3/21/2021','ROI'].mean()
        # sum running P/L 
        cumulative_ROI = pd.Series(np.cumsum(weekly_ROI))
        adj_cumulative_ROI = cumulative_ROI / 2 
        
        # return running revenue, 
        return weekly_ROI, cumulative_ROI, adj_cumulative_ROI
    
    
    def weekly_balance(dataframe, trading_weeks): 
        # repeat lines 34-45 for fund valuation (Starting Balance) 
        # must add column for each new week
        weekly_balance = np.zeros((len(trading_weeks),1))
        weekly_balance[0] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='1/3/2021', 0).sum()
        weekly_balance[1] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='1/10/2021', 0).sum()
        weekly_balance[2] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='1/17/2021', 0).sum()
        weekly_balance[3] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='1/24/2021', 0).sum()
        weekly_balance[4] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='1/31/2021', 0).sum()
        weekly_balance[5] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='2/7/2021', 0).sum()
        weekly_balance[6] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='2/14/2021', 0).sum()
        weekly_balance[7] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='2/21/2021', 0).sum()
        weekly_balance[8] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='2/28/2021', 0).sum()
        weekly_balance[9] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='3/7/2021', 0).sum()
        weekly_balance[10] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='3/14/2021', 0).sum() 
        weekly_balance[11] = dataframe[['Starting Balance']].sum(axis=1).where(dataframe['Trading Week']=='3/21/2021', 0).sum() 
        
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
    
    
    def pull_weekdays(year, trading_weeks):
        # pull all mondays from 1/1/21 to 12/31/21 and convert to string
        mondays = pd.date_range(start=str(year), end=str(year+1), freq='W-MON').strftime("%Y-%m-%d")
        # truncate list to only YTD mondays and convert to pd.Series 
        mondays = pd.Series(mondays[0:len(trading_weeks)], dtype=str)
        # iter thru Mondays to check for no-trading days 
        for i in range(len(mondays)): 
            if (mondays[i] == '2021-01-18'): # check for MLK Jr Day 
                mondays[i] = '2021-01-19' # replace w following Tuesday 
            if (mondays[i] == '2021-02-15'): # check for Pres' Day 
                mondays[i] = '2021-02-16'# replace w following Tuesday 
        # pull all mondays from 1/1/21 to 12/31/21 and convert to string
        fridays = pd.date_range(start=str('2021-01-04'), end=str(year+1), freq='W-FRI').strftime("%Y-%m-%d")
        # truncate list to only YTD mondays and convert to pd.Series
        fridays = pd.Series(fridays[0:len(trading_weeks)], dtype=str) 
        
        return mondays, fridays 
    
    def get_monday_friday_prices(open_prices, close_prices, list_mondays, list_fridays):
        # extract Monday/Friday open/close prices from full lists 
        monday_opens = np.asarray(open_prices.loc[list_mondays])
        friday_closes = np.asarray(close_prices.loc[list_fridays])
        
        return monday_opens, friday_closes
    
    
    def weekly_value(monday_opens, friday_closes, trading_weeks, list_mondays, list_fridays): 
        # initialize array to hold weekly valuation of SPY 
        weekly_value_change = np.zeros((len(trading_weeks),1))
        
        # calculate weekly chagne in valuation 
        weekly_value_change = friday_closes - monday_opens
        # sum running P/L to find cumulative valuation 
        cumulative_value = np.cumsum(weekly_value_change)
        
        # find weekly ROI by dividing weekly value by M open price of that week
        # must transpose monday_opens to [1 x n] array to make division work 
        percent_weekly_ROI = weekly_value_change / monday_opens[0]
        percent_weekly_ROI = percent_weekly_ROI * 100 
        percent_weekly_ROI = percent_weekly_ROI.reshape(len(trading_weeks),1)
        # sum running percent weekly chagne to find cumulative percent change
        cumulative_ROI = np.cumsum(percent_weekly_ROI)
        
        # return running revenue, 
        return weekly_value_change, cumulative_value, percent_weekly_ROI, cumulative_ROI

    def create_dataframe(trading_weeks, monday_opens, close_prices, weekly_ROI, cumulative_ROI): 
        df = pd.DataFrame(trading_weeks)
        df['Monday Open Price'] = pd.Series(monday_opens)
        df['Friday Close Price'] = pd.Series(close_prices)
        df['Weekly ROI (%)'] = weekly_ROI
        df['Cumulative ROI (%)'] = pd.Series(cumulative_ROI)
        return df 
        

