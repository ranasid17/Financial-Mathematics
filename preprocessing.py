#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:14:40 2021

@author: Sid
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import yfinance as yf

## Data preprocessing pipeline
## Goals: 
    # 1) Import AirTable spreadsheet
    # 2) Clean data for analysis 

# import weekly fund valuation csv 
df = pd.read_csv('/Users/Sid/Documents/Finance/fundMoney.csv') 
# remove '$', '%' from all columns
df['Gain/Loss'] = df['Gain/Loss'].str.replace('$', '').astype(float)
df['Starting Balance'] = df['Starting Balance'].str.replace('$', '').astype(float)
df['Funds in Play'] = df['Funds in Play'].str.replace('$', '').astype(float)
df['Friday Return'] = df['Friday Return'].str.replace('$', '').astype(float)
df['EOW Total'] = df['EOW Total'].str.replace('$', '').astype(float)
df['50% Commission'] = df['50% Commission'].str.replace('$', '').astype(float)
df['ROI'] = df['ROI'].str.replace('%', '').astype(float)

# create pd.Series to hold trading weeks 
tradingWeeks = df['Trading Week']
tradingWeeks = tradingWeeks.drop_duplicates()

# create mask for each trading week, find EOW P/L for each week, store in variable
# must add column for each new week
weeklyRevenue = np.zeros((1,len(tradingWeeks)-1))
weeklyRevenue[0,0] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='1/3/2021', 0).sum()
weeklyRevenue[0,1] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='1/10/2021', 0).sum()
weeklyRevenue[0,2] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='1/10/2021', 0).sum()
weeklyRevenue[0,3] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='1/17/2021', 0).sum()
weeklyRevenue[0,4] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='1/24/2021', 0).sum()
weeklyRevenue[0,5] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='2/7/2021', 0).sum()
weeklyRevenue[0,6] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='2/14/2021', 0).sum()
weeklyRevenue[0,7] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='2/21/2021', 0).sum()
weeklyRevenue[0,8] = df[['Gain/Loss']].sum(axis=1).where(df['Trading Week']=='2/28/2021', 0).sum()
# sum running P/L 
cumulativeRevenue = np.cumsum(weeklyRevenue)

# repeat lines 33-46 for ROI colum 
weeklyROI = np.zeros((1,len(tradingWeeks)-1))
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='1/3/2021', 0).mean()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='1/10/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='1/17/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='1/24/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='1/31/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='2/7/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='2/14/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='2/21/2021', 0).sum()
weeklyROI[0,0] = df[['ROI']].sum(axis=1).where(df['Trading Week']=='2/28/2021', 0).sum()
# sum running P/L 
cumulativeROI = np.cumsum(weeklyROI)