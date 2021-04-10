#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:46:35 2021

@author: Sid
"""

import dataAnalysis as data
    
def main(): 
    """ 1) Load and initialize data"""
    # import weekly trading data as data frame object 
    df = data.preprocessing.clean_df() 
    # extract trading weeks 
    weeks = data.preprocessing.extract_trading_weeks(df) 
    # get list of YTD mondays, fridays to define start/stop of each trading wk 
    YTD_mondays, YTD_fridays = data.preprocessing.pull_weekdays(2021, weeks)
    
    """ 2) Analyze weekly trading returns"""
    # collect weekly returns, cumulative returns, and adjusted for fees
    rev, cum_rev, adj_cum_rev = data.trading_analysis.revenue(df, weeks)
    # repeat above for weekly, cumulative, and adjusted ROI (percentage)
    ROI, cum_ROI, adj_cum_ROI = data.trading_analysis.revenue(df, weeks)
    # collect weekly balance (amt money available for trades)
    balance = data.trading_analysis.revenue(df, weeks)
    
    """ 3) Compare to benchmark: SPY """
    # collect daily open and close prices for SPY 
    spy_opens, spy_closes = data.spy_analysis.extract_prices("SPY")
    # truncate spy_opens, spy_closes to just Monday, Friday prices
    spy_opens, spy_close = data.spy_analysis.get_monday_friday_prices(
        spy_opens, spy_closes, YTD_mondays, YTD_fridays) 
    
if __name__ == "__main__":
    main()

