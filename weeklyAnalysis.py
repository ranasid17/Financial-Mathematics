#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:46:35 2021

@author: Sid
"""

import tradingAnalysisMethods as tam
    
if __name__ == "__main__":
    
    """ 1) Load and initialize data"""
    # import weekly trading data as data frame object 
    df = tam.preprocessing.clean_df() 
    # extract trading weeks 
    weeks = tam.preprocessing.extract_trading_weeks(df) 

    """ 2) Analyze weekly trading returns"""
    # collect weekly returns, cumulative returns, and adjusted for fees
    rev, cum_rev, adj_cum_rev = tam.trading_analysis.revenue(df, weeks)
    # repeat above for weekly, cumulative, and adjusted ROI (percentage)
    ROI, cum_ROI, adj_cum_ROI = tam.trading_analysis.ROI(df, weeks)
    # collect weekly balance (amt money available for trades)
    balance = tam.trading_analysis.balance(df, weeks)
    
    """ 3) Compare to benchmark: SPY """
    # collect daily open and close prices for SPY 
    spy_opens, spy_closes = tam.spy_analysis.extract_prices("SPY")
    # get list of YTD mondays, fridays to define start/stop of each trading wk 
    YTD_mondays, YTD_fridays = tam.preprocessing.pull_weekdays(2021, weeks)
    # truncate spy_opens, spy_closes to just Monday, Friday prices
    spy_opens, spy_closes = tam.spy_analysis.get_monday_friday_prices(
        spy_opens, spy_closes, YTD_mondays, YTD_fridays) 
    # calculate weekly and cumulative SPY returns and ROI
    rev_spy, cum_rev_spy, ROI_spy, cum_ROI_spy = tam.spy_analysis.weekly_value(
        spy_opens, spy_closes, weeks, YTD_mondays, YTD_fridays)
    
    
    """ 4) Plot results"""
    # plot weekly and cumulative returns 
    tam.plots.weekly_values(weeks, rev, cum_rev)
    # plot weekly and cumulative ROI 
    tam.plots.weekly_values(weeks, ROI, cum_ROI)
    # plot weekly and cumulative fund ROI vs SPY ROI 
    tam.plots.spy_comparison(weeks, ROI, cum_ROI, ROI_spy, cum_ROI_spy)


