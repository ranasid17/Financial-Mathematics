# Financial Mathematics 

1) Repository containing various programs created in order to better inform and predict stock market trading 
2) 08/2019: perceptron.py 
    This is a 1 layer perceptron that accepts an N x N input matrix (X) the user defines, and a 1 x N output matrix (Y)
    which contains the correct output for each row of X. The perceptron attempts to predict the value of Y for each row.
    Ex: X = [0 0 1; 1 0 1; 0 0 0], Y = [1; 1; 0]. Perceptron.py will produce a binary output within [0,1] that increases
    in accuracy as the number of training sesssions increases. 
3) 10/22/2020: twoDayChange.py
    This is a class that given a CSV of a security's historic data, calculates the relative price change 
    of that instrument's performance over a sliding window (ex: Monday-Wednesday of each week for the past year).
    It then calculates the probability of the relative prices changes over each window and plots the result. 
    twoDayChange.py also estimates the underlying probability density function (PDF) via kernal density estimation (KDE). 
    By knowing the underlying PDF, price change peaks and deviation, short term options can be traded around the 
    security's price movement during sliding window. Documentation within the class indicates how to pass 
    arguments and interpret outputs of each method. Each method should be used sequentially within class. 
4) 12/22/2020: simpleMovingAvg.py 
    This is a class that pulls historical data of a ticker from YahooFinance to calculate long and short 
    term MAs of the security. It then plots both MAs alongside the underlying price. When the short term MA crosses 
    over the long term MA, this is a signal to buy the security (due to positive momentum). Conversely when short 
    term MA crosses under the long term MA, this is a signal to sell the security (due to negative momentum). S/LT MA 
    windows can be modified from default setting (50/100) 
5) 01/06/2021: LSTM.py 
    This is a long short-term memory neural network that uses historical data from a security in order to predict
    the next day average price. It also includes methods to predict next day price from simple MA and exponential MA
    models with the associated visualization and error calculation. 
6) 03/28/2021: WeeklyReview.py 
    
