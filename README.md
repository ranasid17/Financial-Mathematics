# Financial Mathematics 

1) Repository containing various programs created in order to better inform and predict stock market trading 
2) 8/2019: perceptron is a 1 layer perceptron that given an input matrix (X) and the correct output (Y), attempts 
    to predict whether the binary output should be [0,1] 
3) 10/22: twoDayChange is a class that given a CSV of a security's historic data, calculates the relative price change 
    of that instrument's performance over a sliding widnwo (ex: Monday-Wednesday of each week for the past year) and 
    then plots the probability of the relative prices changes. Two estimations of of the underlying probability density 
    function via kernal density estimation (KDE). By knowing the underlying price change peaks and variation, short term 
    options can be traded around the security's price movement. Documentation within the class indicates how to pass 
    arguments and interpret outputs of each method. Each method should be used sequentially within class. 
4) 12/22: simpleMovingAvg is a class that pulls historical data of a ticker from YF to calculate long and short 
    term MAs of the security. It then plots both on top of the underlying price. When the short term MA crosses 
    over the long term MA, it is a signal to buy the security (due to positive momentum). Conversely when short 
    term MA crosses under the long term MA, it is a signal to sell the security (due to negative momentum). S/LT MA 
    windows can be modified from default setting (50/100) 
5) 01/06: LSTM is a long short-term memory neural network that uses historical data from a security in order to predict
    the next day average price. It also includes methods to predict next day price from simple MA and exponential MA
    models with the associated visualization and error calculation. 
    

