# Financial Mathematics 

This repository containing various mathematical analysis programs created in order to better inform and predict stock market trading. These programs are primarily used to trade short-term options. 
1) 08/2019: perceptron.py 
    
    a) This is a 1 layer perceptron that accepts an N x N input matrix (X) the user defines, and a 1 x N output matrix (Y) which contains the correct output for each row of X. The perceptron attempts to predict the value of Y for each row. 
    
    b) For example, if passed X = [0 0 1; 1 0 1; 0 0 0] and Y = [1; 1; 0] the goal is for the 1-layer neural network to predict as close to Y as possible. Perceptron.py will produce a binary output bewteen [0,1] for each row in its 3x1 output that increases in accuracy as the number of training epochs increases. The 1-layer neural network is primarily intended to demonstrate how non-traditional machine learning can predict quantitative data.  
2) 10/22/2020: twoDayChange.py
    
    a) This is a class that passed CSV of a security's historic data, calculates the relative price change of that instrument's performance over a sliding window (ex: Monday-Wednesday of each week for the past year). It then calculates the probability, using kernel density estimation (KDE), of the relative prices changes over each window and plots the result. The underlying probability density function (PDF) estimated by KDE theoretically provides greater accuracy and insight into the security's price change than simply viewing the actual probabilities. 
    
    b) By knowing the underlying PDF, price change peaks and deviation, short term options can be traded around the security's price movement during sliding window. Documentation within the class indicates how to pass arguments and interpret outputs of each method. Each method should be used sequentially within class. 
4) 12/22/2020: simpleMovingAvg.py 
    
    a) This is a class that pulls historical data of a ticker from YahooFinance to calculate long- and short-term MAs of the security. It then plots both MAs alongside the underlying price. When the short term MA crosses over the long-term MA, this is a signal to buy the security (due to positive momentum). Conversely when short-term MA crosses under the long term MA, this is a signal to sell the security (due to negative momentum). S/LT MA windows can be modified from default setting (50/100). I prefer 20/50 ST/LT MAs because in my experience it provides the best balance between noise reduction of recent price changes in volatile markets while also allowing for some flexibility in weighting the most recent price changes. 
5) 01/06/2021: LSTM.py 

    a) This is a long short-term memory neural network that uses historical data from a security in order to predict the next day average price. LSTM neural networks are likely the best type of architecture to predict time series data due to their recurrent structure which allows for prior time values (in this case stock prices) to be refeed into the base layer as parametrized. The current iteration of this can predict next-day NASDAQ index (the top 100 stocks listed on NASDAQ) to about 96% accuracy. Running this model nightly would theoretically allow a trader to accurately asses risk and price the valuation of short term (next day expiry) contracts on any security that tracks the NASDAQ index.  
    
    
    b) LSTM.py also includes methods to predict next day price from simple MA and exponential MA models with the associated visualization and error calculation. I included these methods because the average run time to train LSTM.py is about 6h. Usually running this at night provides ample time to analyze the results before market open at 9:30a EST, and then place trades in the morning. However on nights when this is not possible, ST/LT MA analysis (similar to simpleMovingAverage.py) and its derivative, exponential moving average (EMA), may allow for accurate trades. 
6) 03/28/2021: WeeklyReview.py 

    a) Work in progress. 
    
