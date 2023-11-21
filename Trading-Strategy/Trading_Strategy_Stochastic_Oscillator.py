#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def stochastic_oscillator_strat(commodity_data_df, price_col, overbought_threshold=80, oversold_threshold=20, k = 14, d = 3):
    """The stochastic oscillator is range-bound, meaning it is always between 0 and 100. 
       This makes it a useful indicator of overbought and oversold conditions.
       %K is the fast stochastic oscillator, which presents the location of the closing price 
       of a stock in relation to the high and low prices of the stock over a period of time.
       %D is the slow stoch oscillator, the three day moving average of %K, smooths %K
       Area above 80 indicates an overbought region, while the area below 20 is considered an oversold 
       region. Sell signal is given when the oscillator is above 80 and then crosses back below 80. 
       Buy signal is given when the oscillator is below 20 and then crosses back above 20. 
       80 and 20 are the most common levels used but can be adjusted as needed.
       Sell signal occurs when a decreasing %K line crosses below the %D line in the overbought region
       Buy signal occurs when an increasing %K line crosses above the %D line in the oversold region
       returns oscillator_df, origional data plus new columns for %K, %D, Position, market return, and
       strategy return"""
       
    oscillator_df = commodity_data_df.copy()
    oscillator_df['%K'] = ((oscillator_df[price_col] - oscillator_df[price_col].rolling(k).min()) / (oscillator_df[price_col].rolling(k).max() - oscillator_df[price_col].rolling(k).min())) * 100
    oscillator_df['%D'] = oscillator_df['%K'].rolling(d).mean()
    oscillator_df['Position'] = np.where((oscillator_df['%K'] > oversold_threshold) & (oscillator_df['%K'].shift(1) < oversold_threshold), 1, 0)
    oscillator_df['Position'] = np.where((oscillator_df['%K'] < oversold_threshold) & (oscillator_df['%D'] < oversold_threshold) & (oscillator_df['%K'] > oscillator_df['%D']) & (oscillator_df['%K'] > oscillator_df['%K'].shift(1)), 1, oscillator_df['Position'])
    oscillator_df['Position'] = np.where((oscillator_df['%K'] < overbought_threshold) & (oscillator_df['%K'].shift(1) > overbought_threshold), -1, oscillator_df['Position'])
    oscillator_df['Position'] = np.where((oscillator_df['%K'] > overbought_threshold) & (oscillator_df['%D'] > overbought_threshold) & (oscillator_df['%K'] < oscillator_df['%D']) & (oscillator_df['%K'] < oscillator_df['%K'].shift(1)), -1, oscillator_df['Position'])
    
    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = oscillator_df['Date']
    clean_return_df[price_col] = oscillator_df[price_col]
    clean_return_df['Position'] = oscillator_df['Position']
    
    return clean_return_df


# Use the __main__ section for test cases. 
# This section will automatically be executed when the file is run in Python
if __name__ == "__main__":
    """For Testing Purposes Only, remove before adding to final project file"""
    
    filename = 'Testing_Future_Data.csv'
    commodity_data_master = pd.read_csv(filename)
    Soybean_df = pd.DataFrame()
    Soybean_df['Date'] = commodity_data_master['Date']
    Soybean_df['Soybean'] = commodity_data_master['Soybean']
    Soybean_df['Date'] = pd.to_datetime(Soybean_df['Date'])
    
    stoch_osc_df = stochastic_oscillator_strat(Soybean_df, 'Soybean')