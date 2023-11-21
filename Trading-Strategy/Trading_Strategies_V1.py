#MF703 Final Project
#Trading Strategies V1
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def create_bollinger_bands(commodity_data_df, price_col, window_size = 21, num_of_std = 2):
    """commodity_data, a df of commodities data,
       price_col, contains the column name for the commodity prices
       window, the number of days to use in creating the rolling mean and standard deviation.
       no_of_std, the number of standard deviations to use in calculating the Bollinger bands.
       The function returns a df with the same information as the provided plus the following 
       columns: ['Rolling Mean', 'Upper Band', 'Lower Band']"""
       
    copy_df = commodity_data_df.copy()
    copy_df['Rolling Mean'] = copy_df[price_col].rolling(window_size).mean()    
    copy_df['Upper Band'] = copy_df['Rolling Mean'] + (num_of_std * copy_df[price_col].rolling(window_size).std())
    copy_df['Lower Band'] = copy_df['Rolling Mean'] - (num_of_std * copy_df[price_col].rolling(window_size).std())
        
    return copy_df


def bollinger_band_strat(commodity_data_df, price_col, window_size = 21, num_of_std = 2):
    """This function will evaluate the data elements in the price column against the Bollinger 
       Bands in the columns UpperBound and LowerBound. The function will apply a long/short strategy,
       i.e., create a long position (+1) when the Observation crosses below the LowerBound, and create
       a short position (-1) when the Observation crosses above the UpperBound. A position of 0 means
       there is no signal, position holds. The function will returnthe same df provided but 
       containing additional columns Position, Market Return, and Strategy Return."""
    
    bbands_df = create_bollinger_bands(commodity_data_df, price_col, window_size, num_of_std)
    bbands_df['Position'] = np.where(bbands_df[price_col] < bbands_df['Lower Band'], 1, 0)
    bbands_df['Position'] = np.where(bbands_df[price_col] > bbands_df['Upper Band'], -1, bbands_df['Position'])
    
    temp_df = bbands_df[price_col].shift(1)
    bbands_df['Market Return'] = (bbands_df[price_col] - temp_df)/temp_df
    bbands_df['Strategy Return'] = ((bbands_df[price_col].shift(-1) - bbands_df[price_col])/bbands_df[price_col]) * bbands_df['Position']
       
    return bbands_df


def create_stochastic_oscillator(commodity_data_df, price_col, k = 14, d = 3):
    """The stochastic oscillator is range-bound, meaning it is always between 0 and 100. 
       This makes it a useful indicator of overbought and oversold conditions.
       %K is the fast stochastic oscillator, which presents the location of the closing price 
       of a stock in relation to the high and low prices of the stock over a period of time.
       %D is the slow stoch oscillatyor, the three day moving average of %K, smooths %K
       returns Copy_df, origional data plus two new columns for %K and %D"""
       
    copy_df = commodity_data_df.copy()
    copy_df['%K'] = ((copy_df[price_col] - copy_df[price_col].rolling(k).min()) / (copy_df[price_col].rolling(k).max() - copy_df[price_col].rolling(k).min())) * 100
    copy_df['%D'] = copy_df['%K'].rolling(d).mean()
    
    return copy_df


def stochastic_oscillator_strat(commodity_data_df, price_col, overbought_threshold=80, oversold_threshold=20, k = 14, d = 3):
    """Area above 80 indicates an overbought region, while the area below 20 is considered an oversold 
       region. Sell signal is given when the oscillator is above 80 and then crosses back below 80. 
       Buy signal is given when the oscillator is below 20 and then crosses back above 20. 
       80 and 20 are the most common levels used but can be adjusted as needed.
       Sell signal occurs when a decreasing %K line crosses below the %D line in the overbought region
       Buy signal occurs when an increasing %K line crosses above the %D line in the oversold region
       returns oscillator_df, origional data plus new columns for %K, %D, Position, market return, and
       strategy return"""
    
    oscillator_df = create_stochastic_oscillator(commodity_data_df, price_col, k = 14, d = 3)
    oscillator_df['Position'] = np.where((oscillator_df['%K'] > oversold_threshold) & (oscillator_df['%K'].shift(1) < oversold_threshold), 1, 0)
    oscillator_df['Position'] = np.where((oscillator_df['%K'] < oversold_threshold) & (oscillator_df['%D'] < oversold_threshold) & (oscillator_df['%K'] > oscillator_df['%D']) & (oscillator_df['%K'] > oscillator_df['%K'].shift(1)), 1, oscillator_df['Position'])
    oscillator_df['Position'] = np.where((oscillator_df['%K'] < overbought_threshold) & (oscillator_df['%K'].shift(1) > overbought_threshold), -1, oscillator_df['Position'])
    oscillator_df['Position'] = np.where((oscillator_df['%K'] > overbought_threshold) & (oscillator_df['%D'] > overbought_threshold) & (oscillator_df['%K'] < oscillator_df['%D']) & (oscillator_df['%K'] < oscillator_df['%K'].shift(1)), -1, oscillator_df['Position'])
    
    
    temp_df = oscillator_df[price_col].shift(1)
    oscillator_df['Market Return'] = (oscillator_df[price_col] - temp_df)/temp_df
    #Can only execute our trade tommorow based of todays signal
    oscillator_df['Strategy Return'] = ((oscillator_df[price_col].shift(-1) - oscillator_df[price_col])/oscillator_df[price_col]) * oscillator_df['Position']

    return oscillator_df


def create_t12_t1_moving_avg(commodity_data_df, price_col, window_12M = 252, window_1M = 21):
    
    commodity_data_df['12M MA'] = commodity_data_df[price_col].rolling(window_12M).mean()
    commodity_data_df['1M MA'] = commodity_data_df[price_col].rolling(window_1M).mean()
    
    return commodity_data_df

def t12_t1_moving_avg_strat(commodity_data_df, price_col, window_12M = 252, window_1M = 21, percent_differnce = .1):    
    
    t12_t1_df = create_t12_t1_moving_avg(commodity_data_df, price_col, window_12M, window_1M)
    t12_t1_df['Position'] = np.where(t12_t1_df['1M MA'] > t12_t1_df['12M MA'], 1, 0)
    t12_t1_df['Position'] = np.where(t12_t1_df['1M MA'] < t12_t1_df['12M MA'], -1, t12_t1_df['Position'])
    #Another Option is to intorudce a minimum economically signifigant differennce between the averages
    #t12_t1_df['Position'] = np.where(t12_t1_df['1M MA'] > (t12_t1_df['12M MA'] * percent_differnce), 1, 0)
    #t12_t1_df['Position'] = np.where(t12_t1_df['1M MA'] < (t12_t1_df['12M MA'] * (1-percent_differnce)), -1, t12_t1_df['Position'])
    
    t12_t1_df['Market Return'] = (t12_t1_df[price_col] - t12_t1_df[price_col].shift(1))/t12_t1_df[price_col].shift(1)
    #Can only execute our trade tommorow based of todays signal
    t12_t1_df['Strategy Return'] = ((t12_t1_df[price_col].shift(-1) - t12_t1_df[price_col])/t12_t1_df[price_col]) * t12_t1_df['Position']
    
    return t12_t1_df
    
# Use the __main__ section for test cases. 
# This section will automatically be executed when the file is run in Python
if __name__ == "__main__":
    
    filename = 'Testing_Future_Data.csv'
    commodity_data_master = pd.read_csv(filename)
    Soybean_df = pd.DataFrame()
    Soybean_df['Date'] = commodity_data_master['Date']
    Soybean_df['Soybean'] = commodity_data_master['Soybean']
    bb_df = bollinger_band_strat(Soybean_df, 'Soybean')
    
    moving_averages_df = t12_t1_moving_avg_strat(Soybean_df, 'Soybean')
    plt.figure(1)
    plt.plot(moving_averages_df['Date'], moving_averages_df['12M MA'], label='12 Month')
    plt.plot(moving_averages_df['Date'], moving_averages_df['1M MA'], label='1 Month')
    plt.plot(moving_averages_df['Date'], moving_averages_df['Soybean'], label='Price')
    plt.legend()
    
    plt.figure(2)
    plt.plot(moving_averages_df['Date'], moving_averages_df['Market Return'].cumsum(), label='Market Return')
    plt.plot(moving_averages_df['Date'], moving_averages_df['Strategy Return'].cumsum(), label='Strategy Return')
    plt.legend()
    
    """plt.figure(1)
    plt.plot(bb_df['Date'], bb_df['Soybean'], label='Price')
    plt.plot(bb_df['Date'], bb_df['Lower Band'], label='Lower BB')
    plt.plot(bb_df['Date'], bb_df['Upper Band'], label='Upper BB')
    plt.legend()
    
    plt.figure(2)
    plt.plot(bb_df['Date'], bb_df['Market Return'].cumsum(), label='MKT Return')
    plt.plot(bb_df['Date'], bb_df['Strategy Return'].cumsum(), label='Strat Return')
    plt.legend()"""
    
    """stoch_osc_df = stochastic_oscillator_strat(Soybean_df, 'Soybean')
    plt.figure(3)
    plt.plot(stoch_osc_df['Date'], stoch_osc_df['Market Return'].cumsum(), label='MKT Return')
    plt.plot(stoch_osc_df['Date'], stoch_osc_df['Strategy Return'].cumsum(), label='Strat Return')
    plt.legend()"""