#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill
#This ones bad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def rsi_strat(commodity_data_df, price_col, window=14, overbought_threshold=70, oversold_threshold=30):
        
    rsi_df = commodity_data_df.copy()
    rsi_df['Gain'] = np.where(rsi_df[price_col].diff() > 0, rsi_df[price_col].diff(), 0)
    rsi_df['Loss'] = np.where(rsi_df[price_col].diff() < 0, rsi_df[price_col].diff(), 0)
    rsi_df['Avg Gain'] = rsi_df['Gain'].rolling(window=window, min_periods=1).mean()
    rsi_df['Avg Loss'] = rsi_df['Loss'].rolling(window=window, min_periods=1).mean()
    rsi_df['RSI'] = 100 - (100 / (1 + (rsi_df['Avg Gain']/rsi_df['Avg Loss'])))
    rsi_df['Position'] = np.where((rsi_df['RSI'] < oversold_threshold), 1, 0)
    rsi_df['Position'] = np.where((rsi_df['RSI'] > overbought_threshold), -1, rsi_df['Position'])

    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = rsi_df['Date']
    clean_return_df[price_col] = rsi_df[price_col]
    clean_return_df['Position'] = rsi_df['Position']
    return clean_return_df


# Use the __main__ section for test cases. 
# This section will automatically be executed when the file is run in Python
if __name__ == "__main__":
    """For Testing Purposes Only, remove before adding to final project file"""
    
    filename = 'Example_Cleaned_Data.csv'
    commodity_data_master = pd.read_csv(filename)
    commodities = ['Crude Oil', 'Brent Crude', 'Natural Gas','Gasoline','Heating Oil','Gasoil','Carbon',
                  'Gold','Silver','Copper','LME Copper','Aluminum','Zinc','Tin','Corn','Wheat','Oats',
                  'Soybeans','Soybeans SM']
    
    output_df = pd.DataFrame()
    output_df['Date'] = commodity_data_master['Date']
    for i in range(0,len(commodities)):
        commodity_df = pd.DataFrame()
        commodity_df['Date'] = commodity_data_master['Date']
        commodity_df[commodities[i]] = commodity_data_master[commodities[i]]
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        strat_df = rsi_strat(commodity_df, commodities[i])
        strat_df['Market Return'] = (strat_df[commodities[i]] - strat_df[commodities[i]].shift(1))/strat_df[commodities[i]].shift(1)
        strat_df['Position Scaled Return'] = 1+(((strat_df[commodities[i]].shift(-1) - strat_df[commodities[i]])/strat_df[commodities[i]]) * strat_df['Position'])
        strat_df['Strategy Total Return'] = strat_df['Position Scaled Return'].cumprod()-1
        output_df[commodities[i] + 'Market Return'] = strat_df['Market Return']
        output_df[commodities[i] + 'Position'] = strat_df['Position']
        output_df[commodities[i] + 'Scaled Return'] = strat_df['Position Scaled Return']
        output_df[commodities[i] + 'Strat Total Return'] = strat_df['Strategy Total Return']
        plt.figure(i)    
        plt.plot(strat_df['Date'], strat_df['Strategy Total Return'], label= commodities[i])
        plt.axhline(y = 0, color = 'r', linestyle = '-')
        plt.legend()
    
    #output_df.to_csv('output_data.csv', index=False)
