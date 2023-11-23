#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill
#This ones bad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def breakout_strat(commodity_data_df, price_col, atr_window=14, avg_window = 252):
        
    breakout_df = commodity_data_df.copy()
    breakout_df['ATR'] = breakout_df[price_col].diff().abs().rolling(window=atr_window).mean()
    breakout_df['Upper Channel'] = breakout_df[price_col].rolling(window=350).mean() + 7 * breakout_df['ATR']
    breakout_df['Lower Channel'] = breakout_df[price_col].rolling(window=350).mean() - 3 * breakout_df['ATR']

    breakout_df['Position'] = np.where(breakout_df[price_col]  >  breakout_df['Upper Channel'], 1, 0)    
    breakout_df['Position'] = np.where(breakout_df[price_col]  <  breakout_df['Lower Channel'], -1, breakout_df['Position'])    

    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = breakout_df['Date']
    clean_return_df[price_col] = breakout_df[price_col]
    clean_return_df['Position'] = breakout_df['Position']
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
        strat_df = breakout_strat(commodity_df, commodities[i])
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
