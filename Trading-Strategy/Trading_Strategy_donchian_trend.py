#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill
#This ones bad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def donchian_trend_strat(commodity_data_df, price_col, ema_window=14, avg_window = 350, window_high=20):
        
    dtrend_df = commodity_data_df.copy()
    dtrend_df['EMA_25'] = dtrend_df[price_col].ewm(span=ema_window, adjust=False).mean()
    dtrend_df['MA_350'] = dtrend_df[price_col].rolling(window=avg_window).mean()

    dtrend_df['Position'] = np.where((dtrend_df['EMA_25'] > dtrend_df['MA_350']) & (dtrend_df[price_col] > dtrend_df[price_col].shift(1).rolling(window=window_high).mean()), 1, 0)    
    dtrend_df['Position'] = np.where((dtrend_df['EMA_25'] < dtrend_df['MA_350']) & (dtrend_df[price_col] < dtrend_df[price_col].shift(1).rolling(window=window_high).mean()), -1, dtrend_df['Position'])    

    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = dtrend_df['Date']
    clean_return_df[price_col] = dtrend_df[price_col]
    clean_return_df['Position'] = dtrend_df['Position']
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
        strat_df = donchian_trend_strat(commodity_df, commodities[i])
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
