#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



def t12_t1_moving_avg_strat(commodity_data_df, price_col, window_12M = 252, window_1M = 21, percent_differnce = .1):
    
    mov_avg = commodity_data_df.copy()
    mov_avg['12M MA'] = mov_avg[price_col].rolling(window_12M).mean()
    mov_avg['1M MA'] = mov_avg[price_col].rolling(window_1M).mean()

    mov_avg['Position'] = np.where(mov_avg['1M MA'] > (mov_avg['12M MA'] * percent_differnce), 1, 0)
    mov_avg['Position'] = np.where(mov_avg['1M MA'] < (mov_avg['12M MA'] * (1-percent_differnce)), -1, mov_avg['Position'])
    
    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = mov_avg['Date']
    clean_return_df[price_col] = mov_avg[price_col]
    clean_return_df['Position'] = mov_avg['Position']
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
    
    for i in range(0,len(commodities)):
        commodity_df = pd.DataFrame()
        commodity_df['Date'] = commodity_data_master['Date']
        commodity_df[commodities[i]] = commodity_data_master[commodities[i]]
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        averages_df = t12_t1_moving_avg_strat(commodity_df, commodities[i])
        averages_df['Market Return'] = (averages_df[commodities[i]] - averages_df[commodities[i]].shift(1))/averages_df[commodities[i]].shift(1)
        averages_df['Position Scaled Return'] = 1+(((averages_df[commodities[i]].shift(-1) - averages_df[commodities[i]])/averages_df[commodities[i]]) * averages_df['Position'])
        averages_df['Strategy Total Return'] = averages_df['Position Scaled Return'].cumprod()-1
        plt.figure(i)    
        plt.plot(averages_df['Date'], averages_df['Strategy Total Return'], label= commodities[i])
        plt.legend()
