#MF703 Final Project
#Trading Strategy: MACD
#Author: Jim Burrill
#This one is good

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def MACD_strat(commodity_data_df, price_col, short_window=12, long_window=26, signal_window=9):
    
    mcad_df = commodity_data_df.copy()
    mcad_df['Short_EMA'] = mcad_df[price_col].ewm(span=short_window, adjust=False).mean()
    mcad_df['Long_EMA'] = mcad_df[price_col].ewm(span=long_window, adjust=False).mean()
    mcad_df['MACD'] = mcad_df['Short_EMA'] - mcad_df['Long_EMA']
    mcad_df['Signal_Line'] = mcad_df['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    mcad_df['Position'] = np.where((mcad_df['MACD'] > mcad_df['Signal_Line']) & (mcad_df['MACD'].shift(1) <= mcad_df['Signal_Line'].shift(1)), 1, 0)
    mcad_df['Position'] = np.where((mcad_df['MACD'] < mcad_df['Signal_Line']) & (mcad_df['MACD'].shift(1) >= mcad_df['Signal_Line'].shift(1) ), -1, mcad_df['Position'])

    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = mcad_df['Date']
    clean_return_df[price_col] = mcad_df[price_col]
    clean_return_df['Position'] = mcad_df['Position']
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
        strat_df = MACD_strat(commodity_df, commodities[i])
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
