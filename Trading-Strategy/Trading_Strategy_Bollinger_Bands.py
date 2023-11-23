#MF703 Final Project
#Trading Strategy: Stochastic Oscillator
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def bollinger_band_strat(commodity_data_df, price_col, window_size = 21, num_of_std = 2):
    """This function will evaluate the data elements in the price column against the Bollinger 
       Bands in the columns UpperBound and LowerBound. The function will apply a long/short strategy,
       i.e., create a long position (+1) when the Observation crosses below the LowerBound, and create
       a short position (-1) when the Observation crosses above the UpperBound. A position of 0 means
       there is no signal, position holds. The function will returnthe same df provided but 
       containing additional columns Position, Market Return, and Strategy Return."""
    
    bbands_df = commodity_data_df.copy()
    bbands_df['Rolling Mean'] = bbands_df[price_col].rolling(window_size).mean()    
    bbands_df['Upper Band'] = bbands_df['Rolling Mean'] + (num_of_std * bbands_df[price_col].rolling(window_size).std())
    bbands_df['Lower Band'] = bbands_df['Rolling Mean'] - (num_of_std * bbands_df[price_col].rolling(window_size).std())
    bbands_df['Position'] = np.where(bbands_df[price_col] < bbands_df['Lower Band'], 1, 0)
    bbands_df['Position'] = np.where(bbands_df[price_col] > bbands_df['Upper Band'], -1, bbands_df['Position'])
    
    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = bbands_df['Date']
    clean_return_df[price_col] = bbands_df[price_col]
    clean_return_df['Position'] = bbands_df['Position']
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
        bbands_df = bollinger_band_strat(commodity_df, commodities[i])
        bbands_df['Market Return'] = (bbands_df[commodities[i]] - bbands_df[commodities[i]].shift(1))/bbands_df[commodities[i]].shift(1)
        bbands_df['Position Scaled Return'] = 1+(((bbands_df[commodities[i]].shift(-1) - bbands_df[commodities[i]])/bbands_df[commodities[i]]) * bbands_df['Position'])
        bbands_df['Strategy Total Return'] = bbands_df['Position Scaled Return'].cumprod()-1
        output_df[commodities[i] + 'Market Return'] = bbands_df['Market Return']
        output_df[commodities[i] + 'Position'] = bbands_df['Position']
        output_df[commodities[i] + 'Scaled Return'] = bbands_df['Position Scaled Return']
        output_df[commodities[i] + 'Strat Total Return'] = bbands_df['Strategy Total Return']
        plt.figure(i)    
        plt.plot(bbands_df['Date'], bbands_df['Strategy Total Return'], label= commodities[i])
        plt.legend()
    
    output_df.to_csv('output_data.csv', index=False)