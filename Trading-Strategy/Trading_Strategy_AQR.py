#MF703 Final Project
#AQR Trading Strategy
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def aqr_momentum_calculation(commodity_data_df, price_col, window_size = 252):    
        
    momentum_df = commodity_data_df.copy()
    pct_change = momentum_df[price_col].pct_change().rolling(window=window_size).sum()
    momentum_df['Total Return'] = (1 + pct_change).cumprod() - 1
    
    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = momentum_df['Date']
    clean_return_df[price_col] = momentum_df[price_col]
    return clean_return_df

def aqr_signal_generation(commodity_data_master, commodities, window = 252):
    
    returns_df = pd.DataFrame()
    clean_return_df = commodity_data_master.copy()
    returns_df['Date'] = commodity_data_master['Date']
    for i in range(0,len(commodities)):
        commodity_df = pd.DataFrame()
        commodity_df['Date'] = commodity_data_master['Date']
        commodity_df[commodities[i]] = commodity_data_master[commodities[i]]
        commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])
        temp_df = aqr_momentum_calculation(commodity_df, commodities[i])
        returns_df[commodities[i]] = temp_df[commodities[i]]
    returns_df = returns_df.rank(axis=1, ascending=False)
    returns_df.insert(0, 'Date', commodity_df['Date'])
    
    for i in range(0,len(commodities)):
        signals = pd.DataFrame()
        signals = np.where(returns_df[commodities[i]] <= 4, 1, 0)
        signals = np.where(returns_df[commodities[i]] >= 16, -1, signals)
        clean_return_df.insert(i+1, commodities[i] + ' Position', signals)
    
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
    
    cum_strat_total_return = pd.DataFrame()
    individual_total_returns = pd.DataFrame()
    output_df = pd.DataFrame()
    output_df['Date'] = commodity_data_master['Date']
    strat_df = aqr_signal_generation(commodity_data_master, commodities)    
    for i in range(0,len(commodities)):     
        strat_df['Position Scaled Return'] = 1+(((commodity_data_master[commodities[i]].shift(-1) - commodity_data_master[commodities[i]])/commodity_data_master[commodities[i]]) * strat_df[commodities[i] + ' Position'])
        individual_total_returns[commodities[i] + ' Strategy Total Return'] = strat_df['Position Scaled Return'].cumprod()-1
    cum_strat_total_return = individual_total_returns.sum(axis=1)
    strat_df['Date'] = pd.to_datetime(strat_df['Date'])
    plt.figure(1)    
    plt.plot(strat_df['Date'], cum_strat_total_return, label= 'Strategy Total Return')
    plt.legend()
    output_df = cum_strat_total_return
    output_df.to_csv('output_data.csv', index=False)
