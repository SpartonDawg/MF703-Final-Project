#MF703 Final Project
#Trading Days Identification
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def valid_trading_day(commodity_data_df, price_col):
    '''Checks wether a particular date is a valid trading date based on the applicable exchange trading schedule.
       commodity_data_df, input df containing the price and date information for a particular commodity.
       clean_return_df, return df containing price, date, and Tradeable day columns. A 1 in the Tradeable
       Days column means valid trading day, a 0 means invalid trading day (weekend, holiday, backfilled data'''
    
    NYMEX_commodities = ['CL1:COM', 'NG1:COM', 'XB1:COM', 'HO1:COM', 'QS1:COM']
    ICE_commodities = ['CO1:COM', 'MO1:COM']
    Tokyo_commodities = ['JX1:COM']
    COMEX_commodities = ['GC1:COM', 'SL1:COM', 'HG1:COM']
    LME_commodities = ['LMCADS03:COM', 'LMAHDS03:COM', 'LMZSDS03:COM', 'LMNSNDS03:COM']
    CBOT_commodities = ['C1:COM', 'W1:COM', 'O1:COM', 'S1:COM', 'SM1:COM']
    
    if price_col in NYMEX_commodities:
        exchange_name = 'CMEGlobex_Energy'
    elif price_col in ICE_commodities:
        exchange_name = 'ICE'
    elif price_col in Tokyo_commodities:
        exchange_name = 'Financial_Markets_JP'
    elif price_col in COMEX_commodities:
        exchange_name = 'CMEGlobex_Metals'
    elif price_col in LME_commodities:
        exchange_name = 'Financial_Markets_UK'
    else:
        exchange_name = 'CBOT_Agriculture'
        
    exchange = mktcal.get_calendar(exchange_name)
    prices_df = commodity_data_df.copy()

    valid_days_df = pd.to_datetime(exchange.valid_days(start_date=prices_df['Date'].min(), end_date=prices_df['Date'].max()))
    valid_days_df = valid_days_df.date
       
    prices_df['Tradeable Day'] = np.where((prices_df[price_col].isna() == True), 0, 1)
    prices_df['Tradeable Day'] = np.where(prices_df['Date'].isin(valid_days_df) == False, 0, prices_df['Tradeable Day'])

    clean_return_df = pd.DataFrame()
    clean_return_df['Date'] = prices_df['Date']
    clean_return_df[price_col] = prices_df[price_col]
    clean_return_df['Tradeable Day'] = prices_df['Tradeable Day']
    return clean_return_df


def calc_returns(df, fwd_daily_return_col, position_col):
    '''Calculates the strategy return for the given data'''
    
    copy_df = df.copy()
    copy_df['Position Scaled Return'] = 1+(copy_df[fwd_daily_return_col] * copy_df[position_col] * copy_df['Tradeable Day'])
    copy_df['Strategy Total Return'] = copy_df['Position Scaled Return'].cumprod()-1
    return copy_df


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
        valid_trading_days_df = valid_trading_day(commodity_df, commodities[i])
        output_df[commodities[i]] = valid_trading_days_df[commodities[i]]
        output_df[commodities[i] + 'Tradeable Day'] = valid_trading_days_df['Tradeable Day']
    output_df.to_csv('output_data.csv', index=False)
