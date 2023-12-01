#MF703 Final Project
#Backtester


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def t12_t1_moving_avg_strat(commodity_data_df, window_12M = 252, window_1M = 21):
    '''INSERT DESCRIPTION HERE'''
    
    commodity_names = list(commodity_data_df.columns)
    index_data_df = pd.DataFrame()
    mov_avg = commodity_data_df.copy()
    
    #Calculate the moving averages for each commodity, and the difference of the 12MA and 1MA
    for i in range(len(commodity_names)):
        mov_avg[commodity_names[i] + ' 12M MA'] = mov_avg[commodity_names[i]].rolling(window_12M).mean()
        mov_avg[commodity_names[i] + ' 1M MA'] = mov_avg[commodity_names[i]].rolling(window_1M).mean()
        mov_avg[commodity_names[i] +' Position'] = (mov_avg[commodity_names[i] + ' 12M MA'] - mov_avg[commodity_names[i] + ' 1M MA']) / mov_avg[commodity_names[i] + ' 1M MA']
        index_data_df[commodity_names[i]] = mov_avg[commodity_names[i] +' Position']
    
    #Rank the difference in the commodity MA for each date, rank 1-4 short, rank 13-16 long (scaled on how many availible)
    #Reccomend long position in top 4 commodiites, and short position in bottom 4 commodities      
    index_data_df = index_data_df.rank(axis=1, method='min')
    index_data_df['Row Count'] = index_data_df.apply(lambda row: row.count(), axis=1)
    index_data_df = index_data_df.div(index_data_df['Row Count'], axis=0)

    index_data_df = index_data_df.applymap(
        lambda x: -1 if x <= .25 else (1 if x > .80 else 0))

    index_data_df.drop('Row Count',axis=1,inplace=True)
    return index_data_df


def daily_returns(prices_df, positions_df):
    '''Calculates the daily returns of the different commodites based on the signals provided by the
       trading strategy.
       Inputs: prices_df, df containing daily price data for the different commodities,
       positions_df, df containing the recommended positions for each commodity based on the trading strategy
       Outputs: scaled_daily_returns, df containing scaled daily returns for each commodity in our universe'''
    
    daily_returns_df = prices_df.shift(-2).pct_change()
    scaled_daily_returns = 1+(daily_returns_df * positions_df)
    
    return scaled_daily_returns


def portfolio_reblancing_identification(positions_df):    
    '''This function identifies when portfolio rebalancing is required. The portfolio should
       be rebalanced at a minimum of every 7 trading days or when a position change is reccomended. 
       Reblanacing is to occur based on whicever event occurs first.
       inputs: positions_df which provides the positions for the assets
       outputs: signal_df, which has 1 if rebalancing is needed on a date, 0 if not needed'''
    
    #Generate a reblancing signal if the reccomended position changes AND its a valid trading day
    temp_df = positions_df.ne(positions_df.shift())
    temp_df = temp_df.any(axis=1)
    #valid_trading_day_df = valid_trading_day_df.set_index(positions_df.index)
    #temp_df = temp_df * valid_trading_day_df['Tradeable Day']
    signal_df = pd.DataFrame(index=positions_df.index)
    signal_df['Rebalance'] = temp_df
    signal_array = signal_df.values
    
    #Generate a rebalancing signal if it has been 5 trading days since the last rebalancing
    consecutive_false_count = 0
    for i in range(len(signal_array)):
        if signal_array[i][0] == 0:
            consecutive_false_count += 1
            if consecutive_false_count == 5:
                signal_array[i][0] = 1
                consecutive_false_count = 0
        else:
            consecutive_false_count = 0
    signal_df = pd.DataFrame(signal_array, columns=['Rebalance'])
    signal_df = signal_df.set_index(positions_df.index)
    
    return signal_df


def Back_Tester():
    #Ensure that the BCOM column has been properly cleaned (remove the few excess data points)
    #Place commodity index data into a df, ensure sorted from oldest to newest data, make date the index
    commodity_prices_df_header = pd.read_csv("S&P Commodity Data (with BCOM).csv")
    commodity_prices_df = commodity_prices_df_header
    commodity_prices_df['Date'] = pd.to_datetime(commodity_prices_df['Date'])
    commodity_prices_df.sort_values(by='Date',inplace=True)
    commodity_prices_df.iloc[:,1:] = commodity_prices_df.iloc[:,1:].astype(float)
    commodity_prices_df.set_index('Date', inplace=True)
    
    #Place BCOM index data into a seperate df and drop it from commodity_prices
    BCOM_prices_df = pd.DataFrame(index=commodity_prices_df.index)
    BCOM_prices_df['BCOM Index'] = commodity_prices_df['BCOM Index']
    commodity_prices_df = commodity_prices_df.drop('BCOM Index', axis=1)
    
    #Call the trading strategey to generate trade signals, and provide rewccomended positions
    positions_df = t12_t1_moving_avg_strat(commodity_prices_df)
    
    #Calculate the scaled daily returns of the trading strategy
    scaled_daily_returns_df = daily_returns(commodity_prices_df, positions_df)
    
    #Identify valid rebalancing dates
    rebalancing_dates_df = portfolio_reblancing_identification(positions_df)
    
    #Determine portfolio weights
    
    #Determine returns with rebalancing
    
    return rebalancing_dates_df


if __name__ == "__main__":
    """For Testing Purposes Only, remove before adding to final project file"""
    results = Back_Tester()
    output_df = pd.DataFrame()
    output_df = results
        
    results.to_csv('output_data.csv', index=True)