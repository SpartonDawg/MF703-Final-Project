#MF703 Final Project
#Returns Calculation
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def returns_rebalancing(scaled_daily_returns_df, position_weights_df, rebalancing_dates_df, transaction_costs = 0.0005):   
    ''This function calculates the returns of the overall strategy when reblancing is performed according
       to the portfolio_reblancing_identification funtion.
       Inputs: scaled_daily_returns_df, a dataframe of sclaed daily returns based on the trading strategy,
       position_weights_df, a dataframe containing weights to be taken in each position reccomended by the trading
       strategy based on the output of the calculate_weights function,
       Outputs: total_returns_df, a dataframe that contains the total strategy returns'''
    
    selected_indices = rebalancing_dates_df.index[rebalancing_dates_df > 0].tolist()
    all_port_returns = []
    for i in range(len(selected_indices)):
        temp_df = scaled_daily_returns_df.loc[selected_indices[i]:selected_indices[i+1]]
        temp_returns = temp_df.prod()-1
        period_port_returns = np.dot(temp_returns, position_weights_df[i])
        all_port_returns.append(period_port_returns)
        
    total_returns_df = pd.DataFrame(index=rebalancing_dates_df.index)
    total_returns_df['Strategy Returns'] = all_port_returns

    return total_returns_df


def daily_returns(prices_df, positions_df):
    '''Calculates the daily returns of the different commodites based on the signals provided by the
       trading strategy.
       Inputs: prices_df, df containing daily price data for the different commodities,
       positions_df, df containing the recommended positions for each commodity based on the trading strategy
       Outputs: scaled_daily_returns, df containing scaled daily returns for each commodity in our universe'''

    daily_returns_df =  prices_df.shift(-2).pct_change()
    scaled_daily_returns = 1+(daily_returns_df * positions_df)
    
    return scaled_daily_returns
