#MF703 Final Project
#Portfolio Rebalancing
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


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
    
    #Generate a rebalancing signal if it has been 7 trading days since the last rebalancing
    consecutive_false_count = 0
    for i in range(len(signal_array)):
        if signal_array[i][0] == 0:
            consecutive_false_count += 1
            if consecutive_false_count == 7:
                signal_array[i][0] = 1
                consecutive_false_count = 0
        else:
            consecutive_false_count = 0
    signal_df = pd.DataFrame(signal_array, columns=['Rebalance'])
    signal_df = signal_df.set_index(positions_df.index)
    
    return signal_array