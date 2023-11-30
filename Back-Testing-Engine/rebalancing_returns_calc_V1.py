#MF703 Final Project
#Returns Calculation
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def returns_rebalancing(scaled_daily_returns_df, position_weights_df, rebalancing_dates_df, transaction_costs = 0.0005):   
    
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
    
    daily_returns_df =  prices_df.shift(-2).pct_change()
    scaled_daily_returns = 1+(daily_returns_df * positions_df)
    
    return scaled_daily_returns
