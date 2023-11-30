#MF703 Final Project
#Returns Calculation
#Author: Jim Burrill

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mktcal


def returns_rebalancing(prices_df, position_weights_df, rebalancing_dates_df, transaction_costs = 0.0005):   
 
    #Determine position values via dot product with weights
    position_values_df = prices_df.dot(position_weights_df.T)
    #If not a valid rebalancing date, position values go to zero
    rebalancing_values_df = position_values_df * rebalancing_dates_df
    #Drop rows with all zeros
    rebalancing_values_df = rebalancing_values_df[(rebalancing_values_df != 0).any(axis=1)]
    
    port_returns_df = pd.DataFrame(index=rebalancing_values_df.index)
    cumulative_returns_df = pd.DataFrame(index=rebalancing_values_df.index)
    
    port_returns_df = (1 - transaction_costs) + (((rebalancing_values_df.shift(-1) - rebalancing_values_df)/rebalancing_values_df))
    cumulative_returns_df = port_returns_df.cumprod()-1
    
    return cumulative_returns_df