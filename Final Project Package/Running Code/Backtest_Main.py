#MF703 Final Project
#Author: Jack & Jim

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import Weight_Optimizer as Covariance_Optimization


def breakout_strat(commodity_data_df, atr_window=14, avg_window = 252):
    '''This strategy is breakout strategy. Takes the average true range over a 14 day period.
       The upper channel is defined by adding a 7 ATR to the 1year moving average, and the lower channel
       is defined by subtracting a 3 ATR to the 1year moving average. A buy signal is generated when the
       price of the commodity rises above the upper channel, and a sell signal is generated when the
       commodity price falls below the lower channel.
       Inputs: commodity_data_df, df containing daily price data for the different commodities,
       atr_window, window used to calculate the ATR, avg_window, window used to calc the moving average
       Outputs: clean_return_df, contains position signals for all commodities'''

    commodity_names = list(commodity_data_df.columns)
    clean_return_df = pd.DataFrame()
    breakout_df = commodity_data_df.copy()

    for i in range(len(commodity_names)):
        breakout_df[commodity_names[i] + ' ATR'] = breakout_df[commodity_names[i]].diff().abs().rolling(window=atr_window).mean()
        breakout_df[commodity_names[i] + ' Upper Channel'] = breakout_df[commodity_names[i]].rolling(window=avg_window).mean() + 7 * breakout_df[commodity_names[i] + ' ATR']
        breakout_df[commodity_names[i] + ' Lower Channel'] = breakout_df[commodity_names[i]].rolling(window=avg_window).mean() - 3 * breakout_df[commodity_names[i] + ' ATR']
        breakout_df[commodity_names[i] + ' Position'] = np.where(breakout_df[commodity_names[i]]  >  breakout_df[commodity_names[i] + ' Upper Channel'], 1, 0)
        breakout_df[commodity_names[i] + ' Position'] = np.where(breakout_df[commodity_names[i]]  <  breakout_df[commodity_names[i] + ' Lower Channel'], -1, breakout_df[commodity_names[i] + ' Position'])
        clean_return_df[commodity_names[i]] = breakout_df[commodity_names[i] + ' Position']

    clean_return_df = clean_return_df.set_index(commodity_data_df.index)
    return clean_return_df


def AQR_strat(commodity_data_df, window_aqr = 252):
    '''This strategy calculates the return for all commodities over a rolling 1 yr window. The
       moving averages are then ranked from lowest to highest. The commodities ranked 1-3(4) are shorted
       and those ranked 13-16 are longed (scaled based on the number of commodities in the unverse at a given time).
       Inputs: commodity_data_df, df containing daily price data for the different commodities,
       Outputs: clean_return_df, contains position signals for all commodities'''

    commodity_names = list(commodity_data_df.columns)
    index_data_df = pd.DataFrame()
    mov_avg = commodity_data_df.copy()

    #Calculate the moving averages for each commodity, and the difference of the 12MA and 1MA
    for i in range(len(commodity_names)):
        mov_avg[commodity_names[i] +' Position'] = mov_avg[commodity_names[i]].pct_change(periods=window_aqr)
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


def returns_rebalancing(scaled_daily_returns_df, position_weights_df, rebalancing_dates_df, transaction_costs = 0.0000):
    """This function calculates the returns of the overall strategy when reblancing is performed according
       to the portfolio_reblancing_identification funtion.
       Inputs: scaled_daily_returns_df, a dataframe of sclaed daily returns based on the trading strategy,
       position_weights_df, a dataframe containing weights to be taken in each position reccomended by the trading
       strategy based on the output of the calculate_weights function,
       Outputs: total_returns_df, a dataframe that contains the total strategy returns'''
     """

    selected_indices = rebalancing_dates_df[rebalancing_dates_df['Rebalance'] > 0].index.values
    selected_indices = pd.to_datetime(selected_indices)
    all_port_returns = []
    for i in range(len(selected_indices)-1):
        temp_df = scaled_daily_returns_df.loc[selected_indices[i]:selected_indices[i+1]]
        temp_df = temp_df.fillna(1)
        temp_returns = temp_df.prod()-(1 + transaction_costs)
        period_port_returns = np.dot(temp_returns, position_weights_df.loc[temp_df.index[0]])
        all_port_returns.append(period_port_returns)

    total_returns_df = pd.DataFrame({"Date":selected_indices[:-1],"Portfolio Return":all_port_returns})

    return total_returns_df


def portfolio_reblancing_identification(positions_df, rebal_period=5):
    '''This function identifies when portfolio rebalancing is required. The portfolio should
       be rebalanced at a minimum of every 5 trading days or when a position change is reccomended.
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
    signal_array[:251] = 1

    #Generate a rebalancing signal if it has been 5 trading days since the last rebalancing
    consecutive_false_count = 0
    for i in range(len(signal_array)):
        if signal_array[i][0] == 0:
            consecutive_false_count += 1
            if consecutive_false_count == rebal_period:
                signal_array[i][0] = 1
                consecutive_false_count = 0
        else:
            consecutive_false_count = 0
    signal_df = pd.DataFrame(signal_array, columns=['Rebalance'])
    signal_df = signal_df.set_index(positions_df.index)
    signal_df['Rebalance'].iloc[:251] = False

    return signal_df


def generate_equal_weights(input_position_df, scaled_returns_df):
    '''Generater equal weights for the portfolio based on the reccomended positions'''
    scaled_returns_df = scaled_returns_df-1
    equal_weights = []
    for i in range(len(input_position_df)):
        len_i = len(input_position_df.iloc[i][input_position_df.iloc[i].fillna(0) != 0])
        equal_weights.append(list(np.array(scaled_returns_df.iloc[i].fillna(0) != 0) / len_i))

    return_df = pd.DataFrame(equal_weights)
    return_df.columns = scaled_returns_df.columns
    return_df.index = scaled_returns_df.index
    return return_df


def Back_Tester(weight_type,prices_df):
    #Ensure that the BCOM column has been properly cleaned (remove the few excess data points)
    #Place commodity index data into a df, ensure sorted from oldest to newest data, make date the index
    #weight_type 0 = Optimized Weights
    #weight_type 1 = Equal Weights
    commodity_prices_df = prices_df.copy()
    BCOM_prices_df = pd.DataFrame(index=commodity_prices_df.index)
    BCOM_prices_df['BCOM Index'] = commodity_prices_df['BCOM Index']
    commodity_prices_df = commodity_prices_df.drop('BCOM Index', axis=1)

    #Call the trading strategey to generate trade signals, and provide rewccomended positions
    positions_df = AQR_strat(commodity_prices_df)

    #Calculate the scaled daily returns of the trading strategy
    scaled_daily_returns_df = daily_returns(commodity_prices_df, positions_df)

    #Identify valid rebalancing dates
    rebalancing_dates_df = portfolio_reblancing_identification(positions_df)
    loop_dates = rebalancing_dates_df[rebalancing_dates_df['Rebalance'] == True]

    if weight_type == 0:
        WeightOptimizer = Covariance_Optimization.WeightOptimization(BCOM_prices_df, commodity_prices_df, positions_df)
        covar_min_weights = [[0]*len(scaled_daily_returns_df.columns)]
        covar_dates = []
        for date_i in loop_dates.index:
            try:
                opt_weights = WeightOptimizer.calculate_weights(date_i, alpha=100)
                covar_min_weights.append(list(opt_weights))
                covar_dates.append(date_i)
            except:
                covar_min_weights.append(covar_min_weights[-1])
                covar_dates.append(date_i)


        covar_df = pd.DataFrame(covar_min_weights[1:]).abs()
        covar_df.columns = scaled_daily_returns_df.columns
        covar_df.index = loop_dates.index

        portfolio_returns_df = returns_rebalancing(scaled_daily_returns_df.iloc[1:], covar_df, rebalancing_dates_df.iloc[1:], transaction_costs = 0.0000)
        portfolio_returns_df.to_csv('trans_portfolio_returns_covar_data_breakout.csv', index=True)
    if weight_type == 1:
        equal_weights = generate_equal_weights(positions_df,scaled_daily_returns_df)
        portfolio_returns_df = returns_rebalancing(scaled_daily_returns_df.iloc[1:], equal_weights, rebalancing_dates_df.iloc[1:], transaction_costs = 0.0000)
        portfolio_returns_df.to_csv('trans_portfolio_returns_equal_data_breakout.csv', index=True)
    return portfolio_returns_df


if __name__ == "__main__":
    plt.style.use("seaborn-talk")
    plt.figure(figsize=(11,10))


    commodity_prices_df = pd.read_excel("S&P Commodity Data (with BCOM).xlsx",index_col='Date')
    commodity_prices_df = commodity_prices_df.fillna(method='ffill')
    commodity_prices_df.index = pd.to_datetime(commodity_prices_df.index)
    commodity_prices_df.iloc[:,1:] = commodity_prices_df.iloc[:,1:].astype(float)

    plt.plot(commodity_prices_df.index[252:],(1+commodity_prices_df['BCOM Index'].iloc[252:].pct_change()).cumprod()-1,label="BCOM Index")

    equal_weights_returns = Back_Tester(weight_type = 1, prices_df=commodity_prices_df)
    plt.plot(pd.to_datetime(equal_weights_returns['Date']),(1+equal_weights_returns['Portfolio Return']).cumprod()-1,label="Equal Weights")
    covar_optimal_returns = Back_Tester(weight_type = 0, prices_df=commodity_prices_df)
    plt.plot(pd.to_datetime(covar_optimal_returns['Date']),(1+covar_optimal_returns['Portfolio Return']).cumprod()-1,label="Optimized Weights")

    plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title("AQR Trading Strategy", size=22)
    plt.ylabel("Total Return",size=16)
    plt.legend()
    plt.show()
