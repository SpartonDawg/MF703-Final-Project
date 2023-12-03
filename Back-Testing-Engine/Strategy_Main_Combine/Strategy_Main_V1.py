#MF703 Final Project
#Backtester


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import Covariance_V5 as Covariance_Optimization

def MACD_strat(commodity_data_df, short_window=12, long_window=26, signal_window=9):

    commodity_names = list(commodity_data_df.columns)
    mcad_df = commodity_data_df.copy()
    clean_return_df = pd.DataFrame()

    for i in range(len(commodity_names)):
        mcad_df[commodity_names[i] + ' Short_EMA'] = mcad_df[commodity_names[i]].ewm(span=short_window, adjust=False).mean()
        mcad_df[commodity_names[i] + ' Long_EMA'] = mcad_df[commodity_names[i]].ewm(span=long_window, adjust=False).mean()
        mcad_df[commodity_names[i] + ' MACD'] = mcad_df[commodity_names[i] + ' Short_EMA'] - mcad_df[commodity_names[i] + ' Long_EMA']
        mcad_df[commodity_names[i] + ' Signal_Line'] = mcad_df[commodity_names[i] + ' MACD'].ewm(span=signal_window, adjust=False).mean()

        mcad_df[commodity_names[i] + ' Position'] = np.where((mcad_df[commodity_names[i] + ' MACD'] > mcad_df[commodity_names[i] + ' Signal_Line']) & (mcad_df[commodity_names[i] + ' MACD'].shift(1) <= mcad_df[commodity_names[i] + ' Signal_Line'].shift(1)), 1, 0)
        mcad_df[commodity_names[i] + ' Position'] = np.where((mcad_df[commodity_names[i] + ' MACD'] < mcad_df[commodity_names[i] + ' Signal_Line']) & (mcad_df[commodity_names[i] + ' MACD'].shift(1) >= mcad_df[commodity_names[i] + ' Signal_Line'].shift(1) ), -1, mcad_df[commodity_names[i] + ' Position'])
        clean_return_df[commodity_names[i]] = mcad_df[commodity_names[i] + ' Position']

    clean_return_df = clean_return_df.set_index(commodity_data_df.index)
    return clean_return_df

def t12_t1_moving_avg_strat(commodity_data_df, window_12M = 45, window_1M = 15):
    '''INSERT DESCRIPTION HERE'''

    commodity_names = list(commodity_data_df.columns)
    index_data_df = pd.DataFrame()
    mov_avg = commodity_data_df.copy()

    #Calculate the moving averages for each commodity, and the difference of the 12MA and 1MA
    for i in range(len(commodity_names)):
        mov_avg[commodity_names[i] + ' 12M MA'] = mov_avg[commodity_names[i]].rolling(window_12M).mean()
        mov_avg[commodity_names[i] + ' 1M MA'] = mov_avg[commodity_names[i]].rolling(window_1M).mean()
        # mov_avg[commodity_names[i] +' Position'] = (mov_avg[commodity_names[i] + ' 12M MA'] - mov_avg[commodity_names[i] + ' 1M MA']) / mov_avg[commodity_names[i] + ' 1M MA']
        mov_avg[commodity_names[i] +' Position'] = (mov_avg[commodity_names[i] + ' 1M MA'] - mov_avg[commodity_names[i] + ' 12M MA']) / mov_avg[commodity_names[i] + ' 12M MA']
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



def AQR_strat(commodity_data_df):
    '''INSERT DESCRIPTION HERE'''

    commodity_names = list(commodity_data_df.columns)
    index_data_df = pd.DataFrame()
    mov_avg = commodity_data_df.copy()

    #Calculate the moving averages for each commodity, and the difference of the 12MA and 1MA
    for i in range(len(commodity_names)):
        print(mov_avg[commodity_names[i]])
        # mov_avg[commodity_names[i] + ' 12M MA'] = mov_avg[commodity_names[i]].rolling(window_12M).mean()
        # mov_avg[commodity_names[i] + ' 1M MA'] = mov_avg[commodity_names[i]].rolling(window_1M).mean()
        # mov_avg[commodity_names[i] +' Position'] = (mov_avg[commodity_names[i] + ' 12M MA'] - mov_avg[commodity_names[i] + ' 1M MA']) / mov_avg[commodity_names[i] + ' 1M MA']
        mov_avg[commodity_names[i] +' Position'] = mov_avg[commodity_names[i]].pct_change(periods=125)
        print(mov_avg[commodity_names[i] +' Position'])
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


def returns_rebalancing(scaled_daily_returns_df, position_weights_df, rebalancing_dates_df, transaction_costs = 0.0005):
    """This function calculates the returns of the overall strategy when reblancing is performed according
       to the portfolio_reblancing_identification funtion.
       Inputs: scaled_daily_returns_df, a dataframe of sclaed daily returns based on the trading strategy,
       position_weights_df, a dataframe containing weights to be taken in each position reccomended by the trading
       strategy based on the output of the calculate_weights function,
       Outputs: total_returns_df, a dataframe that contains the total strategy returns'''
     """
    # rebalancing_dates_df.index = pd.to_datetime(rebalancing_dates_df.index)
    # selected_indices = rebalancing_dates_df.index[rebalancing_dates_df > 0].tolist()


    selected_indices = rebalancing_dates_df[rebalancing_dates_df['Rebalance'] > 0].index.values
    selected_indices = pd.to_datetime(selected_indices)
    # import time
    # print("----------------------------")
    # time.sleep(100)
    all_port_returns = []
    for i in range(len(selected_indices)-1):
        temp_df = scaled_daily_returns_df.loc[selected_indices[i]:selected_indices[i+1]]

        temp_df = temp_df.fillna(1)
        temp_returns = temp_df.prod()-1
        # print(temp_returns)
        # print(position_weights_df)
        #
        # import time
        # time.sleep(100)
        # print(temp_df)
        # print(temp_returns)
        # print("------------------")
        # print(position_weights_df.iloc[i])
        # print("---------------------")
        period_port_returns = np.dot(temp_returns, position_weights_df.loc[temp_df.index[0]])
        all_port_returns.append(period_port_returns)

    total_returns_df = pd.DataFrame({"Date":selected_indices[:-1],"Portfolio Return":all_port_returns})
    # total_returns_df.to_csv("TR.csv")

    # total_returns_df = pd.DataFrame(index=rebalancing_dates_df.index)
    # total_returns_df['Strategy Returns'] = all_port_returns
    # print(total_returns_df)
    # total_returns_df.to_csv("tr.csv")
    return total_returns_df


def portfolio_reblancing_identification(positions_df,rebal_period=20):
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
    scaled_returns_df = scaled_returns_df-1
    equal_weights = []
    for i in range(len(input_position_df)):
        len_i = len(input_position_df.iloc[i][input_position_df.iloc[i].fillna(0) != 0])
        equal_weights.append(list(np.array(scaled_returns_df.iloc[i].fillna(0) != 0) / len_i))

    return_df = pd.DataFrame(equal_weights)
    return_df.columns = scaled_returns_df.columns
    return_df.index = scaled_returns_df.index
    return return_df


# def Back_Tester(short_window=12, long_window=26, signal_window=9):
def Back_Tester():
    #Ensure that the BCOM column has been properly cleaned (remove the few excess data points)
    #Place commodity index data into a df, ensure sorted from oldest to newest data, make date the index

    commodity_prices_df = pd.read_excel("S&P Commodity Data (with BCOM).xlsx",index_col='Date')
    commodity_prices_df.index = pd.to_datetime(commodity_prices_df.index)
    # commodity_prices_df_header = commodity_prices_df_header.iloc[:,:-1]

    # commodity_prices_df_header = pd.read_csv("S&P Commodity Data (with BCOM).csv")
    # commodity_prices_df = commodity_prices_df_header
    # commodity_prices_df['Date'] = pd.to_datetime(commodity_prices_df['Date'])
    # commodity_prices_df.sort_values(by='Date',inplace=True)

    commodity_prices_df.iloc[:,1:] = commodity_prices_df.iloc[:,1:].astype(float)

    # commodity_prices_df.set_index('Date', inplace=True)

    #Place BCOM index data into a seperate df and drop it from commodity_prices
    BCOM_prices_df = pd.DataFrame(index=commodity_prices_df.index)
    BCOM_prices_df['BCOM Index'] = commodity_prices_df['BCOM Index']
    commodity_prices_df = commodity_prices_df.drop('BCOM Index', axis=1)

    #Call the trading strategey to generate trade signals, and provide rewccomended positions
    positions_df = t12_t1_moving_avg_strat(commodity_prices_df)
    #Calculate the scaled daily returns of the trading strategy
    scaled_daily_returns_df = daily_returns(commodity_prices_df, positions_df)

    equal_weights = generate_equal_weights(positions_df,scaled_daily_returns_df)
    # equal_weights.to_csv("eql_weights.csv")

    # print()

    #Identify valid rebalancing dates
    rebalancing_dates_df = portfolio_reblancing_identification(positions_df)
    loop_dates = rebalancing_dates_df[rebalancing_dates_df['Rebalance'] == True]

    # rebalancing_dates_df.to_csv("rebal_dates.csv")
    # print(rebalancing_dates_df)

    # portfolio_returns_df = returns_rebalancing(scaled_daily_returns_df, equal_weights, rebalancing_dates_df, transaction_costs = 0.0005)
    # print(x)
    # Determine portfolio weights
    # Covariance_Optimization.WeightOptimization()
    # sample_date = rebalancing_dates_df.index[1]
    # print(sample_date)
    # print(type(sample_date))
    # sample_date = pd.to_datetime('1999-06-01')
    # print(sample_date)
    # print(type(sample_date))
    # import time
    # time.sleep(5)
    sample_date = rebalancing_dates_df.index[3000]
    # sample_date = pd.to_datetime('1999-06-01')
    WeightOptimizer = Covariance_Optimization.WeightOptimization(BCOM_prices_df, commodity_prices_df, positions_df)

    weights = WeightOptimizer.calculate_weights(sample_date, alpha=100)
    print(weights)
    print(sum(weights))
    # Determine returns with rebalancing

    return portfolio_returns_df

#
# if __name__ == "__main__":
#     """For Testing Purposes Only, remove before adding to final project file"""
# for i in range(2):

results = Back_Tester()
plt.plot(pd.to_datetime(results['Date']),(1+results['Portfolio Return']).cumprod()-1,label="1")

# results = Back_Tester(short_window=6, long_window=26, signal_window=9)
# plt.plot(pd.to_datetime(results['Date']),(1+results['Portfolio Return']).cumprod()-1,label="2")
#
# results = Back_Tester(short_window=9, long_window=26, signal_window=9)
# plt.plot(pd.to_datetime(results['Date']),(1+results['Portfolio Return']).cumprod()-1,label="3")
#
# results = Back_Tester(short_window=12, long_window=26, signal_window=9)
# plt.plot(pd.to_datetime(results['Date']),(1+results['Portfolio Return']).cumprod()-1,label="4")
# plt.legend()

# plt.plot(pd.to_datetime(df['Date']),(1+df['BCOM Index'].pct_change()).cumprod()-1)
plt.show()
# output_df = pd.DataFrame()
# output_df = results
#
# results.to_csv('output_data.csv', index=True)
