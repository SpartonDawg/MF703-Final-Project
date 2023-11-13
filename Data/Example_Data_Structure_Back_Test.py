import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


def Back_Tester(df, fwd_daily_return_col, position_col):
    """
    Assumptions:
    Daily Close data
    The position column is the assuming we got filled at the closing price on that day, T.
    So the return needs to be calculated from T to T+1.

    Example:
    1/1/2000:
    Closing Price: 100
    Position: 1
    Return: 0%
    1/2/2000:
    Closing Price: 101
    Position: 1
    Return: 1%

    Position Types:
        1: Long
        0: Not Holding
       -1: Short
    """
    copy_df = df.copy()
    copy_df['Position Scaled Return'] = 1+(copy_df[fwd_daily_return_col] * copy_df[position_col])
    copy_df['Strategy Total Return'] = copy_df['Position Scaled Return'].cumprod()-1
    return copy_df


def calculate_bollinger_bands(df,price_col, window_size=20, num_std_dev=2):
    # Calculate the rolling mean and standard deviation
    copy_df = df.copy()
    copy_df['Rolling Mean'] = copy_df[price_col].rolling(window=window_size).mean()
    copy_df['Upper Band'] = copy_df['Rolling Mean'] + (copy_df[price_col].rolling(window=window_size).std() * num_std_dev)
    copy_df['Lower Band'] = copy_df['Rolling Mean'] - (copy_df[price_col].rolling(window=window_size).std() * num_std_dev)

    return copy_df

def find_first_date(df):
    return df.dropna()['Date'].min()

"""
Loading testing data
"""
index_prices_df_header = pd.read_csv("Testing_Future_Data.csv")
index_prices_df = index_prices_df_header.iloc[1:]
index_prices_df['Date'] = pd.to_datetime(index_prices_df['Date'])
index_prices_df.sort_values(by='Date',inplace=True)
index_prices_df.iloc[:,1:] = index_prices_df.iloc[:,1:].astype(float)


"""
Cleaning individual commodity index
Creating dictionary of dataframes structure
"""
list_of_index_names = index_prices_df_header.columns[1:]
Commodity_Data_Master = {}

for i in list_of_index_names:
    temp_df = index_prices_df[['Date',i]]
    temp_df = temp_df[temp_df['Date'] > find_first_date(temp_df)]
    temp_df = temp_df.fillna(method='bfill').dropna()
    temp_df.reset_index(inplace=True)
    temp_df.drop("index",axis=1,inplace=True)

    temp_df[i+" Daily Return"] = temp_df[i].pct_change()
    temp_df[i+" FWD Daily Return"] = np.nan
    temp_df[i+" FWD Daily Return"].iloc[:-1] = temp_df[i+" Daily Return"].iloc[1:]
    Commodity_Data_Master[i] = temp_df


print("  ")
print(Commodity_Data_Master['Soybean'])
print("-------------------------------")
print(Commodity_Data_Master['Gold'])

Soybean_df = Commodity_Data_Master['Soybean']
Soybean_df = Soybean_df.iloc[1:]
Soybean_df.reset_index(inplace=True)
Soybean_df = Soybean_df.iloc[:,1:]

window_size = 21
Soybean_df = calculate_bollinger_bands(Soybean_df,'Soybean').iloc[window_size:]

Soybean_df['Position'] = 0
Soybean_df['Position'][Soybean_df['Soybean'] > Soybean_df['Upper Band']] = -1
Soybean_df['Position'][Soybean_df['Soybean'] < Soybean_df['Lower Band']] = 1

Soybean_df = Back_Tester(Soybean_df, 'Soybean FWD Daily Return', 'Position')
Soybean_df['Index TR'] = (1+Soybean_df['Soybean Daily Return']).cumprod()-1
print(Soybean_df)
plt.plot(Soybean_df['Date'], Soybean_df['Strategy Total Return'],label='Strategy Return')
plt.plot(Soybean_df['Date'], Soybean_df['Index TR'],label='Index Return')
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.legend()
plt.show()
