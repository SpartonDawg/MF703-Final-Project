import pandas as pd
import numpy as np

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
