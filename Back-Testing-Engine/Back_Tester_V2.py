import pandas as pd
import numpy as np

def back_tester(df,index_return_col, position_col):
    """
    Position Types:
        1: Long
        0: Not Holding
       -1: Short
    """
    copy_df = df.copy()
    copy_df['Position Scaled Return'] = 1+(copy_df[index_return_col] * copy_df[position_col])
    copy_df['Strategy Total Return'] = copy_df['Position Scaled Return'].cumprod()-1
    return copy_df
