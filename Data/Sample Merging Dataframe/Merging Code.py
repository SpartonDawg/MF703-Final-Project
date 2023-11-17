#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 20:31:00 2023

@author: Sagar
"""

import pandas as pd

def merge_data(files, start_date='2023-11-14'):
    start_date = pd.to_datetime(start_date)

    merged_df = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file)

        df['Date'] = pd.to_datetime(df['Date'])

        df = df[df['Date'] <= start_date]

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Date', how='inner')

    missing_dates = merged_df['Date'].isnull().any()

    if missing_dates:
        print("Warning: There are missing dates in the merged dataframe.")

    merged_df.sort_values(by='Date', inplace=True, ascending=False)

    merged_df.reset_index(drop=True, inplace=True)

    merged_df.fillna(0, inplace=True)

    return merged_df

files_list = ['CL1.csv', 'CO1.csv', 'CP1.csv']

start_date = '2023-11-14'

merged_data = merge_data(files_list, start_date)

merged_data.to_csv('merged_data.csv', index=False)


