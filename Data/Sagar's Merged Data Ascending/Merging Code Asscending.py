import pandas as pd

files_list = ['CL1.csv', 'CO1.csv', 'CP1.csv', 'SM1.csv', 'C 1.csv', 'GC1.csv', 'HG1.csv', 'HO1.csv', 'JX1.csv',
              'LMAHDS03.csv', 'LMCADS03.csv', 'LMSNDS03.csv', 'LMZSDS03.csv', 'MO1.csv', 'GC1.csv', 'NG1.csv',
              'O 1.csv', 'QS1.csv', 'S1.csv', 'SI1.csv', 'W 1.csv', 'XB1.csv']

merged_df = pd.DataFrame()

for file in files_list:

    df = pd.read_csv(file, parse_dates=['Date'], dayfirst=True, infer_datetime_format=True)

    df.set_index('Date', inplace=True)

    if merged_df.empty:
        merged_df = df
    else:
        merged_df = merged_df.join(df, how='outer', rsuffix=f'_{file.split(".")[0]}')

merged_df.sort_index(inplace=True)

merged_df.reset_index(inplace=True)

merged_df.to_csv('merged_data_ascending.csv', index=False)

print("Merged data has been saved to 'merged_data_ascending.csv'")
