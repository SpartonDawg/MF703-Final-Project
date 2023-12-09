#MF703 Final Project
#DataValidation
#Author: Sagar

import pandas as pd
import matplotlib.pyplot as plt

def data_validation(csv_file_path):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, parse_dates=['Date'], dayfirst=True)

    # List of ticker names
    tickers = ['SPGCCLP Index', 'SPGCHUP Index', 'SPGCNGP Index', 'SPGCGOP Index', 'SPGCHOP Index',
               'SPGCGCP Index', 'SPGCSIP Index', 'SPGCCNP Index', 'SPGCWHP Index', 'SPGCKCP Index',
               'SPGCSBP Index', 'SPGCILP Index', 'SPGCIKP Index', 'SPGCIAP Index', 'SPGCICP Index', 'SPGCIZP Index']

    # Threshold for anomaly detection
    threshold = 50

    # Check anomalies for each commodity ticker
    for ticker in tickers:
        # Calculate the daily return
        daily_return = df[ticker].pct_change()

        # Check for anomalies
        if daily_return.max() > threshold or daily_return.min() < -threshold:
            return f"Anomaly detected for {ticker}. Daily return exceeds the threshold."

    return df  # Return the original DataFrame if no anomalies detected

def plot_original_data(df):
    # List of ticker names
    tickers = ['SPGCCLP Index', 'SPGCHUP Index', 'SPGCNGP Index', 'SPGCGOP Index', 'SPGCHOP Index',
               'SPGCGCP Index', 'SPGCSIP Index', 'SPGCCNP Index', 'SPGCWHP Index', 'SPGCKCP Index',
               'SPGCSBP Index', 'SPGCILP Index', 'SPGCIKP Index', 'SPGCIAP Index', 'SPGCICP Index', 'SPGCIZP Index']

    # Plot original data for each commodity
    for ticker in tickers:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df[ticker], label=ticker)
    
        plt.title(f'Original Data for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# Replace 'commodity_data.csv' with the actual path to your CSV file
csv_file_path = 'commodity_data.csv'

result = data_validation(csv_file_path)

if type(result) is str:
    print(f"Data issue detected: {result}")
else:
    print("Data is clean. Proceed with further analysis.")
    plot_original_data(result)
