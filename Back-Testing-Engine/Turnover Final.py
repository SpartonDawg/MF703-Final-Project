import pandas as pd
import matplotlib.pyplot as plt

covar_weights = pd.read_csv(r"C:\Users\bobyu\OneDrive\Desktop\MF703 Project\AQR Strategy Backtest\AQR Strategy Backtest\position_weights_covar_data_AQR.csv", parse_dates=['Date'])
covar_weights.set_index('Date', inplace=True)

def calculate_turnover(data):
    daily_turnover = data.diff().abs().sum(axis=1)

    annualized_turnover = daily_turnover.mean() * 252

    yearly_turnover = daily_turnover.resample('Y').sum()

    return annualized_turnover, yearly_turnover

annualized_turnover_covar, yearly_turnover_covar = calculate_turnover(covar_weights)

fig, ax = plt.subplots()
yearly_turnover_covar.plot(kind='bar', ax=ax)
ax.set_title('Year by Year Portfolio Turnover (Covar Weight Data)')
ax.set_xlabel('Year')
ax.set_ylabel('Turnover')

ax.set_xticklabels([date.strftime('%Y') for date in yearly_turnover_covar.index])

plt.show()

print("Annualized Turnover (Covar Weight Data):", annualized_turnover_covar)
