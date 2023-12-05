import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df_returns = pd.read_csv(r"C:\Users\bobyu\OneDrive\Desktop\MF703 Project\AQR Strategy Backtest\AQR Strategy Backtest\portfolio_returns_covar_data_AQR.csv", index_col='Date', parse_dates=['Date'])

df_returns['Cumulative Returns'] = (1 + df_returns['Portfolio Return']).cumprod()

volatility = df_returns['Portfolio Return'].std()
sharpe_ratio = df_returns['Portfolio Return'].mean() / volatility
roll_max = df_returns['Cumulative Returns'].cummax()
daily_drawdown = df_returns['Cumulative Returns']/roll_max - 1.0
max_drawdown = daily_drawdown.min()
var_95 = np.percentile(df_returns['Portfolio Return'], 5)
cvar_95 = df_returns[df_returns['Portfolio Return'] <= var_95]['Portfolio Return'].mean()

risk_metrics = {
    'Volatility': volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Maximum Drawdown': max_drawdown,
    'Value at Risk (95%)': var_95,
    'Conditional Value at Risk (95%)': cvar_95
}

plt.figure(figsize=(14, 7))
plt.plot(df_returns['Cumulative Returns'], label='Cumulative Returns', color='blue')
plt.title('Cumulative Returns Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_returns.png')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(daily_drawdown.index, daily_drawdown.values, label='Daily Drawdown', color='red', linewidth=1.5)
plt.title('Drawdown Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  
plt.grid(True)
plt.tight_layout()
plt.savefig('enhanced_drawdown.png')
plt.show()

print('Risk Metrics for AQR Strategy:')
print(risk_metrics)
