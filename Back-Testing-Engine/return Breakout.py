import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df_returns_breakout = pd.read_csv(r"C:\Users\bobyu\OneDrive\Desktop\MF703 Project\Breakout Strategy Backtest\Breakout Strategy Backtest\portfolio_returns_covar_data_Breakout.csv", index_col='Date', parse_dates=['Date'])

df_returns_breakout['Cumulative Returns'] = (1 + df_returns_breakout['Portfolio Return']).cumprod()

volatility_breakout = df_returns_breakout['Portfolio Return'].std()
sharpe_ratio_breakout = df_returns_breakout['Portfolio Return'].mean() / volatility_breakout
roll_max_breakout = df_returns_breakout['Cumulative Returns'].cummax()
daily_drawdown_breakout = df_returns_breakout['Cumulative Returns']/roll_max_breakout - 1.0
max_drawdown_breakout = daily_drawdown_breakout.min()
var_95_breakout = np.percentile(df_returns_breakout['Portfolio Return'], 5)
cvar_95_breakout = df_returns_breakout[df_returns_breakout['Portfolio Return'] <= var_95_breakout]['Portfolio Return'].mean()

risk_metrics_breakout = {
    'Volatility': volatility_breakout,
    'Sharpe Ratio': sharpe_ratio_breakout,
    'Maximum Drawdown': max_drawdown_breakout,
    'Value at Risk (95%)': var_95_breakout,
    'Conditional Value at Risk (95%)': cvar_95_breakout
}

plt.figure(figsize=(14, 7))
plt.plot(df_returns_breakout['Cumulative Returns'], label='Cumulative Returns', color='blue')
plt.title('Cumulative Returns Over Time (Breakout Strategy)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_returns_breakout.png')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(daily_drawdown_breakout, label='Daily Drawdown', color='red')
plt.title('Drawdown Over Time (Breakout Strategy)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Adds a reference line at y=0
plt.grid(True)
plt.tight_layout()
plt.savefig('drawdown_breakout.png')
plt.show()

print('Risk Metrics for Breakout Strategy:')
print(risk_metrics_breakout)


