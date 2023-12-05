import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df_returns_equal = pd.read_csv(r"C:\Users\bobyu\OneDrive\Desktop\MF703 Project\AQR Strategy Backtest\AQR Strategy Backtest\portfolio_returns_equal_data_AQR.csv", index_col='Date', parse_dates=['Date'])

df_returns_equal['Cumulative Returns'] = (1 + df_returns_equal['Portfolio Return']).cumprod()

volatility_equal = df_returns_equal['Portfolio Return'].std()
sharpe_ratio_equal = df_returns_equal['Portfolio Return'].mean() / volatility_equal
roll_max_equal = df_returns_equal['Cumulative Returns'].cummax()
daily_drawdown_equal = df_returns_equal['Cumulative Returns']/roll_max_equal - 1.0
max_drawdown_equal = daily_drawdown_equal.min()
var_95_equal = np.percentile(df_returns_equal['Portfolio Return'], 5)
cvar_95_equal = df_returns_equal[df_returns_equal['Portfolio Return'] <= var_95_equal]['Portfolio Return'].mean()

risk_metrics_equal = {
    'Volatility': volatility_equal,
    'Sharpe Ratio': sharpe_ratio_equal,
    'Maximum Drawdown': max_drawdown_equal,
    'Value at Risk (95%)': var_95_equal,
    'Conditional Value at Risk (95%)': cvar_95_equal
}

plt.figure(figsize=(14, 7))
plt.plot(df_returns_equal['Cumulative Returns'], label='Cumulative Returns', color='blue')
plt.title('Cumulative Returns Over Time (Equal Weight)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('cumulative_returns_equal.png')
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(daily_drawdown_equal, label='Daily Drawdown', color='red')
plt.title('Drawdown Over Time (Equal Weight)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Drawdown', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Adds a reference line at y=0
plt.grid(True)
plt.tight_layout()
plt.savefig('drawdown_equal.png')
plt.show()

print('Risk Metrics for Equal Weight Strategy:')
print(risk_metrics_equal)
