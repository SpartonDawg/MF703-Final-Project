# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:07:35 2023

@author: Hanbo Yu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)


simulated_daily_returns = np.random.normal(0, 0.01, 252)

df_returns = pd.DataFrame(simulated_daily_returns, columns=['Daily Returns'])

df_returns['Cumulative Returns'] = (1 + df_returns['Daily Returns']).cumprod()


volatility = df_returns['Daily Returns'].std()

sharpe_ratio = df_returns['Daily Returns'].mean() / volatility

roll_max = df_returns['Cumulative Returns'].cummax()
daily_drawdown = df_returns['Cumulative Returns']/roll_max - 1.0
max_drawdown = daily_drawdown.min()

var_95 = np.percentile(df_returns['Daily Returns'], 5)

cvar_95 = df_returns[df_returns['Daily Returns'] <= var_95]['Daily Returns'].mean()

risk_metrics = {
    'Volatility': volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Maximum Drawdown': max_drawdown,
    'Value at Risk (95%)': var_95,
    'Conditional VaR (95%)': cvar_95
}

plt.figure(figsize=(10, 6))
plt.plot(df_returns['Cumulative Returns'], label='Cumulative Returns')
plt.title('Simulated Strategy Cumulative Returns')
plt.xlabel('Days')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

risk_metrics

def plot_histogram(returns, title, bins=30):
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=bins, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(drawdown_series, title):
    plt.figure(figsize=(10, 6))
    drawdown_series.plot(color='red')
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_var(returns, var_value, title):
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=var_value, color='red', linestyle='--', label=f'VaR 95%: {var_value:.2%}')
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_histogram(df_returns['Daily Returns'], 'Histogram of Daily Returns')

plot_drawdown(daily_drawdown, 'Strategy Drawdown')

plot_var(df_returns['Daily Returns'], var_95, 'Value at Risk (95%)')
