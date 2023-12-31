# Market Neutral Commodity Trading Strategy
Group 3: Jack Bartoluzzi, Jim Burrill, Tim Lee, Sagar Mirchandani, Kevin Yang, Hanbo Yu, Hamza Zbiri

Inspired by the J.P. Morgan Optimax Market-Neutral Index, we set out with the goal of developing an investment strategy that seeks to generate consistent returns through a selection of commodity-linked component sub-indices. This required the identification of a L/S strategy to generate trading signals based on commodity price movements, and a portfolio optimization technique to minimize portfolio beta and variance for the generated trading signals. To validate the identified investment strategy an extensive back test was required.

## Introduction

### The project is broken into 4 main categories.

- Data Collection, Cleaning and Verifying
- Trading Strategy
- Weight Optimization
- Backtest Analysis

#### Data:
We used data on 16 commodity indicies from the S&P GSCI Index via the Bloomberg terminal. The S&P GSCI is a widely recognized commodity index provider and there was good data availability through time; dating back to 1991.

#### Trading Strategy: 
The main trading strategy is based off the AQR momentum trading strategy. This approach involves calculating the trailing 12 month return of all the commodities in our universe for the past year and ranking these trailing returns at every point in time. Then long the best ranking and short the worse.

#### Weight Optimization:
See the "Weight_Optimzer.py" file for the WeightOptimizer class. This class is initialized with a market index, prices data, and historical long-short signals. Important methods include the calculate_weights function, which takes in a date and an alpha and outputes the optimal weights that minimize portfolio beta and variance, and the plot_frontier function, which plots the set of optimal portfolios as alpha varies on the beta-squared variance space.

#### Backtest Analysis:
A main part of the project was learning how to avoid forward looking bias when performing a backtest. Since we used daily close data the hyptothetical workflow would be we observe the data after the close, run the algorithm to decide what positions and weights to take and get filled at prices the next close. If there are no position changes then we rebalance weekly but that is a parameter in the backtest.

## Running Code
To run the code that applies the trading strategy, weight optimizer, and backtesting function look to the folder: Final Project Package/Running Code
You should only need the 3 files to run, which can be found in the folder or in the zip combined.
#### Backtest_main.py - The Python File to run
#### Weight_Optimizer.py - Backtest_main imports the class from this file
#### S&P Commodity Data (with BCOM).xlsx - Input price data

Backtest_main.py takes around 5 minutes to run. 
