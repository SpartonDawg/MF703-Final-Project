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

index benchmark
#### Trading Strategy: 
The main trading strategy is based off the AQR momentum trading strategy. This approach involves calculating the trailing 12 month return of all the commodities in our universe for the past year and ranking these trailing returns at every point in time. Then long the best ranking and short the worse.

#### Weight Optimization:

#### Backtest Analysis:


## Running Code
