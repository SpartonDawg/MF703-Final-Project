# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:05:55 2023

@author: Hanbo Yu
"""

class Portfolio:
    def __init__(self):
        self.transactions = []
        self.average_assets = 0

    def add_transaction(self, amount):
        # A positive amount represents a buy, a negative amount represents a sell
        self.transactions.append(amount)

    def set_average_assets(self, average_assets):
        # Set the average asset value of the portfolio for the period
        self.average_assets = average_assets

    def calculate_turnover(self):
        # Calculate the total buys and sells
        total_buys = sum(t for t in self.transactions if t > 0)
        total_sells = sum(abs(t) for t in self.transactions if t < 0)
        
        # The portfolio turnover is calculated as the minimum of total buys or sells divided by the average assets
        turnover = min(total_buys, total_sells) / self.average_assets
        
        # Convert to percentage
        turnover_rate = turnover * 100
        
        # Reset transactions for the next period
        self.transactions = []
        
        return turnover_rate

# Example usage
portfolio = Portfolio()
portfolio.set_average_assets(10000000)  # Example value for average total assets
portfolio.add_transaction(500000)  # Example buy transaction
portfolio.add_transaction(-300000)  # Example sell transaction

# Calculate the turnover rate for the period
turnover_rate = portfolio.calculate_turnover()
print(f"Portfolio Turnover Rate: {turnover_rate:.2f}%")
