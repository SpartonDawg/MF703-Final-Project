#MF703 Final Project
#CovarianceMarketNeutralStrategy
#Author: Kevin & Tim 

# Summary:
# Synthetic Index (To be completed...)
# Minimize f(w_1, w_2, ..., w_n) = a * ß_portfolio^2 + var(portfolio)
    # B_portfolio = ∑ wi * ß_i = <ß,wi> = (∑cov(wiRi, R_mkt) / (var(R_mkt)))
    # var(portfolio) = π wi * var(commodity_i) + ∑ wi * wj * cov(commodity_i, commodity_j) = <w,Cw> = w'Cw
    # a : scale (to equalize the importance of ß^2 + var(portfolio))
# ßi = cov(synthetic index, commodity_i) / sigma(commodity_i) retrievable via regression
# Goal: find weights (wi) such that a*ß(port)^2 + Var(portfolio) is minimized given constraint: ∑abs(wi) = 1

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize 

# Artificial Testing Data 
test_ret = pd.DataFrame(np.random.randn(25, 19), 
                        index = pd.date_range('2023-01-01', periods = 25), 
                        columns = [f"Commodity {i}" for i in range(1, 20)])

test_strat = pd.DataFrame(np.random.randint(-1, 2, size= (25, 19)), 
                        index = pd.date_range('2023-01-01', periods = 25), 
                        columns = [f"Commodity {i}" for i in range(1, 20)])

test_syn_index = pd.DataFrame(np.random.randn(25,1), 
                        index = pd.date_range('2023-01-01', periods = 25), 
                        columns = ["Commodity Index"])

test_date = '2023-1-22'

class WeightOptimization:
    
    def __init__(self, df_syn_index, df_ret, df_strategy):
        self.df_syn_index = df_syn_index
        self.df_ret = df_ret
        self.df_strategy = df_strategy
        self.num_securities = len(df_ret.columns)
        
    def slicer(self, date):
        """slices df_ret to output TTM data for commodities with non-zero positions"""
        sliced = pd.DataFrame(index = self.df_ret.index)
        date_index = self.df_ret.index.get_loc(date)
        
        # Only considering commodities with non-zero positions
        for column in self.df_strategy.columns:
            if self.df_strategy.loc[date, column] != 0:
                sliced[column] = self.df_ret[column]
        
        # Taking only TTM data of any given commoditiy 
        start_index = max(0, date_index - 251)
        end_index = date_index + 1
        sliced = sliced.iloc[start_index : end_index]
        
        return sliced
    
    def get_covar(self, date):
        """outputs TTM covariance matrix for commodities with non-zero positions"""
        sliced_data = self.slicer(date)
        cov_matrix = sliced_data.cov()
        
        return cov_matrix
        
    def get_betas(self, date):
        """outputs TTM ßs for commodities with non-zero positions"""
        sliced_data = self.slicer(date)
        sliced_index = self.df_syn_index.loc[sliced_data.index] 
        betas = []
        
        X = sliced_index.values.reshape(-1,1)
        
        # regressing each non-zero position commodity on synthetic index to acquire ß
        for column in sliced_data.columns:
            y = sliced_data[column].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X,y)
            
            betas.append(model.coef_[0][0])
            
        return pd.Series(betas, index = sliced_data.columns)
        
    def calculate_weights(self, date, alpha=100):
        """outputs appropriate weights given minimisation function"""
        #get today's strategy
        strat_today = self.df_strategy.loc[date].values
        #remove 0s
        positions = [v for v in strat_today if v != 0]
        
        #get covariance and correlation to market of the individual commodities
        C = self.get_covar(date)
        betas = self.get_betas(date)
        
        def objective_function(w):
           term1 = alpha * (w @ np.transpose(betas))**2
           term2 = w @ C @ np.transpose(betas)
           return term1 + term2

        # Create a function to generate constraints based on the strategy
        def generate_constraints(strategy):
            constraints = []
            for i, s in enumerate(strategy):
                if s == 1:
                    constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: weights[i]})
                elif s == -1:
                    constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: -weights[i]})
            constraints.append({'type': 'eq', 'fun': lambda weights: np.abs(weights).sum() - 1})
            return constraints
            
        # Initial guess for the weights
        initial_guess = [1/len(positions) * x for x in positions]
 
        # Generate constraints based on the strategy
        constraints = generate_constraints(positions)
 
        # Perform the optimization
        result = minimize(fun=objective_function, x0=initial_guess, constraints=constraints)
        if result.success:
            optimized = result.x
        else:
            raise ValueError("Optimization failed.")
            
        final_weights = np.zeros(self.num_securities)
        index = 0
        for i in range(len(strat_today)):
            if strat_today[i] != 0:
                final_weights[i] = optimized[index]
                index += 1
                
        return final_weights
            
# Testing:
weight_optimizer = WeightOptimization(test_syn_index, test_ret, test_strat)

sliced_data = weight_optimizer.slicer(test_date)
covar_mat = weight_optimizer.get_covar(test_date)
b = weight_optimizer.get_betas(test_date)
w = weight_optimizer.calculate_weights(test_date, alpha=50)

print("TTM Covariance Matrix:")
print(covar_mat, "\n")
print("TTM Commodity Betas:")
print(b, "\n")
print("Strategies:")
print(w, "\n")

      
      
