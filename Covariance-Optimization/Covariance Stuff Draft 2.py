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
# Goal: find weights (wi) such that a*ß(port)^2 + Var(portfolio) is minimized given constraint: ∑abs(w_i) = 1

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

class WeightOptimization:
    
    def __init__(self, df_syn_index, df_ret, df_strategy):
        self.df_syn_index = df_syn_index.ffill() # pandas Series indexed by date containing the index returns on that date
        self.df_ret = df_ret # dataframe containing daily returns. Indexed by date and with columns labeled by commodities
        self.df_strategy = df_strategy # dataframe containing -1,0,1. indexed by date and with columns labeled by commodities
        self.num_securities = len(df_ret.columns)
        
    def slicer(self, date, numdays=252):
        """
        date: datetime object
        slices df_ret to output recent data for commodities with non-zero positions.
        Default length is one year
        """
        sliced = pd.DataFrame(index = self.df_ret.index)
        date_index = self.df_ret.index.get_loc(date)
        
        # Only considering commodities with non-zero positions
        for column in self.df_strategy.columns:
            if self.df_strategy.loc[date, column] != 0:
                sliced[column] = self.df_ret[column]
        
        # Taking only TTM data of any given commoditiy 
        start_index = max(0, date_index - numdays + 1)
        end_index = date_index + 1
        sliced = sliced.iloc[start_index : end_index]
        
        return sliced
    
    def get_covar(self, date, numdays=252):
        """
        date: datetime object
        Computes the covariance matrix for commodities with non-zero positions and outputs it as a dataFrame.
        Default lookback length is one year
        
        """
        sliced_data = self.slicer(date, numdays)
        cov_matrix = sliced_data.cov()
        
        return cov_matrix
        
    def get_betas(self, date, numdays=252):
        """
        computes the TTM ßs for commodities with non-zero positions and outputs it as a pandas Series
        default lookback length is one year.
        """
        sliced_data = self.slicer(date, numdays)
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
        """
        date:datetime object
        computes the optimal weights for a certain date and risk preference alpha
        outputs it as a numpy array with length equal to the number of commodities in the universe
        """
        #get today's strategy
        strat_today = self.df_strategy.loc[date].values
        
        #remove 0s
        positions = [v for v in strat_today if v != 0]
        
        if len(positions) == 0:
            return [0]*self.num_securities
        
        #get covariance and correlation to market of the individual commodities
        C = self.get_covar(date)
        betas = self.get_betas(date)
        
        #define quantity to be minimized
        def objective_function(w):
           term1 = alpha * (w @ np.transpose(betas))**2
           term2 = w @ C @ np.transpose(w)
           return term1 + term2

        # Create a function to generate constraints based on the strategy
        def generate_constraints(strategy):
            constraints = []
            
            #direction constraints: weights must be positive or negative as dictated by strategy
            for i, s in enumerate(strategy):
                if s == 1:
                    constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: weights[i]})
                elif s == -1:
                    constraints.append({'type': 'ineq', 'fun': lambda weights, i=i: -weights[i]})
                    
            # leverage constraint: absolute values of weights add to 1
            constraints.append({'type': 'eq', 'fun': lambda weights: np.abs(weights).sum() - 1})
            return constraints
            
        # Initial guess for the weights
        #initial_guess = [1/len(positions) * x for x in positions]
        initial_guess = [1*positions[0]] + [0]*(len(positions)-1)
 
        # Generate constraints based on the strategy
        constraints = generate_constraints(positions)
 
        # Perform the optimization
        result = minimize(fun=objective_function, x0=initial_guess, constraints=constraints)
        if result.success:
            optimized = result.x # store the optimal weights
        else:
            raise ValueError("Optimization failed.")
        
        #re-insert the zero weights for all the commodities with position 0
        final_weights = np.zeros(self.num_securities)
        index = 0
        for i in range(len(strat_today)):
            if strat_today[i] != 0:
                final_weights[i] = optimized[index]
                index += 1
        
        #return final result as an array with length num_securities
        return final_weights
    
    def plot_frontier(self, date):
        """
        Given a certain date, plots the maximum n.eutrality frontier: the optimum correlation to market vs portfolio variance curve as alpha varies from 1 to 200
        """
        
        C = self.get_covar(date)
        betas = self.get_betas(date)
        
        x = np.empty(1999)
        y = np.empty(1999)
        
        for a in range(1, 200):
            weights = self.calculate_weights(date, a).values
            w = [v for v in weights if v != 0]
            beta_squared = (w @ np.transpose(betas))**2
            port_var = w @ C @ np.transpose(w)
        
            x[a-1] = port_var
            y[a-1] = beta_squared
        
        plt.scatter(x,y)
        plt.xlabel("portfolio variance")
        plt.ylabel('portfolio beta to market squared')
        plt.title('minimum beta variance frontier')
        
if __name__ == '__main__':
    
    # Artificial Testing Data 
    """
    test_strat = pd.DataFrame(np.random.randint(-1, 2, size= (25, 19)), 
                            index = pd.date_range('2023-01-01', periods = 25), 
                            columns = [f"Commodity {i}" for i in range(1, 20)])
    """
    test_syn_index = pd.Series(np.random.randn(5000), 
                            index = pd.date_range('2010-1-1', periods = 5000), 
                            name = "Commodity Index")
    test_syn_index.index = pd.to_datetime(test_syn_index.index)   
    
    data = pd.read_excel("Cleaned_Data_stripped.xlsx", index_col = 0)
    data.index = pd.to_datetime(data.index)
    data = data.fillna(method = 'bfill')
    returns = data.pct_change()
    
    start_date = '2016-01-01'
    end_date = '2017-12-31'
    
    """
    test_ret = pd.DataFrame(np.random.randn(5000, 15), 
                            index = pd.date_range('2005-01-01', periods = 5000), 
                            columns = [returns.columns])
    test_ret.index = pd.to_datetime(test_ret.index)
    """
    strategy = pd.read_csv('positions_with_timeindex.csv', index_col = 0)
    strategy.index = pd.to_datetime(strategy.index)
    filtered_strategy = strategy[(strategy.index >= start_date) & (strategy.index <= end_date)]
    # Testing:
    WeightOptimizer = WeightOptimization(test_syn_index, returns, filtered_strategy)
    
    historical_weights = pd.DataFrame(index = filtered_strategy.index, columns = filtered_strategy.columns)
    #equal_weights = pd.DataFrame(index = filtered_strategy.index, columns = filtered_strategy.columns)
      
    for i in range(len(filtered_strategy.index)):
        date = filtered_strategy.index[i]        
        strat_today = filtered_strategy.iloc[i]
        weights = WeightOptimizer.calculate_weights(date, alpha=100)
        
        positions = [v for v in strat_today if v != 0]
        #eq_weights = [1/len(positions) * x for x in filtered_strategy.iloc[2]]
        
        print(weights)
        historical_weights.loc[date] = weights
        #equal_weights.loc[date] = eq_weights
        
    #weight_optimizer.plot_frontier(test_date) 
    
    #print("TTM Covariance Matrix:")
    #print(covar_mat, "\n")
    #print("TTM Commodity Betas:")
    #print(b, "\n")
    #print("Strategy:")
    #print(strategy.loc[].values, "\n")
    #print("Weights:")
    #print(weights, "\n")
    
    port=[1]*(len(historical_weights.index)+1)
    #equal_weighted_port = [1]*(len(historical_weights.index)+1)
    
    forward_return = returns.shift(-1)
    
    for i in range(len(historical_weights.index)):
        date = historical_weights.index[i]
        daily_return = np.dot(historical_weights.loc[date], forward_return.loc[date])
        #equal_weighted_return = np.dot(equal_weights.loc[date], forward_return.loc[date])
        port[i+1] = port[i] * (1+daily_return)
        #equal_weighted_port[i+1] = equal_weighted_port[i] * (1+equal_weighted_return)
        
    plt.plot(port)
    plt.title('rudimentary backtest, 2016-2019 inclusive')
    plt.xlabel('days')
    plt.ylabel('return')
    ########################################
    