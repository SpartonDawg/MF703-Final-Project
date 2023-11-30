import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

def Generate_Weights_Grid(num_parameters, max_value, increment, min_value=0.05, tolerance=1e-5):
    max_value = min(max_value, num_parameters * increment)
    grid = np.arange(0, max_value + increment, increment)
    combinations = np.array(list(product(grid, repeat=num_parameters)))
    sums = combinations.sum(axis=1)
    valid_indices = np.isclose(sums, 1, rtol=tolerance)
    valid_combinations = combinations[valid_indices]
    valid_combinations = np.maximum(valid_combinations, min_value)

    return valid_combinations

#Constraints, this creates 13,140 combos
num_parameters = 8
max_value = 0.4
increment = 0.1
min_value = 0.05
tolerance = .00001


weight_combos = Generate_Weights_Grid(num_parameters, max_value, increment, min_value, tolerance)
weight_df = pd.DataFrame(weight_combos)
weights_grid = weight_df.div(weight_df.sum(axis=1), axis=0).to_numpy()


def Minimize_Correlation_Weights(current_date, daily_returns_df, positions_df, weights_grid):
    """
    Inputs: {
        current_date: current date index as pd.datetime object
        daily_returns_df: dataframe of only daily returns (t - t-1), pd.datetime is index, columns are just commodity names
        positions_df: dataframe of only position columns, pd.datetime is index, columns are just commodity names
        weights_grid: np array of arrays, spanning weights combination that sum to 1, NOTE: assumes 8 constant positions, has certain contraints in function
    }
    Output:
        Tuple: list of weights thats minimize portfolio pairwise correlation, corresponding column names (standard names), the minimized portfolio correlation value
    """

    lower_date_cutoff = current_date - pd.DateOffset(years=1)
    daily_returns_1YR_df = daily_returns_df.loc[lower_date_cutoff:current_date]
    positions_1YR_df = positions_df.loc[lower_date_cutoff:current_date]

    current_positions = positions_1YR_df.loc[current_date][positions_1YR_df.loc[current_date] != 0]
    current_positions_names = current_positions.index.values

    correlation_matrix = daily_returns_1YR_df[current_positions_names]*current_positions.values # Scaling correlation by direction L/S
    correlation_matrix = correlation_matrix.corr().to_numpy()
    np.fill_diagonal(correlation_matrix,0)
    best_weights = []
    best_port_correlation = 100

    if len(current_positions) != 0:
        for weights_i in weights_grid:

            weights_i_matrix = np.outer(weights_i, weights_i)
            total_port_correlation = np.sum(correlation_matrix*weights_i_matrix)

            if abs(total_port_correlation) < best_port_correlation:
                best_port_correlation = total_port_correlation
                best_weights = weights_i
    else:
        weighted_returns.append(0)

    return best_weights,current_positions_names,best_port_correlation




#
