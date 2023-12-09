import pandas as pd
import numpy as np

"""
X: Portfolio Returns
Y: Benchmark Returns
RF: Risk Free
"""

def Annualized_Return(X):
    try:
        X = X+1
        return ( (np.prod(X)**(1 / (len(X)/12))) -1)
    except:
        return np.nan


def TR(X):
    try:
        X = X+1
        return np.prod(X) -1
    except:
        return np.nan

def STD_Calc(X):
    try:
        X = X+1
        X_bar = X.mean()
        X_Diff = (X-X_bar)**2
        std = np.sqrt(X_Diff.sum() / (len(X) - 1))
        return (np.sqrt(12)*std)
    except:
        return np.nan


def Beta_Calc(X,Y):
    try:
        XY = X*Y
        YY = Y*Y
        N = len(X)
        beta = ((N*XY.sum()) - (X.sum()*Y.sum())) / ((N*YY.sum()) - (Y.sum()*Y.sum()))
        return beta
    except:
        return np.nan

def R_Calc(X,Y):
    try:
        x = np.array(X.astype(float))
        y = np.array(Y.astype(float))
        r = np.corrcoef(x, y)
        return r[0,1]
    except:
        return np.nan

def Alpha_Calc(X,Y):
    try:
        return (np.mean(X)-(Beta_Calc(X,Y)*np.mean(Y)))*1200
    except:
        return np.nan


def Max_DrawDown_Calc(X):
    try:
        X = X.values
        CumulativeX = [100]
        for i in X:
            CumulativeX.append((1+i)*CumulativeX[-1])
        max_draw = 0
        for i in range(len(CumulativeX)):
            if i == 0:
                pass
            else:
                temp_md = (CumulativeX[i] - np.max(CumulativeX[:i]))  / np.max(CumulativeX[:i])
                if temp_md < max_draw:
                    max_draw = temp_md
        return abs(max_draw)*-1
    except:
        return np.nan

def Sharpe_Calc(X,RF):
    try:
        return ((Annualized_Return(X) - Annualized_RF(RF)) / (STD_Calc(X)))*100
    except:
        return np.nan

def Tracking_Error_Calc(X,Y):
    try:
        Z = X-Y
        return STD_Calc(Z)
    except:
        return np.nan

def Information_Ratio_Calc(X,Y):
    try:
        excess = Annualized_Return(X) - Annualized_Return(Y)
        return (excess / Tracking_Error_Calc(X,Y))
    except:
        return np.nan


def VaR(X):
    mean = np.mean(X)
    std = np.std(X)
    return norm.ppf(1-0.950, mean, std)
