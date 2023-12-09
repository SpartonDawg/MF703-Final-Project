import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import math
from scipy.stats import norm
import time
pd.options.mode.chained_assignment = None  # default='warn'



"""
X: Portfolio Returns
Y: Benchmark Returns
"""


def Annualized_Return(X):

    try:
        X = X+1
        return ( (np.prod(X)**(1 / (len(X)/12))) -1)
    except:
        return np.nan

def Annualized_RF(RF):
    try:
        Amount = 100
        counter_i = 0
        Quarter_Returns = []
        for i in range(int(np.ceil(len(RF)/3))):
            i = i*3
            Quarter_Returns.append(1+((RF.iloc[i]/100)/(365/91)))
            counter_i = i
        rf_return =  ((np.prod(Quarter_Returns)**(4/(len(Quarter_Returns))))-1)
        return rf_return

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
        return (np.mean(X)-(Beta_Calc(X,Y)*np.mean(Y)))*12
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
        return ((Annualized_Return(X) - .024) / (STD_Calc(X)))
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

def calc_monthly_returns(df,return_col,output_col):
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year

    date = []
    months = []
    years = []
    returns = []
    for year_i in df['Year'].unique():
        temp_df = df[df['Year'] == year_i]
        for month_i in temp_df['Month'].unique():
            temp_monthly_df = temp_df[temp_df['Month'] == month_i]

            monthly_i_tr = (1+temp_monthly_df[return_col]).prod()-1
            months.append(month_i)
            years.append(year_i)
            returns.append(monthly_i_tr)
            date.append(temp_monthly_df['Date'].iloc[0])
    return pd.DataFrame({"Year":years,"Month":months,output_col:returns})


benchmark_df = pd.read_excel("S&P Commodity Data (with BCOM).xlsx")
benchmark_df = benchmark_df[['Date','BCOM Index']]
benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
benchmark_df['BCOM Return'] = benchmark_df['BCOM Index'].pct_change().fillna(0)

covar_opt_portfolio_returns_df = pd.read_csv(r"portfolio_returns_covar_data_breakout.csv")
covar_opt_portfolio_returns_df['Date'] = pd.to_datetime(covar_opt_portfolio_returns_df['Date'])
covar_opt_portfolio_returns_df = covar_opt_portfolio_returns_df.iloc[:,1:]
print(covar_opt_portfolio_returns_df)

equal_portfolio_returns_df = pd.read_csv(r"portfolio_returns_equal_data_breakout.csv")
equal_portfolio_returns_df['Date'] = pd.to_datetime(equal_portfolio_returns_df['Date'])
equal_portfolio_returns_df = equal_portfolio_returns_df.iloc[:,1:]
print(equal_portfolio_returns_df)


monthly_portfolio_covar = calc_monthly_returns(covar_opt_portfolio_returns_df,'Portfolio Return','Monthly Portfolio Return (Covar Optimal)')
monthly_portfolio_equal = calc_monthly_returns(equal_portfolio_returns_df,'Portfolio Return','Monthly Portfolio Return (Equal Weight)')
monthly_portfolio_df = monthly_portfolio_equal.merge(monthly_portfolio_covar)
monthly_benchmark = calc_monthly_returns(benchmark_df,'BCOM Return','Monthly BCOM Index Return')
monthly_returns = monthly_benchmark.merge(monthly_portfolio_df, how = 'inner', on = ['Year', 'Month'])
print(monthly_returns)

portfolio_returns_covar = monthly_returns['Monthly Portfolio Return (Covar Optimal)']
portfolio_returns_equal = monthly_returns['Monthly Portfolio Return (Equal Weight)']
benchmark_returns_series = monthly_returns['Monthly BCOM Index Return']


"""
Plotting
"""
colors = ['navy', 'blue', 'royalblue']

plt.style.use("seaborn-talk")
plt.figure(figsize=(11,5))
std_covar = STD_Calc(portfolio_returns_covar)
std_equal = STD_Calc(portfolio_returns_equal)
std_benchmark = STD_Calc(benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [std_covar,std_equal,std_benchmark]
plt.bar(x_labels,y_values,color=colors)
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('(Annualized Monthly Returns: 1992-2023)',fontsize=14, y=1)
plt.suptitle("Standard Deviation - Breakout Strategy", size=22)

plt.ylabel("Standard Deviation",size=16)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val*100,2))+"%",(temp_x_val,temp_y_val+.001),size=12,ha='center')
plt.show()


plt.style.use("seaborn-talk")
plt.figure(figsize=(11,5))
annual_ret_covar = Annualized_Return(portfolio_returns_covar)
annual_ret_equal = Annualized_Return(portfolio_returns_equal)
annual_ret_benchmark = Annualized_Return(benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [annual_ret_covar,annual_ret_equal,annual_ret_benchmark]
plt.bar(x_labels,y_values,color=colors)

plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
plt.suptitle("Annualized Total Return - Breakout Strategy", size=22)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val*100,2))+"%",(temp_x_val,temp_y_val+.0002),size=12,ha='center')
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.show()

plt.style.use("seaborn-talk")
plt.figure(figsize=(11,5))
annual_ret_covar = Sharpe_Calc(portfolio_returns_covar,np.array([0]*len(portfolio_returns_covar)))
annual_ret_equal = Sharpe_Calc(portfolio_returns_equal,np.array([0]*len(portfolio_returns_covar)))
annual_ret_benchmark = Sharpe_Calc(benchmark_returns_series,np.array([0]*len(portfolio_returns_covar)))
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [annual_ret_covar,annual_ret_equal,annual_ret_benchmark]
plt.bar(x_labels,y_values,color=colors)
plt.ylabel("Sharpe Ratio")
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
plt.suptitle("Sharpe Ratio - Breakout Strategy", size=22)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val,2)),(temp_x_val,temp_y_val-.01),size=12,ha='center')

plt.show()

plt.style.use("seaborn-talk")
plt.figure(figsize=(11,10))
drawdown_covar = Max_DrawDown_Calc(portfolio_returns_covar)
drawdown_equal = Max_DrawDown_Calc(portfolio_returns_equal)
drawdown_benchmark = Max_DrawDown_Calc(benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [drawdown_covar,drawdown_equal,drawdown_benchmark]
plt.bar(x_labels,y_values,color=colors)
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.suptitle("Max Drawdown - Breakout Strategy", size=28)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
plt.ylabel("Max Drawdown",size=16)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val*100,2))+"%",(temp_x_val,temp_y_val-.025),size=15,ha='center')
plt.show()


plt.style.use("seaborn-talk")
plt.figure(figsize=(11,10))
VaR_covar = VaR(portfolio_returns_covar)
VaR_equal = VaR(portfolio_returns_equal)
VaR_benchmark = VaR(benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [VaR_covar,VaR_equal,VaR_benchmark]
plt.bar(x_labels,y_values,color=colors)
plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.suptitle("VaR - 5% - Breakout Strategy", size=28)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val*100,2))+"%",(temp_x_val,temp_y_val-.0027),size=15,ha='center')
plt.show()


plt.style.use("seaborn-talk")
plt.figure(figsize=(11,10))
Beta_covar = Beta_Calc(portfolio_returns_covar,benchmark_returns_series)
Beta_equal = Beta_Calc(portfolio_returns_equal,benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted']
y_values = [Beta_covar,Beta_equal]
plt.bar(x_labels,y_values,color=colors)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.suptitle("Beta - Breakout Strategy", size=28)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val,3)),(temp_x_val,temp_y_val-.005),size=15,ha='center')
plt.show()


plt.style.use("seaborn-talk")
plt.figure(figsize=(11,10))
alpha_covar = Alpha_Calc(portfolio_returns_covar,benchmark_returns_series)
alpha_equal = Alpha_Calc(portfolio_returns_equal,benchmark_returns_series)
x_labels = ['Covariance Optimized Weights', 'Equal Weighted']
y_values = [alpha_covar,alpha_equal]
plt.bar(x_labels,y_values,color=colors)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.suptitle("Alpha - Breakout Strategy", size=28)
plt.title('(Monthly Returns: 1992-2023)',fontsize=14, y=1)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val,3)),(temp_x_val,temp_y_val+.001),size=15,ha='center')
plt.show()




plt.style.use("seaborn-talk")
plt.figure(figsize=(11,10))
drawdown_covar = Sharpe_Calc(portfolio_returns_covar,np.array([0]*len(portfolio_returns_covar)))
drawdown_equal =  Sharpe_Calc(portfolio_returns_equal,np.array([0]*len(portfolio_returns_equal)))
drawdown_benchmark =  Sharpe_Calc(benchmark_returns_series,np.array([0]*len(benchmark_returns_series)))
x_labels = ['Covariance Optimized Weights', 'Equal Weighted', "BCOM Index"]
y_values = [drawdown_covar,drawdown_equal,drawdown_benchmark]
plt.bar(x_labels,y_values,color=colors)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
# plt.title("Sharpe Ratio - (Monthly Returns: 1992-2023)", size=22)
plt.suptitle("Sharpe Ratio - Breakout Strategy", size=28)
plt.title('(Annualized Monthly Returns: 1992-2023)',fontsize=14, y=1)
plt.ylabel("Sharpe Ratio",size=16)
for i in range(len(x_labels)):
    temp_y_val= y_values[i]
    temp_x_val = x_labels[i]
    plt.annotate(str(round(temp_y_val,2)),(temp_x_val,temp_y_val+.002),size=15,ha='center')
plt.show()
#
