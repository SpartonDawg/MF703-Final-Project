def calc_monthly_returns(df):
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

            monthly_i_tr = (1+temp_monthly_df['Portfolio Return']).prod()-1
            months.append(month_i)
            years.append(year_i)
            returns.append(monthly_i_tr)
            date.append(temp_monthly_df['Date'].iloc[0])
    return pd.DataFrame({"Year":years,"Months":months,"Monthly Return":returns})
