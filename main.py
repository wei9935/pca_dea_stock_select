import pandas as pd
from models import *
from fin_utils import *
import warnings

warnings.filterwarnings("ignore")

if __name__=='__main__':
    data = pd.read_csv('data/company_data.csv')

    # Define Inputs and Outputs.
    in_list = ['Current Ratio', 'Quick Ratio', 'Interest Expense Ratio', 
               'Debt Ratio', 'Equity to Assets', 'Long and Short-term Loans', 
               'Interest Coverage Ratio', 'Working Capital', 'Total Asset Turnover',
                 'Average Collection Days', 'Inventory Turnover (times)', 
                 'Fixed Asset Turnover', 'Free Cash Flow (D)', 'Number of Employees', 
                 'Operating Expense Ratio', 'Labor Cost Ratio', 'R&D Expense Ratio', 
                 'Bad Debt Expense Ratio', 'Operating Leverage', 'Financial Leverage', 
                 'Cash Flow Ratio', 'Interest-bearing Debt Ratio'
                 ]
    
    out_list = ['Tobins Q', 'Cash Dividend Rate', 'ROA - Comprehensive Income', 
                'ROE - Comprehensive Income', 'Operating Gross Profit Margin', 
                'Operating Gross Profit Growth Rate', 'Operating Profit Margin', 
                'Earnings Per Share', 'Dividend Payout Ratio', 'Revenue Growth Rate',
                  'Pre-tax Net Profit Margin', 'Post-tax Net Profit Margin', 
                  'Non-operating Income/Revenue', 'Return on Operating Assets', 
                  'Equity Growth Rate', 'Total Asset Growth Rate'
                  ]
    
    # deal with NaN values - insert average of previous years or average of the same sectors.
    df = data.copy()
    for col in df.columns:
        if col in ['Company Name', 'Time', 'TSE_industry_name', 'ipo_date']:
            pass
        else:
            df[col] = df.groupby('Company Name')[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(df.groupby('TSE_industry_name')[col].transform(lambda x: x.mean()))

    # start analysing sectors
    perf_dict, port_dict = {}, {}
    eff_port_values, ineff_port_values = {}, {}
    for sec in df['TSE_industry_name'].unique():# Compare Firms Within the Same Industry.
        print(f'Begin: {sec}')
        if sec in ['M2800 Financial Industry', 'W91 Custody Receipts']:
                print(f'{sec}: Pass')
                continue
        # Set up Portfolios.
        eff_port, ineff_port = my_portfolio(), my_portfolio()
        for year in df['Time'].unique():# Analyze Firms By Each Year.
            print(f'Analyzing data: {sec}-{year}')
            ind = df[(df['TSE_industry_name'] == sec)]['Company Name'].to_list()
            ind_data = df[(df['Company Name'].isin(ind)) & (df['Time'] == year)]
            ind_data = ind_data[ind_data['ipo_date']<ind_data['Time']]
    
            # PCA
            pca_data = pca_IO(data=ind_data, in_list=in_list, out_list=out_list)
            x, y = pca_data.fit(var_level='kaiser')
            x.index, y.index = ind_data['Company Name'], ind_data['Company Name']

            # DEA
            dea_result = DEA(DMUs_Name=ind_data['Company Name'], X=x, Y=y)
            result = dea_result.dea()  #dea results
            dea_result.dmus_efficiency()
            if dea_result.max_efficiency != 1 :
                print(f'{sec}-{year} No Efficient Firms.')
                continue

            # Get Efficient and Inefficient Firms. (same amount of stocks in both type)
            eff = dea_result.n_efficient_dmus
            ineff = dea_result.n_inefficient_dmus
            
            # Set Back test Trading Time.
            trading_time = [dt.datetime.strptime(time, '%Y-%m-%d') + relativedelta(days=1) for time in ind_data['Time']]
            trading_time = [dt.datetime.strftime(time, '%Y-%m-%d') for time in trading_time]
            
            # Construct Portfolios. (Equal weights for this project)
            eff_port.add_stocks(stock=eff, weight=[1/len(eff)]*len(eff), time=trading_time) # Equal Weights for efficient firms
            ineff_port.add_stocks(stock=ineff, weight=[1/len(ineff)]*len(ineff), time=trading_time)
            
            print(f'{sec}-{year} Added.')

        # Back test Portfolio Performances.
        eff_port.bt_back_test()
        ineff_port.bt_back_test()
        perf_dict[sec] = [eff_port.bt_annual_return, ineff_port.bt_annual_return] # Store Annual Returns of efficient and inefficient firms
        port_dict[sec] = [eff_port.port, ineff_port.port]
        eff_port_values[sec] = eff_port.bt_port_value
        ineff_port_values[sec] = ineff_port.bt_port_value

    # compare performance metrics
    eff_val_df, ineff_val_df = pd.DataFrame(eff_port_values).ffill(), pd.DataFrame(ineff_port_values).ffill()
    eff_val_df.index, ineff_val_df.index = eff_val_df.index.tz_localize(None), ineff_val_df.index.tz_localize(None)
    
    rf_df = pd.read_csv('data/risk_free_rate.csv', index_col=0)
    eff_perf = calculate_performances(eff_val_df, rf=rf_df)
    ineff_perf = calculate_performances(ineff_val_df, rf=rf_df)

    # Save results
    port_df = pd.DataFrame(port_dict)
    port_df.to_csv('results/portfolio_record.csv')
    for name, sub_df in port_dict.items():
        sub_df[0].to_csv(f'results/weights/{name}_eff_wts.csv', index=False)
        sub_df[1].to_csv(f'results/weights/{name}_ineff_wts.csv', index=False)

    eff_val_df.to_csv('results/efficient_port_values.csv')
    ineff_val_df.to_csv('results/inefficient_port_values.csv')
    eff_perf.to_csv('results/efficient_stocks_perf.csv')
    ineff_perf.to_csv('results/inefficient_stocks_perf.csv')

    ew_port = pd.concat([eff_val_df.mean(axis=1), ineff_val_df.mean(axis=1)], axis=1).ffill()
    ew_port.columns = ['Efficient_port_EW', 'InEfficient_port_EW']
    ew_perf = calculate_performances(ew_port, rf=rf_df)
    ew_port.to_csv('results/equal_sector_weights_values.csv')
    ew_perf.to_csv('results/equal_sector_weights_performance.csv')


    # Plot annual return comparison
    bar_width = 0.35 
    index = np.arange(len(eff_perf.index)) 
    plt.figure(figsize=(10, 6)) 
    plt.bar(index, eff_perf['Annual Return'], bar_width, label='Efficient')
    plt.bar(index + bar_width, ineff_perf['Annual Return'], bar_width, label='Inefficient')
    plt.xlabel('Sectors'), plt.ylabel('Return')
    plt.title('Comparison of Efficient and Inefficient Portfolios\' Annual Returns by sector')
    plt.xticks(index + bar_width / 2, [item.split()[0] for item in eff_perf.index], rotation = 90)

    plt.tight_layout()
    plt.legend()
    #plt.show()
    plt.savefig('results/annual_rt_compare_plot.png')


    # Resample to monthly returns by compounding the daily returns
    eff_yr_rt = (1 + ew_port.iloc[:, 0].pct_change().dropna()).resample('Y').prod() - 1
    ineff_yr_rt = (1 + ew_port.iloc[:, 1].pct_change().dropna()).resample('Y').prod() - 1

    bar_width = 0.35 
    index = np.arange(len(eff_yr_rt.index))
    plt.figure(figsize=(10, 6)) 
    plt.bar(index, eff_yr_rt, bar_width, label='Efficient')
    plt.bar(index + bar_width, ineff_yr_rt, bar_width, label='Inefficient')
    plt.xlabel('Year'), plt.ylabel('Return')
    plt.title('Comparison of Equal Weight Efficient and Inefficient Portfolios\' Yearly Returns')
    plt.xticks(index + bar_width / 2, [item.strftime('%Y') for item in eff_yr_rt.index], rotation = 90)

    plt.tight_layout()
    plt.legend()
    #plt.show()
    plt.savefig('results/yearly_returns.png')

    
    bench = yf.download('^TWII', start=ew_port.index[0].strftime('%Y-%m-%d'), end=ew_port.index[-1].strftime('%Y-%m-%d'))['Close']
    bench = (bench / bench.iloc[0]) * 1000000
    plt.figure(figsize=(10, 6))
    plt.plot(ew_port.iloc[:, 0], label=ew_port.columns[0])
    plt.plot(ew_port.iloc[:, 1], label=ew_port.columns[1])
    plt.plot(bench, label = 'Bechmark - TWII')
    plt.title("Portfolios' Value Compare with Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig('results/port_val.png')