
import pandas as pd
from models import *
from fin_utils import *
import warnings

warnings.filterwarnings("ignore")

if __name__=='__main__':
    file = 'full_data.csv'
    data = pd.read_csv(file)

    # Define Inputs and Outputs.
    in_list = ['流動比率', '速動比率', '利息支出率', '負債比率', 
        '淨值/資產', '長短期借款', '利息保障倍數',
        '營運資金', '總資產週轉次數', '平均收帳天數', '存貨週轉率（次）', '固定資產週轉次數',
        '自由現金流量(D)', '員工人數', '營業費用率', '用人費用率', '研究發展費用率', '呆帳費用率', '營運槓桿度',
        '財務槓桿度', '現金流量比率', '有息負債利率']
    out_list = ['Tobins Q', '現金股利率', 'ROA－綜合損益',
        'ROE－綜合損益', '營業毛利率', '營業毛利成長率', '營業利益率', '每股盈餘', '股利支付率', '營收成長率', '稅前淨利率',
        '稅後淨利率', '業外收支/營收', '營業資產報酬率', '淨值成長率', '總資產成長率']

    # deal with NaN values - insert average of previous years or average of the same sectors.
    df = data.copy()
    for col in df.columns:
        if col in ['公司', '年月', 'TSE新產業名', 'ipo_date']:
            pass
        else:
            df[col] = df.groupby('公司')[col].transform(lambda x: x.fillna(x.mean()))
            df[col] = df[col].fillna(df.groupby('TSE新產業名')[col].transform(lambda x: x.mean()))

    # start analysing sectors
    perf_dict, port_dict = {}, {}
    eff_port_values, ineff_port_values = {}, {}
    for sec in df['TSE新產業名'].unique():# Compare Firms Within the Same Industry.
        print(f'Begin: {sec}')
        if sec in ['M2800 金融業', 'W91   存託憑證']:
                print(f'{sec}: Pass')
                continue
        # Set up Portfolios.
        eff_port, ineff_port = my_portfolio(), my_portfolio()
        for year in df['年月'].unique():# Analyze Firms By Each Year.
            print(f'Analyzing data: {sec}-{year}')
            ind = df[(df['TSE新產業名'] == sec)]['公司'].to_list()
            ind_data = df[(df['公司'].isin(ind)) & (df['年月'] == year)]
            ind_data = ind_data[ind_data['ipo_date']<ind_data['年月']]
    
            # PCA
            pca_data = pca_IO(data=ind_data, in_list=in_list, out_list=out_list)
            x, y = pca_data.fit(var_level='kaiser')
            x.index, y.index = ind_data['公司'], ind_data['公司']

            # DEA
            dea_result = DEA(DMUs_Name=ind_data['公司'], X=x, Y=y)
            result = dea_result.dea()  #dea results
            dea_result.dmus_efficiency()
            if dea_result.max_efficiency != 1 :
                print(f'{sec}-{year} No Efficient Firms.')
                continue

            # Get Efficient and Inefficient Firms. (same amount of stocks in both type)
            eff = dea_result.n_efficient_dmus
            ineff = dea_result.n_inefficient_dmus
            
            # Set Back test Trading Time.
            trading_time = [dt.datetime.strptime(time, '%Y-%m-%d') + relativedelta(days=1) for time in ind_data['年月']]
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


    sectors = list(perf_dict.keys())
    eff_ret = [v[0] for v in perf_dict.values()]
    ineff_ret = [v[1] for v in perf_dict.values()]
    
    comp_dict={}
    for sec_name in perf_dict.keys():
        single_sector = perf_dict[sec_name]
        comp_dict[sec_name] = single_sector[0]-single_sector[1]
    
    pos_dict = {key: value for key, value in comp_dict.items() if value > 0}
    neg_dict = {key: value for key, value in comp_dict.items() if value < 0}
    len(pos_dict)
    len(neg_dict)
    sum(pos_dict.values())/len(pos_dict)
    sum(neg_dict.values())/len(neg_dict)


    # Set up the bar width and positions
    bar_width = 0.35 
    index = np.arange(len(sectors)) 

    plt.bar(index, eff_ret, bar_width, label='Efficient')
    plt.bar(index + bar_width, ineff_ret, bar_width, label='Inefficient')
    plt.xlabel('Sectors'), plt.ylabel('Returns')
    plt.xlabel(rotation=90)
    plt.title('Comparison of Efficient and Inefficient Portfolios\' Returns')
    plt.xticks(index + bar_width / 2, sectors)

    plt.legend()
    plt.show()

    # compare performance metrics
    eff_val_df, ineff_val_df = pd.DataFrame(eff_port_values).ffill(), pd.DataFrame(ineff_port_values).ffill()
    eff_val_df.index, ineff_val_df.index = eff_val_df.index.tz_localize(None), ineff_val_df.index.tz_localize(None)
    
    rf_df = pd.read_csv('risk_free_rate.csv', index_col=0)
    eff_perf = calculate_performances(eff_val_df, rf=rf_df)
    ineff_perf = calculate_performances(ineff_val_df, rf=rf_df)

    # show average of all metrics
    eff_perf.mean()
    ineff_perf.mean()
