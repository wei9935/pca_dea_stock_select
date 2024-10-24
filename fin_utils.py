import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

import backtrader as bt

# define portfolio strategy that rebalance monthly with a given weight
class port_rebalabce_strategy(bt.Strategy):
    params = dict(
        rebalance_months = list(range(1, 13)),
        rebalance_day = None
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        if isinstance(dt, float):
            dt = bt.num2date(dt)
        print("%s, %s" % (dt.date(), txt))

    def print_signal(self):
        self.log(
            f"o {self.datas[0].open[0]:7.2f} "
            f"h {self.datas[0].high[0]:7.2f} "
            f"l {self.datas[0].low[0]:7.2f} "
            f"c {self.datas[0].close[0]:7.2f} "
            f"v {self.datas[0].volume[0]:7.0f} "
        )

    def notify_order(self, order):
        """ Triggered upon changes to orders. """
        # Suppress notification if it is just a submitted order.
        if order.status == order.Submitted:
            return
        
        type = "Buy" if order.isbuy() else "Sell"
        date = self.data.datetime.datetime().date()
        # Check if an order has been completed
        if order.status in [order.Completed]:
            self.log(
                f"{date} Order Complete: {type} {order.data._name} | "
                f"Price: {order.executed.price:6.2f} | " 
                f"Cost: {order.executed.value:6.2f} | " 
                f"Size: {order.created.size:9.4f}"
                )

    def notify_trade(self, trade):
        """Provides notification of closed trades."""
        if trade.isclosed:
            self.log(
                f"{trade.data._name} Closed: PnL Gross {round(trade.pnl, 2)}, Net {round(trade.pnlcomm, 1)}"
            )

    def __init__(self):
        self.comp = []
        self.rebalance_record = dict()
        for d in self.datas:
            d.target = {
                dt.datetime.strptime(date, '%Y-%m-%d').date(): allocation
                for date, allocation in d.target.items()
            }
            self.rebalance_record[d] = dict()
            d.wt = 0
          
    def next(self):
        date = self.data.datetime.date()
        for d in self.datas:
            dname = d._name
            pos = self.getposition(d).size
            mon_year = date.strftime("%b-%y")
            # Check if monthly rebalanced.
            if mon_year not in self.rebalance_record[d]:
                self.rebalance_record[d][mon_year] = False

            if date in d.target:
                self.comp.append(dname)
                d.wt = d.target[date]
            
            target_weight = d.wt
            if date.month in self.p.rebalance_months and self.rebalance_record[d][mon_year] == False:
                if target_weight > 0:
                    print(f'{date} Sending Order: {dname} | Month {date.month} | Pos: {pos} | Target Weight: {target_weight*100}%')
                    self.order_target_percent(d, target=target_weight)
                else:
                    # If target weight is zero, close the position if it exists
                    if pos > 0:
                        print(f'{date} Closing Position: {dname} | Month {date.month} | Pos: {pos}')
                        self.close(d)
                self.rebalance_record[d][mon_year] = True

# create portfolio with backtest functions                       
class my_portfolio(object):
    def __init__(self, rf_df=None):
        self.rf = rf_df
        self.performance = []
        self.port = pd.DataFrame(columns=['time', 'stock', 'weight'])

    def add_stocks(self, stock, weight, time):
        # Check if the input is a list or a single value
        if isinstance(stock, list) and isinstance(weight, list):
            if len(stock) != len(weight):
                raise ValueError("Stock and weight lists must have the same length.")
            for s, w, t in zip(stock, weight, time):
                # Add to the portfolio DataFrame
                self.port = self.port.append({'stock': s, 'weight': w, 'time': t}, ignore_index=True)
        else:
            # Assume single value input
            self.port = self.port.append({'stock': stock, 'weight': weight, 'time': time}, ignore_index=True)
    
    def get_next_trading_day(self, date): # make sure the target date is a trading day
        if date in self.trading_days:
            return dt.datetime.strftime(date, '%Y-%m-%d')
        else:
            # Find the next trading day after the current date
            next_trading_day = self.trading_days[self.trading_days > date].min()
            return dt.datetime.strftime(next_trading_day, '%Y-%m-%d')
    
    def fill_missing_values(self, group):
        # fill missing values to prevent backtrader to cut off datas, fill 0 even when the stock has not yet ipo.
        for col in group.columns:
            # fill 0s for all values before the first valid entry
            first_valid_idx = group[col].first_valid_index()
            if first_valid_idx is not None:
                group[col].loc[:first_valid_idx] = group[col].loc[:first_valid_idx].fillna(0)
            # forward fill the remaining NaN values (previous value filling)
            group[col] = group[col].ffill()

        return group

    def yf_btData(self, df, fill_date=None):
        # reformat yfinace download data for back trading.
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.dropna(axis=1)
        data = pd.melt(df, id_vars=[('Date', '')], var_name=['Metric', 'Ticker'])

        data.columns = ['Date', 'Metric', 'Ticker', 'Value']# Rename columns
        data = data.pivot_table(index=['Date', 'Ticker'], columns='Metric', values='Value').reset_index()# Pivot the DataFrame to rearrange the data
        data.columns.name = None# Reset column names

        data = data.sort_values(by='Ticker').reset_index(drop=True) #Sorting
        data = data[['Ticker', 'Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
        data.set_index(['Ticker', 'Date'], inplace=True)
        data.sort_index(level=['Ticker', 'Date'], inplace=True)
        data.apply(pd.to_numeric, errors='coerce')
        
        if fill_date == True:
            data.index = data.index.set_levels([data.index.levels[0], pd.to_datetime(data.index.levels[1])])
            # Define the full date range
            full_date_range = data.index.get_level_values(1).unique().sort_values()
            # Reindex each 'Ticker' to have the full date range and fill missing values with 0
            data = data.groupby(level='Ticker').apply(
                lambda group: group.reindex(pd.MultiIndex.from_product([[group.name], full_date_range], names=['Ticker', 'Date']),
                                            fill_value=None))
            if len(data.index.levels) == 3:
                data.index = data.index.droplevel(0)
            
            data = data.groupby(level='Ticker').apply(self.fill_missing_values)
        
        return data

    def bt_data(self):
        # prepare datas requiered for backtesting.
        port = self.port.copy()
        port['stock'] = [item.split()[0] + '.TW' for item in port['stock'].to_list()]
        ticker_names = port['stock'].unique()

        days = pd.to_datetime(port['time'])
        start = days.min()
        end = days.max()+relativedelta(years=1)
        prices = yf.download(list(ticker_names), 
                             start=dt.datetime.strftime(start, '%Y-%m-%d'), 
                             end=dt.datetime.strftime(end, '%Y-%m-%d'))
        prices = self.yf_btData(prices, fill_date=True)
        self.prices = prices
        self.trading_days = prices.index.get_level_values(1)
        
        port['time'] = port['time'].apply(lambda date: self.get_next_trading_day(date))
        
        targets = {key: {row['time']: row['weight'] for _, row in group.iterrows()} 
                   for key, group in port.groupby('stock')}
        all_dates = {d for v in targets.values() for d in v}  # Store the set of dates
        for key in targets:
            _ = [targets[key].setdefault(date, 0) for date in all_dates]  # Use list comprehension but don't print
        self.targets = targets
        return start, end, prices, targets

    def bt_back_test(self, rebalance = None, ini_value=1000000, commission=0.0012):
        start, end, prices, targets = self.bt_data()
        cerebro = bt.Cerebro()
        for ticker, data in prices.groupby(level=0):
            data = bt.feeds.PandasData(dataname=data.droplevel(level=0),
                                        name=str(ticker),
                                        fromdate=start,
                                        todate=end,
                                        plot=False)
            data.target = targets[ticker]
            cerebro.adddata(data, name=ticker)
        print(f'Data Downloaded: {start}-{end}')

        # Execute
        cerebro.addstrategy(port_rebalabce_strategy)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        cerebro.broker.setcash(ini_value)
        cerebro.broker.setcommission(commission=commission)
                
        result = cerebro.run()
        
        strat = result[0]
        pyfoliozer = strat.analyzers.getbyname('pyfolio')
        self.returns, self.bt_positions, self.bt_transactions, self.bt_gross_lev = pyfoliozer.get_pf_items()

        self.bt_port_value = self.returns.cumsum().apply(np.exp) * ini_value
        self.bt_total_return = (self.bt_port_value.iloc[-1] - self.bt_port_value.iloc[0]) / self.bt_port_value.iloc[0]
        self.bt_annual_return = (1 + self.bt_total_return) ** (252 / len(self.bt_port_value)) - 1
        

def calculate_performances(port_values, rf):
        trading_days_per_year = 252
        sharpe_ratios, sortino_ratios, annual_returns = {}, {}, {}
        annual_volatilities, max_drawdowns, daily_vars = {}, {}, {}

        rf.index = pd.to_datetime(rf.index)
        daily_rf_rate = rf.resample('D').ffill()

        # Align the risk-free rate series with the portfolio DataFrame's index
        # Assuming portfolio_df index starts at '2023-01-01' and matches the date range
        port_values.index = pd.to_datetime(port_values.index)
        aligned_rf = daily_rf_rate.reindex(port_values.index).ffill()
        
        for portfolio_name, portfolio_values in port_values.items():
            portfolio_series = pd.Series(portfolio_values)
            # Daily returns
            daily_returns = portfolio_series.pct_change().dropna()
            risk_free_rate = aligned_rf.reindex(daily_returns.index).values.flatten()/100
            rf = (1 + risk_free_rate)**(1/252) - 1
            # Annual Return
            total_return = (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]
            annual_return = (1 + total_return) ** (trading_days_per_year / len(portfolio_series)) - 1
            annual_returns[portfolio_name] = annual_return
            
            # Annual Volatility
            annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
            annual_volatilities[portfolio_name] = annual_volatility
            
            # Sharpe Ratio
            sharpe_ratio = (daily_returns - rf).mean() / daily_returns.std() * np.sqrt(trading_days_per_year)
            sharpe_ratios[portfolio_name] = sharpe_ratio
            
            # Sortino Ratio
            downside_deviation = daily_returns[daily_returns < 0].std()
            sortino_ratio = (daily_returns - rf).mean() / downside_deviation * np.sqrt(trading_days_per_year)
            sortino_ratios[portfolio_name] = sortino_ratio
            
            # Maximum Drawdown (MDD)
            cumulative_returns = (1 + daily_returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            max_drawdowns[portfolio_name] = max_drawdown
            
            # Daily VaR (Value at Risk)
            confidence_level = 0.95
            daily_var = -np.percentile(daily_returns, 100 * (1 - confidence_level))
            daily_vars[portfolio_name] = daily_var

        # Combine the results into a DataFrame
        results_df = pd.DataFrame({
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatilities,
            'Sharpe Ratio': sharpe_ratios,
            'Sortino Ratio': sortino_ratios,
            'Max Drawdown': max_drawdowns,
            'Daily VaR (95%)': daily_vars
        })

        return results_df

