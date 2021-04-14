from pandas_datareader import data as pdr
from datetime import datetime
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd
from hmm import *
from stockdata import *

def get_stock_observations(ticker, start_y, start_m, start_d, end_y, end_m, end_d):
    start_date= datetime(start_y, start_m, start_d)
    end_date=datetime(end_y, end_m, end_d)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    
    observations = []
    for t in range(data.shape[0]):
        open_price = data.iat[t, data.columns.get_loc('Open')]
        close_price = data.iat[t, data.columns.get_loc('Close')]
        change = 100*(close_price - open_price) / open_price
        if change >= 10:
            obs = '10+'
        elif change < 10 and change >= 5:
            obs = '10:5'
        elif change < 5 and change >= 1:
            obs = '5:1'
        elif change < 1 and change > -1:
            obs = '1:-1'
        elif change <= -1 and change > -5:
            obs = '-1:-5'
        elif change <= -5 and change > -10:
            obs = '-5:-10'
        elif change <= -10:
            obs = '-10+'

        observations.append(obs)
    
    return observations

if __name__ == '__main__':
    dis_obs = get_stock_observations('DIS', 2020, 1, 14, 2021, 4, 13)
    aapl_obs = get_stock_observations('AAPL', 2020, 1, 14, 2021, 4, 13)
    tsla_obs = get_stock_observations('TSLA', 2020, 1, 14, 2021, 4, 13)

    dis_hmm = hmm("initial_stock.json")
    aapl_hmm = hmm("initial_stock.json")
    tsla_hmm = hmm("initial_stock.json")

    dis_hmm.runEM(dis_obs, 10, 'disney_hmm.json')
    aapl_hmm.runEM(aapl_obs, 10, 'apple_hmm.json')
    tsla_hmm.runEM(tsla_obs, 10, 'tesla_hmm.json')
