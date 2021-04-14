from pandas_datareader import data as pdr
from datetime import datetime
from datetime import date
import yfinance as yf
yf.pdr_override()
import pandas as pd

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
