import pandas as pd
import numpy as np
from hmmlearn import hmm
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()
import itertools

def get_stock_observations(ticker, start_end):
    data = pdr.get_data_yahoo(ticker, start=start_end[0], end=start_end[1])
    data.reset_index(inplace=True,drop=False)
    observations = []
    dates = []
    open_prices = []
    for t in range(data.shape[0]):
        new_observation = []
        open_price = data.iat[t, data.columns.get_loc('Open')]
        close_price = data.iat[t, data.columns.get_loc('Close')]
        high_price = data.iat[t, data.columns.get_loc('High')]
        low_price = data.iat[t, data.columns.get_loc('Low')]
        date = data.iat[t, data.columns.get_loc('Date')]

        change = 100*(close_price - open_price) / open_price
        high = 100*(high_price - open_price) / open_price
        low = 100*(open_price - low_price) / open_price 

        new_observation = [change, high, low]
        observations.append(new_observation)
        dates.append(date)
        open_prices.append(open_price)
    return observations, dates, open_prices

class predictStock(object):
    def __init__(self, ticker, train_dates, test_dates, test_size=0.33, n_latency_days=10, n_hidden_states=4, n_steps_frac_change=50, n_steps_frac_high=10, n_steps_frac_low=10):
        self.ticker = ticker
        self.n_latency_days = n_latency_days
        self.hmm = hmm.GaussianHMM(n_components=n_hidden_states)
        self.train_x, self.train_dates, _ = get_stock_observations(self.ticker, train_dates)
        self.test_x, self.test_dates, self.test_opens = get_stock_observations(self.ticker, test_dates)
        self.all_outcomes(n_steps_frac_change,n_steps_frac_high,n_steps_frac_low)

    def fit(self):
        self.hmm.fit(self.train_x)

    def all_outcomes(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)
        self.outcomes = np.array(list(itertools.product(frac_change_range, frac_high_range, frac_low_range)))

    def likely_outcome(self, day_idx):
        prev_start_idx = max(0, day_idx - self.n_latency_days)
        prev_end_idx = max(0, day_idx)
        previous_data = self.test_x[prev_start_idx: prev_end_idx]
        outcome_score = []
        for o in self.outcomes:
            obs_seq = np.row_stack((previous_data, o))
            outcome_score.append(self.hmm.score(obs_seq))
        
        most_likely_outcome = self.outcomes[np.argmax(outcome_score)]
 
        return most_likely_outcome

    def predict_close(self,day):
        idx = self.test_dates.index(day)
        open_price = self.test_opens[idx]
        prediction = self.likely_outcome(idx)
        close_prediction = open_price*(1+prediction[0])
        return close_prediction

    def score_model(self,test_range):
        start_idx = self.test_dates.index(test_range[0])
        end_idx = self.test_dates.index(test_range[1])
        test_data = self.test_x[start_idx: end_idx+1]
        score = self.hmm.score(test_data)
        print('Evaluation of model for ',test_range[0], ' to ', test_range[1], ": ",score)
        return score

if __name__ == '__main__':
    train_dates = [datetime(2017, 1, 1), datetime(2020, 12, 31)]
    test_dates  = [datetime(2021, 1, 1), datetime(2021, 4, 15)] #does all dates in between

    dis_ps = predictStock('DIS',train_dates,test_dates)
    dis_ps.fit()

    dis_ps.score_model([dis_ps.test_dates[0],dis_ps.test_dates[9]])
    dis_ps.score_model([dis_ps.test_dates[10],dis_ps.test_dates[19]])
    dis_ps.score_model([dis_ps.test_dates[20],dis_ps.test_dates[29]])
    dis_ps.score_model([dis_ps.test_dates[30],dis_ps.test_dates[39]])
    dis_ps.score_model([dis_ps.test_dates[40],dis_ps.test_dates[49]])
    dis_ps.score_model([dis_ps.test_dates[50],dis_ps.test_dates[59]])

    dis_ps_2 = predictStock('DIS',train_dates,test_dates,n_hidden_states=5)
    dis_ps_2.fit()

    dis_ps_2.score_model([dis_ps.test_dates[0],dis_ps.test_dates[9]])
    dis_ps_2.score_model([dis_ps.test_dates[10],dis_ps.test_dates[19]])
    dis_ps_2.score_model([dis_ps.test_dates[20],dis_ps.test_dates[29]])
    dis_ps_2.score_model([dis_ps.test_dates[30],dis_ps.test_dates[39]])
    dis_ps_2.score_model([dis_ps.test_dates[40],dis_ps.test_dates[49]])
    dis_ps_2.score_model([dis_ps.test_dates[50],dis_ps.test_dates[59]])
