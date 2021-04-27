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
    close_prices = []
    for t in range(data.shape[0]):
        new_observation = []
        open_price = data.iat[t, data.columns.get_loc('Open')]
        close_price = data.iat[t, data.columns.get_loc('Close')]
        high_price = data.iat[t, data.columns.get_loc('High')]
        low_price = data.iat[t, data.columns.get_loc('Low')]
        date = data.iat[t, data.columns.get_loc('Date')]

        change = (close_price - open_price) / open_price
        # high = (high_price - open_price) / open_price
        # low = (open_price - low_price) / open_price 

        # if change > 0:
        #     obs = 1
        # else:
        #     obs = 0
        if change >= 0.04:
            obs = 5
        elif change < 0.04 and change >= 0.02:
            # obs = 'medium_gainz'
            obs = 4
        elif change < 0.02 and change > 0:
            # obs = 'little_gainz'
            obs = 3
        elif change <= 0 and change > -0.02:
            # obs = 'little_loss'
            obs = 2
        elif change <= 0.02 and change > -0.05:
            # obs = 'medium_loss'
            obs = 1
        elif change <= -0.05:
            # obs = 'large_loss'
            obs = 0
        # new_observation = [change, high, low]
        new_observation = [obs]
        observations.append(new_observation)
        dates.append(date)
        open_prices.append(open_price)
        close_prices.append(close_price)
    return observations, dates, open_prices, close_prices

class predictStock(object):
    def __init__(self, ticker, train_dates, test_dates, n_latency_days=10, n_hidden_states=4):
        self.ticker = ticker
        self.n_latency_days = n_latency_days
        
        self.train_x, self.train_dates, _, _ = get_stock_observations(self.ticker, train_dates)

        self.test_x, self.test_dates, self.test_opens, self.test_close = get_stock_observations(self.ticker, test_dates)

        self.hmm = hmm.MultinomialHMM(n_components=n_hidden_states, n_iter=1000)
        # self.hmm = hmm.GaussianHMM(n_components=n_hidden_states, n_iter=1000)
        # self.hmm = hmm.GMMHMM(n_components=n_hidden_states, n_mix=4, n_iter=1000)
        
        self.all_outcomes()

    def fit(self, lengths_val):
        lengths = []
        remaining_data = len(self.train_x)
        for t in range(0,len(self.train_x)//lengths_val):
            lengths.append(lengths_val)
            remaining_data -= lengths_val
        if remaining_data > 0:
            lengths.append(remaining_data)
        self.hmm.fit(self.train_x,lengths)


    # find all outcomes given the range defined here calculations = n_steps_frac_change * n_steps_frac_high * n_steps_frac_low
    def all_outcomes(self):
        self.outcomes = [0, 1, 2, 3, 4, 5]
        # self.outcomes = [0, 1]

    def likely_outcome(self, day_idx):
        # combine observations so that you can look before the test set starts
        full_obs = self.train_x+self.test_x
        full_id = day_idx+len(self.train_x)
        #find an array of previous observations to use for latency
        prev_start_idx = max(0, full_id - self.n_latency_days)
        prev_end_idx = max(0, full_id)
        previous_data = full_obs[prev_start_idx: prev_end_idx]
        outcome_score = []
        #test all outcomes 
        for o in self.outcomes:
            obs_seq = np.row_stack((previous_data, o))
            outcome_score.append(self.hmm.score(obs_seq))
        
        most_likely_outcome = self.outcomes[np.argmax(outcome_score)]
 
        return most_likely_outcome

    def predict_close(self,day):
        idx = self.test_dates.index(day)
        open_price = self.test_opens[idx]
        prediction = self.likely_outcome(idx)
        prediction = prediction - 2 #change for symbols representing negative values
        close_prediction = prediction
        # print(prediction)
        # print(self.test_opens[idx])
        # print(close_prediction)
        # print(self.test_close[idx])
        return close_prediction, prediction

    #score model from observations in given range (must be within test data)
    def score_model(self,test_range):
        start_idx = self.test_dates.index(test_range[0])
        end_idx = self.test_dates.index(test_range[1])
        test_data = self.test_x[start_idx: end_idx+1]
        score = self.hmm.score(test_data)
        print('Evaluation of model for ',test_range[0], ' to ', test_range[1], ": ",score)
        return score 
    
    def print_predictions(self):
        predictions = []
        fractions = []
        for t in self.test_dates:
            close_prediciton, fracChange = self.predict_close(t)
            predictions.append(close_prediciton)
            fractions.append(fracChange)
        d = {'Date': self.test_dates, "Actual Open": self.test_opens, "Fractional Change Prediction": fractions, 'Predictions' : predictions, 'Actual Close' : self.test_close}
        df = pd.DataFrame(data = d)
        filename = 'csv_files/'+self.ticker+'.csv'
        df.to_csv(filename)
        return df

if __name__ == '__main__':
    train_dates = [datetime(2015, 1, 1), datetime(2020, 12, 31)]
    test_dates  = [datetime(2021, 1, 1), datetime(2021, 4, 15)] #does all dates in between

    stocks = ['DIS','AAPL', 'TSLA', 'SPY', 'AAL', 'JNJ', 'COST', 'PFE','TGT','GME']

    for s in stocks:
        dis_ps = predictStock(s,train_dates,test_dates,n_hidden_states=4)
        # dis_ps.fit(100)
        dis_ps.fit(len(dis_ps.train_x))
        print('-------')
        print(s)
        print('Trans Probability')
        print(dis_ps.hmm.transmat_)
        print('Emmission Probability')
        print(dis_ps.hmm.emissionprob_)
        dis_ps.print_predictions()

    # dis_ps.score_model([dis_ps.test_dates[0],dis_ps.test_dates[9]])
    # dis_ps.score_model([dis_ps.test_dates[10],dis_ps.test_dates[19]])
    # dis_ps.score_model([dis_ps.test_dates[20],dis_ps.test_dates[29]])
    # dis_ps.score_model([dis_ps.test_dates[30],dis_ps.test_dates[39]])
    # dis_ps.score_model([dis_ps.test_dates[40],dis_ps.test_dates[49]])
    # dis_ps.score_model([dis_ps.test_dates[50],dis_ps.test_dates[59]])


        



