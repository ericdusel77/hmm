from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()
from hmm import *
import csv
import pandas as pd
import numpy as np

def get_stock_observations(ticker, start_end):
    data = pdr.get_data_yahoo(ticker, start=start_end[0], end=start_end[1])
    data.reset_index(inplace=True,drop=False)
    observations = []
    values = []
    date_list = []
    opens = []
    closes = []
    for t in range(data.shape[0]):
        open_price = data.iat[t, data.columns.get_loc('Open')]
        close_price = data.iat[t, data.columns.get_loc('Close')]
        date = data.iat[t, data.columns.get_loc('Date')]
        change = 100*(close_price - open_price) / open_price
        if change >= 3:
            obs = 'large_gainz'
            value = 3
        elif change < 3 and change >= 2:
            obs = 'medium_gainz'
            value = 2
        elif change < 2 and change > 0:
            obs = 'little_gainz'
            value = 1
        elif change <= 0 and change > -2:
            obs = 'little_loss'
            value = 0
        elif change <= -2 and change > -3:
            obs = 'medium_loss'
            value = -1
        elif change <= -3:
            obs = 'large_loss'
            value = -2
        # if change >= 0:
        #     obs = 'gainz'
        #     value = 1
        # elif change < 0 :
        #     obs = 'loss'
        #     value = -1
        date_list.append(date.date())
        values.append(value)
        observations.append(obs)
        opens.append(open_price)
        closes.append(close_price)

    return observations, date_list, values, opens, closes

def predictStock(ticker, train_dates, test_dates):
    train_x, train_dates, train_val, train_open, train_close = get_stock_observations(ticker, train_dates)

    test_x, test_dates, test_val, test_open, test_close = get_stock_observations(ticker, test_dates)

    new_hmm = hmm("hmm_json_files/initial_stock.json")

    new_hmm.runEM(train_x)
     # combine observations so that you can look before the test set starts
    full_obs = train_x+test_x

    print(new_hmm.A)
    print(new_hmm.B)

    predictions = []

    for t in range(0,len(test_dates)):
        full_id = t+len(train_x)
        #find an array of previous observations to use for latency
        prev_start_idx = full_id - 4
        prev_end_idx = full_id
        previous_data = full_obs[prev_start_idx: prev_end_idx]
        outcome_score = []
        for k in new_hmm.symbols:
            print(k)
            previous_data.append(k)
            outcome_score.append(new_hmm.forward(previous_data))
        curr_prediction = new_hmm.symbols[np.argmax(outcome_score)]
        if 'large_gainz' in curr_prediction:
            value = 3
        elif 'medium_gainz' in curr_prediction:
            value = 2
        elif 'little_gainz' in curr_prediction:
            value = 1
        elif 'little_loss' in curr_prediction:
            value = 0
        elif 'medium_loss' in curr_prediction:
            value = -1
        elif 'large_loss' in curr_prediction:
            value = -2
        # if 'gainz' in curr_prediction:
        #     value = 1
        # elif 'loss' in curr_prediction:
        #     value = -1
        predictions.append(value)

    d = {'Date': test_dates, "Actual Open": test_open, "Fractional Change Prediction": predictions, 'Predictions' : predictions, 'Actual Close' : test_close}

    df = pd.DataFrame(data = d)
    filename = 'csv_files/'+ticker+'.csv'
    df.to_csv(filename)
    return predictions

if __name__ == '__main__':

    # # print(prob)
    # train_dates = [datetime(2020, 1, 1), datetime(2020, 12, 31)]
    # test_dates  = [datetime(2021, 1, 1), datetime(2021, 4, 15)] #does all dates in between


    # stocks = ['DIS','AAPL', 'TSLA', 'SPY', 'AAL', 'JNJ', 'COST', 'PFE','TGT','GME']

    # for s in stocks:
    #     predictStock(s,train_dates,test_dates)
    #     print()

