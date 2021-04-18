from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()
from hmm import *
import csv
import pandas as pd
import numpy as np

def get_stock_observations(ticker, start_y, start_m, start_d, end_y, end_m, end_d):
    start_date= datetime(start_y, start_m, start_d)
    end_date=datetime(end_y, end_m, end_d)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True,drop=False)
    observations = []
    values = []
    date_list = []
    for t in range(data.shape[0]):
        open_price = data.iat[t, data.columns.get_loc('Open')]
        close_price = data.iat[t, data.columns.get_loc('Close')]
        date = data.iat[t, data.columns.get_loc('Date')]
        change = 100*(close_price - open_price) / open_price
        if change >= 10:
            obs = '10+'
            value = 3
        elif change < 10 and change >= 5:
            obs = '10:5'
            value = 2
        elif change < 5 and change >= 1:
            obs = '5:1'
            value = 1
        elif change < 1 and change > -1:
            obs = '1:-1'
            value = 0
        elif change <= -1 and change > -5:
            obs = '-1:-5'
            value = -1
        elif change <= -5 and change > -10:
            obs = '-5:-10'
            value = -2
        elif change <= -10:
            obs = '-10+'
            value = -3
        date_list.append(date.date())
        values.append(value)
        observations.append(obs)

    d = {'Date': date_list, 'Values' : values}
    plot_dataframe = pd.DataFrame(data = d)

    return observations, plot_dataframe

def split_obs(full_arr, n_sections):
    split_arr = np.array_split(full_arr,n_sections)

    monthly_arr = np.array_split(full_arr,12)
    # initialize hmm for stocks
    section_hmm = hmm("hmm_json_files/initial_stock.json")

    #train hmm on initial data
    init_obs = split_arr[0]
    section_hmm.runEM(init_obs)
    prob = 0

    for t in range(1,n_sections):
        prob += section_hmm.forward(split_arr[t])
        print(t)
        print(prob)

    prob_avg = prob/n_sections
    
    return section_hmm, prob_avg

def compare_likelihood(ticker, year_1, year_end):

    # MONTHLY
    #get observations for training
    obs, df = get_stock_observations(ticker, year_1, 1, 1, year_end, 1, 1)
    full_arr = np.array(obs)

    print('--------------')
    print('Beginning testing for 12 one-month segments')
    monthly_hmm, monthly_prob = split_obs(full_arr,12)
    print('Comparing likelihood for ',ticker,' from 1/1/', year_1,' to 1/1/', year_end,': Prob = ',monthly_prob,' for 11 segments')
    print('--------------')
    print('Beginning testing for 6 two-month segments')
    bi_hmm, bi_monthly_prob = split_obs(full_arr,6)
    print('Comparing likelihood for ',ticker,' from 1/1/', year_1,' to 1/1/', year_end,': Prob = ',bi_monthly_prob,' for 5 segments')
    print('--------------')
    print('Beginning testing for 4 three-month segments')
    three_month_hmm, three_month_prob = split_obs(full_arr,4)
    print('Comparing likelihood for ',ticker,' from 1/1/', year_1,' to 1/1/', year_end,': Prob = ',three_month_prob,' for 3 segments')
    print('--------------')
    print('Beginning testing for 3 four-month segments')
    four_month_hmm, four_month_prob = split_obs(full_arr,3)
    print('Comparing likelihood for ',ticker,' from 1/1/', year_1,' to 1/1/', year_end,': Prob = ',four_month_prob,' for 2 segments')

if __name__ == '__main__':
    #get observations for given range of dates
    compare_likelihood('DIS', 2020, 2021)

    #intialize model with given hmm json file
    # dis_hmm = hmm("hmm_json_files/initial_stock.json")

    # #run EM to optimize model
    # dis_hmm.runEM(dis_obs, 'hmm_json_files/disney_hmm.json')

    # #get observations for given range of dates
    # new_obs, new_df = get_stock_observations('DIS', 2020, 2, 1, 2020, 3, 1)

    # #put dataframe with dates and values from the df above into csv for matlab
    # #df consists of dates (x axis) and measureable value to predict against
    # # dis_df.to_csv('test.csv')

    # prob = dis_hmm.forward(new_obs)

    # print(prob)

