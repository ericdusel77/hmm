import numpy as np
from hmmlearn import hmm
from pandas_datareader import data as pdr
from datetime import datetime
import yfinance as yf
yf.pdr_override()

def get_stock_observations(ticker, start_y, start_m, start_d, end_y, end_m, end_d):
    start_date= datetime(start_y, start_m, start_d)
    end_date=datetime(end_y, end_m, end_d)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True,drop=False)
    observations = []
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
    return observations

if __name__ == '__main__':
    # startprob = np.array([0.6, 0.3, 0.1, 0.0])
    # # The transition matrix, note that there are no transitions possible
    # # between component 1 and 3
    # transmat = np.array([[0.7, 0.2, 0.0, 0.1],
    #                     [0.3, 0.5, 0.2, 0.0],
    #                     [0.0, 0.3, 0.5, 0.2],
    #                     [0.2, 0.0, 0.2, 0.6]])
    # # The means of each component n-components by n_features (fracChange, fracHigh, fracLow)
    # means = np.array([[0.0,  0.0, 0.0],
    #                 [0.0, 11.0, 0.0],
    #                 [9.0, 10.0, 0.0],
    #                 [11.0, -1.0, 0.0]])
    # # The covariance of each component

    # covars = .5 * np.tile(np.identity(3), (4,1,1))

    train_X = get_stock_observations("DIS", 2020, 1, 1, 2020, 2, 1)

    # Build an HMM instance and set parameters
    model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=1000)

    model.fit(train_X)
    # print(model.n_features)
    score_X = get_stock_observations("DIS", 2020, 3, 1, 2020, 3, 6)
    score = model.score(score_X)

    print(score)
    print(np.exp(score))
    # Instead of fitting it from the data, we directly set the estimated
    # parameters, the means and covariance of the components
    # model.startprob_ = startprob
    # model.transmat_ = transmat
    # model.means_ = means
    # model.covars_ = covars