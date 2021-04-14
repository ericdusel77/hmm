from hmm import *
from stockdata import *
import json

def runEM(hmm, observations, iterations, output_file_name):
    # print(dis_hmm.A)
    # print(dis_hmm.B)
    # print(dis_hmm.pi)
    for t in range(iterations):
        hmm.baumwelch(observations)
        # print('-----------------------')
        # print(dis_hmm.A)
        # print(dis_hmm.B)
        # print(dis_hmm.pi)

    hmm_write  = {}
    hmm_write['hmm'] = {}
    hmm_write['hmm']['A'] = hmm.A
    hmm_write['hmm']['B'] = hmm.B
    hmm_write['hmm']['pi'] = hmm.pi

    with open(output_file_name, 'w') as outfile:
        json.dump(hmm_write, outfile)

if __name__ == '__main__':
    dis_obs = get_stock_observations('DIS', 2020, 1, 14, 2021, 4, 13)
    aapl_obs = get_stock_observations('AAPL', 2020, 1, 14, 2021, 4, 13)
    tsla_obs = get_stock_observations('TSLA', 2020, 1, 14, 2021, 4, 13)

    dis_hmm = hmm("initial_stock.json")
    aapl_hmm = hmm("initial_stock.json")
    tsla_hmm = hmm("initial_stock.json")

    runEM(dis_hmm, dis_obs, 10, 'disney_hmm.json')
    runEM(aapl_hmm, aapl_obs, 10, 'apple_hmm.json')
    runEM(tsla_hmm, tsla_obs, 10, 'tesla_hmm.json')
