from hmm import *

if __name__ == '__main__':
    test_hmm = hmm("test_model.json")

    

    seq0 = ('Up', 'Up', 'Up')
    seq1 = ('Up', 'Up', 'Down')
    seq2 = ('Up', 'Down', 'Up')
    seq3 = ('Up', 'Down', 'Down')  
    seq4 = ('Down', 'Up', 'Up')
    seq5 = ('Down', 'Up', 'Down')
    seq6 = ('Down', 'Down', 'Up')
    seq7 = ('Down', 'Down', 'Down')

    observation_list = [seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7]
    # total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    # for obs in observation_list:
        # p1 = test_hmm.forward(obs)
        # p2 = test_hmm.backward(obs)
        # total1 += p1
        # total2 += p2
        # print("Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2, " total_1 = ", total1, " total_2 = ", total2)

 # test the Viterbi algorithm
    observations = seq6   # you can set this variable to any arbitrary length of observations

    test_hmm.baumwelch(observations)
    # prob, hidden_states = test_hmm.viterbi(observations)
    # print ("Max Probability = ", prob, " Hidden State Sequence = ", hidden_states)