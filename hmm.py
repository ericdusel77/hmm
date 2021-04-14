import json

class hmm():
    def __init__(self, model_name):
        #EXTRACT MODEL PARAMETERS FROM JSON FILE 
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A     = self.model["A"] #transition probability matrix TPM
        self.B     = self.model["B"] #emission probability matrix EPM
        self.pi    = self.model["pi"] #initial state prob distribution PI

        self.states   = list(self.A.keys())
        self.symbols  = list(list(self.B.values())[0].keys())
        self.N        = len(self.states)
        self.M        = len(self.symbols)

        return

    # take series of observations from 0(1), O(2), O(3),.... O(T) and get the P(O | HMM) from forward perspective
    def forward(self, obs):
        alpha = []

        # initialize base case as seeing O(1) at each state (s) with chance pi(s) for starting at state s
        base_case = {}
        for s in self.states:
            base_case[s] = self.pi[s] * self.B[s][obs[0]]
        alpha.append(base_case)

        # for each time step t, to find new alpha for state j, sum up the alpha values for getting to state j from every state i 
        # and seeing observation O(t) (note: observations are 1:T, indexes are 0:T-1 where len(obs) = T e.g. O(3) = O(t=2) ) 
        # this goes from 1:T-1 index for O(2):O(T)
        for t in range(1, len(obs)):
            new_case = {}
            for j in self.states:
                new_case[j] = sum( [ alpha[t-1][i] * self.A[i][j] * self.B[j][obs[t]] for i in self.states ] )
            alpha.append(new_case)

        #sum up all alpha values for every state at the last time step T. (T = len(obs) )
        #to calculate probability of seeing that sequence of observations given the HMM
        prob = sum( [ alpha[len(obs) - 1][y] for y in self.states ] )
        self.alpha = alpha
        return prob

    # take series of observations from 0(t+1), O(t+2), O(t+3),.... O(T) and get the P(O | HMM) from backward perspective
    def backward(self, obs):
        beta = []

        # initialize base case as  being in state s at time T
        base_case = {}
        for s in self.states:
            base_case[s] = 1
        beta.append(base_case)

        # for each time step going from index 1:T-1, to get beta for state i, look for future beta values 
        # (which are currently in 0 index spot of beta) for every state and
        # sum the transition from state i to every state j multiplied by corresponding beta and seeing O(T - t) at state j
        # (note: observations are 1:T, indexes are 0:T-1 where len(obs) = T e.g. O(3) = O(t=2) ) 
        # this goes from 1:T-1 and allows observations in reverse by
        # actual O(T),   .... O(t+3),      O(t+2)
        # actual O(T),   .... O(3),        O(2)
        # index  O(T-1), .... O(T - (T-2), O(T - (T-1))
        # index  O(T-1), .... O(index=2),  O(index=1))
        for t in range(1, len(obs)):
            new_case = {}
            for i in self.states:
                new_case[i] = sum( [  beta[0][j] * self.A[i][j] * self.B[j][obs[len(obs)-t]] for j in self.states ] )
            beta.insert(0,new_case)

        # sum up all beta values for every state at the first time step.
        # multiplied by the prob that we start there (pi) and see initial observation O(t+1) or index O(0) 
        self.beta = beta
        prob = sum( (self.pi[y]* self.B[y][obs[0]] * self.beta[0][y]) for y in self.states)
        return prob

    def viterbi(self, obs):
        delta = []
        path = {} # initialize path. will end with one path for each state (the likeliest path we took to get to that state given the obs)
        # initialize base case as seeing O(1) at each state (s) with chance pi(s) for starting at state s
        # initialize path with the first state we are in
        base_case = {}
        for s in self.states:
            base_case[s] = self.pi[s] * self.B[s][obs[0]]
            path[s] = [s]
        delta.append(base_case)

        # for each time step t, to find new delta for state j, find the max value for getting to state j from a state i 
        # and seeing observation O(t) (note: observations are 1:T, indexes are 0:T-1 where len(obs) = T e.g. O(3) = O(t=2) ) 
        # make sure to add this j state to the path that we took to get to i (whichever we chose as the max) to complete to the path to j
        # this goes from 1:T-1 index for O(2):O(T)
        for t in range(1, len(obs)):
            new_case = {}
            new_path = {} # don't update path until time step is complete (if you update path to a j which is also the prev_state for a j that has not been checked yet)
            for j in self.states:
                (prob, prev_state) = max( [ delta[t-1][i] * self.A[i][j] * self.B[j][obs[t]],i] for i in self.states  )
                new_case[j] = prob
                new_path[j] = path[prev_state] + [j]
            delta.append(new_case)
            path = new_path
        # get the max probability and corresponding state that gives us the most likely state we ended observations with
        # use that state to return the path that ended here and seeing all given observations  
        n = 0           # if only one element is observed max is found in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((delta[n][y], y) for y in self.states)
        return (prob, path[state])

    def baumwelch(self, obs):
        gamma = []
        xi = []
        
        # create alpha and beta tables 
        denominator = self.forward(obs) # use P(O | HMM) for calculating xi and gamma
        self.backward(obs)
        
        # calculate xi and gamma values
        # stop at second to last time step, since we can't get a xi for the last state, since there is no j
        # goes from t=0:T-2 for O(1):O(T-1)
        for t in range(len(obs)-1):
            new_xi = {} # xi and gamma list objects for this time setp
            new_gamma = {}
            for i in self.states:
                summation = 0.0
                new_xi[i] = {} # initialize the xi's for this i at this time step
                for j in self.states:
                    numerator = ( self.alpha[t][i] * self.A[i][j] * self.B[j][obs[t+1]] * self.beta[t+1][j]  ) # prob of being in i at t, and j at t+1
                    new_xi[i][j] = numerator / denominator # normalize by the probability of this Obs set given our current model for every transition at any state
                    summation += new_xi[i][j] # sum of xi for every state j that we can go to is equal to gamma
                new_gamma[i] = summation
            xi.append(new_xi)
            gamma.append(new_gamma)
        
        pi_tilde = {} # expected frequency in state i at time t=0
        final_gamma = {}
        for i in self.states:
            pi_tilde[i] = gamma[0][i] # find gamma for first time step
            final_gamma[i] = ( self.alpha[len(obs)-1][i] * self.beta[len(obs)-1][i] ) / denominator # find final gamma value for lat time step T
        gamma.append(final_gamma)

        a_tilde = {}
        for i in self.states:
            # initialize a_tilde for this i state
            a_tilde[i] = {}
            a_denom = sum( gamma[t][i] for t in range(len(obs) - 1) ) #  expected number of transitions from state i
            for j in self.states:
                numer = sum( xi[t][i][j] for t in range(len(obs) - 1) ) #  expected number of transitions from state i TO state j
                a_tilde[i][j] = numer / a_denom
        
        b_tilde = {}
        for i in self.states:
            b_tilde[i] = {}
            b_denom = sum( gamma[t][i] for t in range(len(obs)) ) #  expected number of times in state i
            for k in self.symbols:
                summation = 0.0
                # this goes from index 0:T-1 or O(1):O(T)
                for t in range(len(obs)):
                    if obs[t] == k:  # expected number of times in state i and observing symbol k
                        summation += gamma[t][i]
                b_tilde[i][k] = summation / b_denom
                
        self.A = a_tilde
        self.B = b_tilde
        self.pi = pi_tilde
        return