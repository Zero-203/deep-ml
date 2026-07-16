import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    res = np.zeros((num_steps+1),dtype=int)
    samples = np.zeros((len(transition_matrix)))
    for i in range(len(samples)):
        samples[i]=i
    res[0] = initial_state
    for i in range(1,num_steps+1):
        res[i] = np.random.choice(samples, p=transition_matrix[res[i-1]])
    return res