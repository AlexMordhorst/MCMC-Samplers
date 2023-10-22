#%%
import numpy as np

#%%
def accept_reject_sampling(candidate_mu, candidate_sigma, scale_factor, target_distribution, proposal_distribution, trials = 1000000):
    samps = []
    for i in range(trials):
        candidate = np.random.normal(candidate_mu, candidate_sigma)
        prob_accept = target_distribution(candidate)/(scale_factor*proposal_distribution(candidate, candidate_mu, candidate_sigma))
        if np.random.random() < prob_accept:
            samps.append(candidate)
    return samps
#%%
def metropolis_sampling(candidate_sigma, target_distribution, trials = 1000000, burn_in = 1000):
    samps = [1]
    num_accept = 0
    for i in range(trials):
        candidate = np.random.normal(samps[-1], candidate_sigma)
        prob = min(1, target_distribution(candidate)/target_distribution(samps[-1]))
        if np.random.random() < prob:
            samps.append(candidate)
            num_accept += 1
        else:
            samps.append(samps[-1])
    samps = samps[burn_in+1:]
    return samps

#%%
