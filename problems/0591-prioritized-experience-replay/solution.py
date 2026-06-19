import numpy as np

def prioritized_replay_sample(priorities: list, batch_size: int, alpha: float = 0.6, beta: float = 0.4, seed: int = 42) -> dict:
	"""
	Sample a batch from a replay buffer using prioritized experience replay.

	Args:
		priorities: list of priority values for each experience (positive floats)
		batch_size: number of experiences to sample
		alpha: prioritization exponent (0 = uniform, 1 = full prioritization)
		beta: importance sampling exponent (0 = no correction, 1 = full correction)
		seed: random seed for reproducibility

	Returns:
		dict with 'indices', 'probabilities', and 'weights'
	"""
	np.random.seed(seed)
	N = len(priorities)
	p_arr = np.array(priorities)
	
	p_w = p_arr ** alpha / np.sum(p_arr ** alpha)
	idx = np.random.choice(N, size=batch_size, replace = False, p = p_w)
	samples = np.empty((batch_size))
	for i in range(batch_size):
		samples[i] = p_w[idx[i]]

	is_w = (1 /(N * samples)) ** beta
	w_h = is_w / np.max(is_w)
	return {
		'indices':idx,
		'probabilities':list(map(lambda x: round(x, 4), p_w.tolist())),
		'weights':list(map(lambda x: round(x, 4), w_h.tolist())),
	}



	pass