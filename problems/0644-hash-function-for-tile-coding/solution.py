import numpy as np

def tile_coding_hash(state: list, num_tilings: int, tiles_per_dim: int, memory_size: int, state_bounds: list = None) -> list:
	"""
	Compute active tile indices using tile coding with hash-based index mapping.

	Args:
		state: list of floats, continuous state variables.
		num_tilings: int, number of tilings.
		tiles_per_dim: int, tiles per dimension per tiling.
		memory_size: int, total hash table size.
		state_bounds: list of (low, high) tuples per dimension (default: (0,1)).

	Returns:
		List of int, one active tile index per tiling.
	"""
	ans = []
	state_bounds = [(0,1)]*len(state) if state_bounds is None else state_bounds
	for idx in range(num_tilings):
		offset = idx/num_tilings
		cs =  [idx]
		for i,sta in enumerate(state):
			scaled_state = (sta - state_bounds[i][0]) * tiles_per_dim \
							/(state_bounds[i][1]-state_bounds[i][0]) + idx / num_tilings
			coordinate = int(scaled_state)
			cs.append(coordinate)
		
		h = 5381
		for item in cs:
			h = (h*33+item)%(1<<32)
		ans.append(h%memory_size)
	return ans
	