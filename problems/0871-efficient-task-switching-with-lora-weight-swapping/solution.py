import numpy as np

def lora_task_switch(W0: np.ndarray, adapters: list, task_sequence: list, x: np.ndarray, scaling: float = 1.0) -> np.ndarray:
	"""
	Apply a sequence of LoRA-adapted linear transformations to x, switching
	adapters according to task_sequence while keeping W0 frozen.

	Args:
		W0: Base frozen weight matrix of shape (d_out, d_in).
		adapters: List of (A, B) tuples; A has shape (r, d_in), B has shape (d_out, r).
		task_sequence: List of indices into `adapters` indicating which task to use.
		x: Input batch of shape (batch, d_in).
		scaling: Scalar multiplier applied to the low-rank update.

	Returns:
		numpy array of shape (len(task_sequence), batch, d_out) with stacked outputs.
	"""
	task_cnt, batch, d_out = len(task_sequence), x.shape[0], W0.shape[0]
	res = np.zeros((task_cnt, batch, d_out))
	for i in range(task_cnt):
		A,B = adapters[task_sequence[i]]
		y = x @ (W0 + scaling * B @ A).T
		res[i] = y
	return res