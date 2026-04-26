import torch

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	x=torch.as_tensor(logits, dtype=torch.float64)
	p=torch.softmax(x, dim=0)
	y=torch.zeros_like(x)
	y[target]=1
	return (p-y)