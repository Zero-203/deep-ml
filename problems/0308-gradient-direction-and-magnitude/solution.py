import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	grad=torch.tensor(gradient)
	mag=torch.sqrt(torch.sum(grad**2)).item()
	dirc=grad/mag if mag != 0.0 else torch.zeros_like(grad)
	des_dirc=-dirc
	return {'magnitude':mag,'direction':dirc,'descent_direction':des_dirc}