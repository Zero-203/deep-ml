import torch

class GradientBandit:
    def __init__(self, num_actions: int, alpha: float = 0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = torch.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
    
    def softmax(self) -> torch.Tensor:
        # Compute softmax probabilities from preferences
        softmaxed_preferences = torch.exp(self.preferences)/torch.sum(torch.exp(self.preferences))
        return softmaxed_preferences
    
    def select_action(self) -> int:
        # Sample an action according to the softmax distribution
        return torch.multinomial(self.softmax(),1).item()
        
    
    def update(self, action: int, reward: float):
        # Update action preferences using the gradient ascent update
        self.avg_reward = (self.avg_reward * self.time +reward)/(self.time + 1)
        self.time += 1
        pa = self.softmax()[action]
        self.preferences -= self.alpha * (reward - self.avg_reward) * pa
        self.preferences[action] += self.alpha * (reward - self.avg_reward)