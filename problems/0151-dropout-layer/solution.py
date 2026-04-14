import torch
from functools import reduce

class DropoutLayer:
        
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        # Your code here
        self.p=p
        self.mask=None

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        # Your code here
        # Not training, pass original tensor x
        if(not training):
            self.mask=torch.ones_like(x)
            return  x

        # Count elements
        shape_list=x.shape
        total=reduce(lambda x,y: x*y,shape_list)
        # Generate mask by p
        self.mask=(torch.rand(total)>self.p).float().reshape(shape_list)
        # Calculate scale factor
        valid=torch.sum(self.mask)
        scale_factor=total/valid
        # Drop and scale
        x=x*self.mask*scale_factor
        return x


    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        # Your code here
        return grad*self.mask