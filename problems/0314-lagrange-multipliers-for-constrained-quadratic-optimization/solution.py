import torch

def lagrange_optimize(Q: torch.Tensor, c: torch.Tensor, a: torch.Tensor, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix (torch.Tensor)
        c: 2-element vector (linear coefficients, torch.Tensor)
        a: 2-element vector (constraint coefficients, torch.Tensor)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    a,c,b=a.reshape((2,1)),c.reshape((2,1)),torch.as_tensor(b).reshape((1,1))
    # print(Q,-a,a.T,torch.tensor([0.0]).reshape((1,1)))
    A_high=torch.cat((Q,-a),1)
    A_low=torch.cat((a.T,torch.tensor([0.0]).reshape((1,1))),1)

    A=torch.cat((A_high,A_low),0)
    B=torch.cat((-c,b),0)
    x_e=torch.linalg.inv(A)@B
    # print(x_e)
    x,l=x_e[0:-1],x_e[-1]
    obj=0.5*torch.mm(x.T,torch.mm(Q,x))+torch.mm(c.T,x)
    x_list=x.tolist()
    return {
        "x":list(map(lambda val:round(val,4),x.flatten().tolist())),
        "lambda":round(l.item(),4),
        "objective":round(obj.item(),4)
        }