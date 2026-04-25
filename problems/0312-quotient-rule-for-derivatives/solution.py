import torch

def polynomial_function_derivate(f_coeffs: list) -> list:
    res=[]
    for i in range(1,len(f_coeffs)):
        c=f_coeffs[i]
        df_coeff = 0.0 if c == 0.0 else c*i
        res.append(df_coeff)
    return res 

def cal_val_polynomial_function(f_coeffs: list, x: float) -> list:
    res=0.0
    for i in range(len(f_coeffs)):
        res+=f_coeffs[i]*x**i
    return res

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Your code here
    g_coeffs.reverse()
    h_coeffs.reverse()
    dg,dh=polynomial_function_derivate(g_coeffs),polynomial_function_derivate(h_coeffs)
    val_g=cal_val_polynomial_function(g_coeffs,x)
    val_h=cal_val_polynomial_function(h_coeffs,x)
    val_dg=cal_val_polynomial_function(dg,x)
    val_dh=cal_val_polynomial_function(dh,x)
    
    return torch.tensor((val_dg*val_h-val_dh*val_g)/(val_h**2))