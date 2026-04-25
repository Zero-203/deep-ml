import torch
import torch.nn.functional as F

def polynomial_function_derivate(f_coeffs: list) -> list:
    res=[]
    for i in range(1,len(f_coeffs)):
        c=f_coeffs[i]
        df_coeff = 0.0 if c == 0.0 else c*i
        res.append(df_coeff)
    return res 

def polynomial_function_multiply(f_coeffs: list, g_coeffs: list) -> list:
    f_len,g_len=len(f_coeffs),len(g_coeffs)
    res=[0.0]*(f_len+g_len)
    for i in range(f_len):
        for j in range(g_len):
            res[i+j]+=f_coeffs[i]*g_coeffs[j]
    return res

def polynomial_function_add(f_coeffs: list, g_coeffs: list) -> list:
    f_len,g_len=len(f_coeffs),len(g_coeffs)
    res=[0.0]*max(f_len,g_len)
    for i in range(f_len):
        res[i]+=f_coeffs[i]
    for j in range(g_len):
        res[j]+=g_coeffs[j]
    return res

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> torch.Tensor:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        torch.Tensor of coefficients of (f*g)' rounded to 4 decimal places
    """
    # Your code here
    df,dg=polynomial_function_derivate(f_coeffs),polynomial_function_derivate(g_coeffs)
    pro_fg=polynomial_function_add(polynomial_function_multiply(df,g_coeffs),polynomial_function_multiply(f_coeffs,dg))

    while(len(pro_fg)>0 and pro_fg[-1]==0.0):
        del pro_fg[-1]
    pro_fg=pro_fg if len(pro_fg) > 0 else [0.0]
    return torch.tensor(pro_fg)