from math import exp
from math import isnan

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    n = len(scores)
    naive_softmax, stable_softmax= [[]] * n, [[]] * n
    naive_sum, stable_sum, max_abs_diff = 0.0, 0.0, 0.0
    max_num = max(scores)
    naive_error, stable_error = False, False
    for i in range(n):
        try:
            naive_sum += exp(scores[i])
        except Exception as OverflowError:
            naive_error = True
        try:
            stable_sum += exp(scores[i] - max_num)
        except Exception as OverflowError:
            stable_error = True
            

    for i in range(n):
        if not naive_error:
            naive_softmax[i] = round(exp(scores[i])/naive_sum, 6)
        else:
            naive_softmax[i] = float('nan')
        if not stable_error:
            stable_softmax[i] = round(exp(scores[i] - max_num)/stable_sum,6)
        else:
            stable_softmax[i] = float('nan')
        if isnan(naive_softmax[i]) or isnan(stable_softmax[i]):
            max_abs_diff = float('nan')
        else:
            if max_abs_diff != float('nan'):
                max_abs_diff = max(max_abs_diff,abs(naive_softmax[i]-stable_softmax[i]))
    
    return {
        'naive':naive_softmax,
        'stable':stable_softmax,
        'max_abs_diff':max_abs_diff
    }