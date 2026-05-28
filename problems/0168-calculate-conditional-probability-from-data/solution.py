def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    x_cnt, y_cnt = 0, 0
    for sample in data:
      sx, sy = sample
      if sx == x:
        x_cnt+=1
        if sy == y:
          y_cnt+=1
    return round(y_cnt/x_cnt if x_cnt !=0 else 0.0, 4)