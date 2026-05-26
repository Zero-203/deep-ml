def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    val_fre = {}
    for sample in samples:
        if val_fre.get(sample)==None:
            val_fre[sample]=1
        else:
            val_fre[sample]+=1
    
    res = []
    cnt = len(samples)
    for val,fre in val_fre.items():
        res.append((val,fre/cnt))
    res.sort()
    return res