import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    # Your code here
    latencies.sort()
    n = len(latencies)-1
    if n<0:
        return {"P50":0.0,"P95":0.0,"P99":0.0}
    elif n==0:
        return {"P50":latencies[0],"P95":latencies[0],"P99":latencies[0]}

    pos50, pos95, pos99 = int(n*0.5),int(n*0.95),int(n*0.99)
    diff50, diff95, diff99 = n*0.5-pos50,n*0.95-pos95,n*0.99-pos99
    
    P50, P95, P99 = latencies[pos50]+diff50*(latencies[pos50+1]-latencies[pos50]),\
                    latencies[pos95]+diff95*(latencies[pos95+1]-latencies[pos95]),\
                    latencies[pos99]+diff99*(latencies[pos99+1]-latencies[pos99])
    return {
        "P50":round(P50, 4),
        "P95":round(P95, 4),
        "P99":round(P99, 4)
    }