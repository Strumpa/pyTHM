import numpy as np

def if_lisse(x, y, delta, a, b):
    """
    Transition douce entre a et b autour de la condition x > y.
    """
    u = (x - y) / delta
    return b + (a - b) * 0.5 * (1.0 + np.tanh(0.5 * u))

def max_lisse(x, y, delta):
    """
    Smooth MAX (log-sum-exp stable).
    """
    m = np.maximum(x, y)
    return m + delta * np.log1p(np.exp(-np.abs(x - y) / delta))

def min_lisse(x, y, delta):
    """
    Smooth MIN (log-sum-exp stable).
    """
    m = np.minimum(x, y)
    return m - delta * np.log1p(np.exp(-np.abs(x - y) / delta))