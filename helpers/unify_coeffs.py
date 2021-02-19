import numpy as np


def unify_coeffs(coeffs, unified_length=500):
    if unified_length < coeffs.shape[0]:
        return coeffs[:unified_length]
    result = np.zeros((500, coeffs.shape[1]))
    result[:coeffs.shape[0]] = coeffs[:]
    return result