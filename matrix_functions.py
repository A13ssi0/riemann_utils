import numpy as np
from pyriemann.utils.distance import distance_riemann


def metric_riemann(m1, m2, tan_point):
    return np.trace(m1 @ np.linalg.inv(tan_point) @ m2 @ np.linalg.inv(tan_point))

def angle_between_matrices(m1, m2, tan_point):    
    numerator = metric_riemann(m1, m2, tan_point)
    magnitude_v1 = metric_riemann(m1, m1, tan_point)
    magnitude_v2 = metric_riemann(m2, m2, tan_point)
    
    cos_theta = numerator / (np.sqrt(magnitude_v1) * np.sqrt(magnitude_v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    return angle, cos_theta

def matrix_std(matrices, center_point):
    # matrices: n_matrices x n x n
    for idx in range(len(matrices.shape[:-2])):
        if matrices.shape[idx] != center_point.shape[idx]:
            center_point = np.expand_dims(center_point, axis=idx)
            tiles = np.ones(len(center_point.shape), dtype=int)
            tiles[idx] = matrices.shape[idx]
            center_point = np.tile(center_point, tiles)

    distances = distance_riemann(matrices, center_point)**2
    return np.sqrt(np.array([sum(x) for x in distances])/(distances.shape[1]-1))

def evaluate_negative_angles():
    pass