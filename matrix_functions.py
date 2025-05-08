import numpy as np
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.tangentspace import log_map_riemann


def metric_riemann(m1, m2, tan_point):
    return np.trace(m1 @ np.linalg.inv(tan_point) @ m2 @ np.linalg.inv(tan_point))

def angle_between_matrices(m1, m2, tan_point, fullLogMap=True): 
    s1 = log_map_riemann(m1, tan_point, C12=fullLogMap)   
    s2 = log_map_riemann(m2, tan_point, C12=fullLogMap)  
    
    numerator = metric_riemann(s1, s2, tan_point)
    magnitude_v1 = metric_riemann(s1, s1, tan_point)
    magnitude_v2 = metric_riemann(s2, s2, tan_point)
    
    cos_theta = numerator / (np.sqrt(magnitude_v1) * np.sqrt(magnitude_v2))
    angle = np.arccos(cos_theta)
    
    return angle, cos_theta

def matrix_std(matrices, center_point):
    # matrices: ... x n_matrices x n x n
    for idx in range(len(matrices.shape[:-2])):
        if matrices.shape[idx] != center_point.shape[idx]:
            center_point = np.expand_dims(center_point, axis=idx)
            tiles = np.ones(len(center_point.shape), dtype=int)
            tiles[idx] = matrices.shape[idx]
            center_point = np.tile(center_point, tiles)

    distances = distance_riemann(matrices, center_point)**2
    return np.sqrt(np.sum(distances,axis=-1) /(distances.shape[-1]-1))

def matrix_meanAbsoluteDeviation(matrices, center_point):
    # matrices: ... x n_matrices x n x n
    for idx in range(len(matrices.shape[:-2])):
        if matrices.shape[idx] != center_point.shape[idx]:
            center_point = np.expand_dims(center_point, axis=idx)
            tiles = np.ones(len(center_point.shape), dtype=int)
            tiles[idx] = matrices.shape[idx]
            center_point = np.tile(center_point, tiles)

    distances = distance_riemann(matrices, center_point)
    return np.sum(distances,axis=-1) /(distances.shape[-1]-1)

def evaluate_negative_angles(angles, centroids, tan_point, positiveCen=None, positiveAngl=None):
    if positiveCen is None:     positiveCen = centroids[0,0,0]
    if positiveAngl is None:    positiveAngl = angles[0,0,0]
    for run_idx in np.ndindex(angles.shape):
        vec = centroids[run_idx]
        vec_angle = angles[run_idx]
        _, cos_subtract = angle_between_matrices(positiveCen, vec, tan_point)

        if cos_subtract - np.cos(positiveAngl)*np.cos(vec_angle) < 0:   # = sen(alpha) * sen(beta) che è negativo se alpha e beta non fanno parte dello stesso semipiano
            angles[run_idx] = -angles[run_idx]
    return angles