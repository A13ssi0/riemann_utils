import numpy as np

def angle_between_matrices(m1, m2, tan_point):    
    numerator = np.trace(m1 @ np.linalg.inv(tan_point) @ m2 @ np.linalg.inv(tan_point))
    magnitude_v1 = np.trace(m1 @ np.linalg.inv(tan_point) @ m1 @ np.linalg.inv(tan_point))
    magnitude_v2 = np.trace(m2 @ np.linalg.inv(tan_point) @ m2 @ np.linalg.inv(tan_point))
    
    cos_theta = numerator / (np.sqrt(magnitude_v1) * np.sqrt(magnitude_v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    return angle*180/np.pi