import numpy as np

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    # Standard formula to find the angle between three points
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def extract_features(landmarks):
    # 'landmarks' comes from the web as a list of points
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    features = []

    # 1. Distances between consecutive joints
    for i in range(len(points) - 1):
        features.append(np.linalg.norm(points[i] - points[i + 1]))

    # 2. Angles between joints to detect finger bends
    for i in range(2, len(points)):
        features.append(calculate_angle(points[i - 2], points[i - 1], points[i]))

    return np.array(features)