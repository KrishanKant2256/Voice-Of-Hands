import numpy as np

def extract_landmarks(results):
    features = []

    for hand in results.multi_hand_landmarks[:2]:
        for lm in hand.landmark:
            features.extend([lm.x, lm.y, lm.z])

    while len(features) < 126:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features).reshape(1, 126)
