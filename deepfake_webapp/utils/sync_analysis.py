import numpy as np


def combine_features(landmarks, mfcc):

    if landmarks is None or mfcc is None:
        return None

    features = np.concatenate((landmarks, mfcc))

    features = features.reshape(1, -1)

    return features
