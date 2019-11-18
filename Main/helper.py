import numpy as np


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


def find_distance_2_vectors(vector1, vector2):
    # distance.euclidean(vector1, vector2)
    return np.linalg.norm(vector1 - vector2)
