import numpy as np
import os


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


def find_distance_2_vectors(vector1, vector2):
    # distance.euclidean(vector1, vector2)
    return np.linalg.norm(vector1 - vector2)


# Generates a random vector.
def get_rand_vec(dims):
    return np.random.uniform(-1, 1, size=dims)


# Determine which side of the line the point is on
def determine_pnt_side(v1, p1):
    # print(len(v1), len(p1))
    return np.dot(v1, p1)


# Return list of files without extension
def ret_files_without_ext(file_list):
    no_ext_files = []
    for file in file_list:
        no_ext_files.append(os.path.splitext(file)[0])
    return no_ext_files


def add_bin_num(arr, n, allcombo):
    bin_str = ''
    for i in range(0, n):
        bin_str += str(arr[i])
    allcombo.add(bin_str)


# Return all binary numbers having n bits
def ret_n_size_bin_strings(n, arr, i, allcombo):
    if (i == n):
        add_bin_num(arr, n, allcombo)
        return

    arr[i] = 0
    ret_n_size_bin_strings(n, arr, i + 1, allcombo)

    arr[i] = 1
    ret_n_size_bin_strings(n, arr, i + 1, allcombo)