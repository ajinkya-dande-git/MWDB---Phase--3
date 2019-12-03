import os
from os.path import join

import numpy as np

from Main import config


def progress(count, total):
    percent = (count / total) * 100
    print('\r', "%.2f" % round(percent, 2) + "% completed", end=' ')


def find_distance_2_vectors(vector1, vector2):
    # distance.euclidean(vector1, vector2)
    return np.linalg.norm(vector1 - vector2)


def plotInChromeForTask4(test_labels_map,task,test_accuracy):
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}"\
        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> Classified Images using "+task+" Test Accuracy :  "+str(test_accuracy)+" </h2><div class='container'>"

    for key in test_labels_map:
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + join(config.FULL_IMAGESET_FOLDER, key)
        news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>"+test_labels_map[key]+"</span></div>"
        news = news + "</div>"
        s = s + news

    s = s + "</div>"
    f = open(join(config.DATABASE_FOLDER, task+".html"), "w")
    f.write(s)
    f.close()

    import webbrowser

    url = join(config.DATABASE_FOLDER, task+".html")
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open(url)


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