import json
import sys

import pandas as pd
import numpy as np
import os
from os.path import join
import random

from Main import config
from Main.HOG_feature_descriptor import HOG
from Main.helper import progress


def startTask2():
    dorsalImageID = []
    palmerImageID = []
    hogDescriptorsForDorsal = []
    hogDescriptorsForPalmar = []
    hogDescriptorForUnlabelled = []

    unlabeledImageMap = {}

    dorsal_image_index = {}
    palmar_image_index = {}

    print("starting task2")
    print("Enter the folder path containing the labeled images")
    training_folder = input()  # 'H:\Asu\mwdb\project-phase-2\MWDB---Phase--3\Images\Labelled\Set1' input()

    print("Enter the C, number of clusters : ")
    C = int(input())

    print("Enter the folder path containing the test images")
    test_folder = input()  # '/Users/vedavyas/Desktop/CSE515/phase3_data/Unlabelled/Set2'#input()

    train_features, training_file_names = load_features(training_folder)
    test_features, testing_file_names = load_features(test_folder)
    print()
    print("Computing K means...")
    train_features = pd.Series(train_features, index=training_file_names)

    metadata = pd.read_csv(join(config.METADATA_FOLDER))
    dorsal_image_id = get_labels(training_folder, metadata, "dorsal")
    palmer_image_id = get_labels(training_folder, metadata, "palmar")
    hogDescriptorsForDorsal = []
    hogDescriptorsForPalmar = []

    # print(len(dorsal_image_id))
    # print(dorsal_image_id)
    # print(palmer_image_id)
    # print(len(palmer_image_id))

    for i in range(len(dorsal_image_id)):
        hogDescriptorsForDorsal.append(train_features[dorsal_image_id[i]])

    for i in range(len(palmer_image_id)):
        hogDescriptorsForPalmar.append(train_features[palmer_image_id[i]])

    # print(len(hogDescriptorsForDorsal))
    # print(len(hogDescriptorsForPalmar))

    centers_Dorsal = train_k_means_clustering(np.array(hogDescriptorsForDorsal), C, 1000)
    centers_Palmar = train_k_means_clustering(np.array(hogDescriptorsForPalmar), C, 1000)

    dorsal_cluster_label = {}
    for i in range(len(hogDescriptorsForDorsal)):
        nearest_center = None
        nearest_center_dist = sys.maxsize
        for j in range(C):
            euclidean_dist = np.sqrt(np.sum(np.square(hogDescriptorsForDorsal[i] - centers_Dorsal[j])))
            if (nearest_center_dist is None) or (nearest_center_dist > euclidean_dist):
                nearest_center_dist = euclidean_dist
                nearest_center = j
        dorsal_cluster_label.update({dorsal_image_id[i]: nearest_center})

    palmar_cluster_label = {}

    for i in range(len(hogDescriptorsForPalmar)):
        nearest_center = None
        nearest_center_dist = sys.maxsize
        for j in range(C):
            euclidean_dist = np.sqrt(np.sum(np.square(hogDescriptorsForPalmar[i] - centers_Palmar[j])))
            if (nearest_center_dist is None) or (nearest_center_dist > euclidean_dist):
                nearest_center_dist = euclidean_dist
                nearest_center = j
        palmar_cluster_label.update({palmer_image_id[i]: nearest_center})
    print()
    print("predicting on test data")
    palmar_results = []
    dorsal_results = []
    accuracy = 0
    for i in range(len(test_features)):
        min_dist_dorsal = sys.maxsize
        for j in range(len(centers_Dorsal)):
            distance = np.sqrt(np.sum(np.square(np.array(test_features[i]) - centers_Dorsal[j])))
            if distance < min_dist_dorsal: min_dist_dorsal = distance

        min_dist_palmar = sys.maxsize
        for j in range(len(centers_Palmar)):
            distance = np.sqrt(np.sum(np.square(np.array(test_features[i]) - centers_Palmar[j])))
            if distance < min_dist_palmar: min_dist_palmar = distance

        if min_dist_palmar < min_dist_dorsal:
            palmar_results.append(testing_file_names[i])

            if "palmar" in metadata.loc[metadata['imageName'] == testing_file_names[i]]['aspectOfHand'].values[0]:
                accuracy = accuracy + 1
        else:
            dorsal_results.append(testing_file_names[i])

            if "dorsal" in metadata.loc[metadata['imageName'] == testing_file_names[i]]['aspectOfHand'].values[0]:
                accuracy = accuracy + 1

    print("Accuracy: ", accuracy / len(testing_file_names))

    # print(palmar_cluster_label)
    palmar_cluster_label = dict(sorted(palmar_cluster_label.items(), key=lambda x: x[1]))
    # print(palmar_cluster_label)
    # print(dorsal_cluster_label)
    dorsal_cluster_label = dict(sorted(dorsal_cluster_label.items(), key=lambda x: x[1]))
    # print(dorsal_cluster_label)
    # print(palmar_results)
    # print(dorsal_results)
    plotImagesForCluster(dorsal_cluster_label, palmar_cluster_label, "Task2_clusters", C)
    plotInChromeForTask2(dorsal_results, palmar_results, "Task 2_final", accuracy)


def plotImagesForCluster(dorsal_cluster_label, palmar_cluster_label, task, C):
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "h2{margin-top:20px;}"\
        "h2,h5,h3{text-align:center;margin-top:80px;margin-bottom:80px;}</style>"
    s = s + "<h2 style='text-align:center'> Classified Images using " + task + " </h2><div class='container'>"
    for i in range(C):
        s = s + "<h2 style='clear:both;'> Cluster :  " + str(i + 1) + "</h2>"
        s = s + "<h3 style='clear:both;'> Dorsal </h3>"
        for r in dorsal_cluster_label:
            if i == dorsal_cluster_label[r]:
                news = "<div class='images'>"
                news = news + "<img src='"
                news = news + join(config.FULL_IMAGESET_FOLDER, r)
                news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
                       r + "</span></div>"
                news = news + "</div>"
                s = s + news
        s = s + "<h3 style='clear:both;'> Palmer </h3>"
        for r in palmar_cluster_label:
            if i == palmar_cluster_label[r]:
                news = "<div class='images'>"
                news = news + "<img src='"
                news = news + join(config.FULL_IMAGESET_FOLDER, r)
                news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
                       r + "</span></div>"
                news = news + "</div>"
                s = s + news
        s = s + "</div>"
        f = open(join(config.DATABASE_FOLDER, task + ".html"), "w")
        f.write(s)
        f.close()
        import webbrowser

        url = join(join(config.DATABASE_FOLDER, task + ".html"))
        # MacOS
        # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
        # Windows
        chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
        # Linux
        # chrome_path = '/usr/bin/google-chrome %s'
        webbrowser.get(chrome_path).open(url)


def plotInChromeForTask2(dorsal_map, palmer_map, task, test_accuracy):
    s = "<style>"         ".images { width:160px;height:120px;float:left;margin:20px;}"         "img{width:160px;height:120px;}"        "h2,h5{text-align:center;margin-top:60px;}"        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> Classified Images using " + task + " with accuracy = " + str(
        test_accuracy) + " </h2><div class='container'>"
    s = s + "<h2 style='clear:both;'> Dorsal </h2>"
    for row in range(len(dorsal_map)):
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + join(config.FULL_IMAGESET_FOLDER, dorsal_map[row])
        news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
               dorsal_map[row] + "</span></div>"
        news = news + "</div>"
        s = s + news
    s = s + "<h2 style='clear:both;'> Palmer </h2>"
    for row in range(len(palmer_map)):
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + join(config.FULL_IMAGESET_FOLDER, palmer_map[row])
        news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
               palmer_map[row] + "</span></div>"
        news = news + "</div>"
        s = s + news

    s = s + "</div>"
    f = open(join(config.DATABASE_FOLDER, task + ".html"), "w")
    f.write(s)
    f.close()
    import webbrowser

    url = join(join(config.DATABASE_FOLDER, task + ".html"))
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open(url)


def random_centers(data, no_of_data_points, C):
    centers = []
    for i in range(C):
        rand = random.randint(0, no_of_data_points - 1)
        centers.append(data[rand])
    return np.array(centers)


def point_clustering(data, centers):
    nearest_centers = []
    for point in data:
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = np.sqrt(np.sum(np.square(point - centers[i])))
            if (nearest_center_dist is None) or (nearest_center_dist > euclidean_dist):
                nearest_center_dist = euclidean_dist
                nearest_center = i
        nearest_centers.append(centers[nearest_center])
    return np.array(nearest_centers)


def mean_center(data, old_centers, nearest_center):
    new_centers = []
    for i in range(len(old_centers)):
        no_of_points = 0
        center = np.zeros(len(old_centers[0]))
        for j in range(len(data)):
            if np.array_equal(nearest_center[j], old_centers[i]):
                no_of_points += 1
                center = center + data[j]
        new_centers.append(center / no_of_points)
    return np.array(new_centers)


def train_k_means_clustering(data, k, epochs=100):
    datapoints = len(data)
    centers = random_centers(data, datapoints, k)
    for i in range(epochs):
        # print("Iteration - ", i)
        nearest_centers = point_clustering(data, centers)
        centers = mean_center(data, centers, nearest_centers)
    return centers


def load_features(folder_path):
    hog_feature_map = {}
    counter = 1
    training_files = os.listdir(folder_path)
    print()
    print("Extracting features for the training images!")
    for trainingFile in training_files:
        trainingFileJson = os.fsdecode(trainingFile).split('.')[0] + '.' + os.fsdecode(trainingFile).split('.')[
            1] + '.json'

        fileExists = os.path.exists(join(config.FEATURES_FOLDER, trainingFileJson))
        data = {}
        if fileExists:
            with open(join(config.FEATURES_FOLDER, trainingFileJson), "r") as f:
                data = json.load(f)
                hog_feature_map.update(data)
        else:
            data = HOG().HOGForSingleImage(folder_path, trainingFile)
            hog_feature_map.update({trainingFile: data})

        progress(counter, len(training_files))
        counter = counter + 1
    hog_values = list(hog_feature_map.values())
    return hog_values, training_files


def get_labels(image_folder, metadata, column_value):
    image_labels = []
    for file in os.listdir(image_folder):
        file_name = os.fsdecode(file)
        label = metadata[(metadata['imageName'] == file_name) & (metadata['aspectOfHand'].str.contains(column_value))][
            'imageName'].values
        if len(label) == 1:
            image_labels.append(label[0])
    return image_labels


if __name__ == '__main__':
    startTask2()
