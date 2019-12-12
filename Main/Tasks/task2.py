import json

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from os.path import join
from random import randrange
from Main import config
from Main.HOG_feature_descriptor import HOG
from Main.helper import progress


def load_features(folder_path):
    hog_feature_map = {}
    counter = 1
    training_files = os.listdir(folder_path)
    print("Extracting features!")
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
            hog_feature_map.update(data)

        progress(counter, len(training_files))
        counter = counter + 1
    hog_values = list(hog_feature_map.values())
    return hog_values, training_files


def get_labels(image_folder, metadata):
    image_labels = []
    for file in os.listdir(image_folder):
        file_name = os.fsdecode(file)
        label = metadata.loc[metadata['imageName'] == file_name]['aspectOfHand'].iloc[0]
        if "dorsal" in label:
            image_labels.append("dorsal")
        else:
            image_labels.append("palmar")
    return image_labels


def pca_transform(train_features, test_features):
    scaler = StandardScaler()
    scaler.fit(train_features)
    scaler.fit(test_features)
    scaled_data_train = scaler.transform(train_features)
    scaled_data_test = scaler.transform(test_features)

    pca = PCA(n_components=80)
    pca.fit(scaled_data_train)
    reduced_train_data = pca.transform(scaled_data_train)
    reduced_test_data = pca.transform(scaled_data_test)

    return reduced_train_data.tolist(), reduced_test_data.tolist()


def random_centers(data, dim, k):
    centers = []
    for i in range(k):
        rand = randrange(dim)
        centers.append(data[rand])
    return centers


def point_clustering(data, centers, dims, first_cluster=False):
    for point in data:
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = 0
            for d in range(0, dims):
                dist = abs(point[d] - centers[i][d])
                euclidean_dist += dist ** 2
            euclidean_dist = np.sqrt(euclidean_dist)
            if nearest_center_dist == None:
                nearest_center_dist = euclidean_dist
                nearest_center = i
            elif nearest_center_dist > euclidean_dist:
                nearest_center_dist = euclidean_dist
                nearest_center = i
        if first_cluster:
            point.append(nearest_center)
        else:
            point[-1] = nearest_center
    return data


def mean_center(data, centers, dims):
    # print('centers:', centers, 'dims:', dims)
    new_centers = []
    for i in range(len(centers)):
        new_center = []
        n_of_points = 0
        total_of_points = []
        for point in data:
            if point[-1] == i:
                n_of_points += 1
                for dim in range(0, dims):
                    if dim < len(total_of_points):
                        total_of_points[dim] += point[dim]
                    else:
                        total_of_points.append(point[dim])
        if len(total_of_points) != 0:
            for dim in range(0, dims):
                # print(total_of_points, dim)
                new_center.append(total_of_points[dim] / n_of_points)
            new_centers.append(new_center)
        else:
            new_centers.append(centers[i])

    return new_centers


def train_k_means_clustering(data, k, epochs=20):
    print('\nRunning K means Clustering.')
    datapoints = len(data)
    features = len(data[0])

    centers = random_centers(data, datapoints, k)
    clustered_data = point_clustering(data, centers, features, first_cluster=True)

    for i in range(epochs):
        centers = mean_center(clustered_data, centers, features)
        clustered_data = point_clustering(data, centers, features, first_cluster=False)
        progress(i, epochs - 1)
    return centers


def seperate_labels(train_features, training_file_names, train_labels):
    dorsal = []
    palmar = []
    features_dorsal = []
    features_palmar = []
    for i in range(len(training_file_names)):
        if train_labels[i] == 'dorsal':
            dorsal.append(training_file_names[i])
            features_dorsal.append(train_features[i])
        elif train_labels[i] == 'palmar':
            palmar.append(training_file_names[i])
            features_palmar.append(train_features[i])
    return dorsal, palmar, features_dorsal, features_palmar


def group_clusters(features, file_names, K, tag):
    clusters = []
    for i in range(0, K):
        dict = {}
        for j in range(len(features)):
            if features[j][-1] == i:
                dict[file_names[j]] = tag
        clusters.append(dict)
    return clusters


def plotInChromeForTask2(dorsal_map, palmer_map, task):
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "h2,h5{text-align:center;margin-top:80px;margin-bottom:80px;}" \
        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> Classified Images using " + task + " </h2><div class='container'>"
    for row in range(len(dorsal_map)):
        s = s + "<h2 style='clear:both;'> Cluster :  " + str(row + 1) + "</h2>"
        s = s + "<h5 style='clear:both;'> Dorsal </h5>"
        for key in dorsal_map[row]:
            news = "<div class='images'>"
            news = news + "<img src='"
            news = news + join(config.FULL_IMAGESET_FOLDER, key)
            news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
                   dorsal_map[row][key] + "</span></div>"
            news = news + "</div>"
            s = s + news
        s = s + "<h5 style='clear:both;'> Palmer </h5>"
        for palkey in palmer_map[row]:
            news = "<div class='images'>"
            news = news + "<img src='"
            news = news + join(config.FULL_IMAGESET_FOLDER, palkey)
            news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + \
                   palmer_map[row][palkey] + "</span></div>"
            news = news + "</div>"
            s = s + news
    s = s + "</div>"
    f = open(join(config.DATABASE_FOLDER, task + ".html"), "w")
    f.write(s)
    f.close()


def plotInChromeForTask2_Unlabelled(unlabelledmap, task, test_accuracy):
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "h2,h5{text-align:center;margin-top:60px;}" \
        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> Classified Images using " + task + "  Test Accuracy: " + str(
        test_accuracy) + " </h2><div class='container'>"
    for row in unlabelledmap:
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + join(
            "/Users/ajinkyadande/Documents/ASU/CSE 515 - Multimedia and Web Databases/Project/Phase 1/Hands", row)
        news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>" + unlabelledmap[
            row] + "</span></div>"
        news = news + "</div>"
        s = s + news

    s = s + "</div>"
    f = open(join(config.DATABASE_FOLDER, task + ".html"), "w")
    f.write(s)
    f.close()


def euclidean_distance(point, center):
    return np.linalg.norm(point - center)


def compute_distances_with_centers(dorsal_centers, palmar_centers, test_features):
    dorsal_distances = []
    palmar_distances = []
    for point in test_features:
        dorsal_distance = []
        palmar_distance = []
        for dcenter in dorsal_centers:
            dist = euclidean_distance(np.asarray(point), np.asarray(dcenter))
            dorsal_distance.append(dist)
        dorsal_distance.sort()
        dorsal_distances.append(dorsal_distance)

        for pcenter in palmar_centers:
            dist = euclidean_distance(np.asarray(point), np.asarray(pcenter))
            palmar_distance.append(dist)
        palmar_distance.sort()
        palmar_distances.append(palmar_distance)

    return dorsal_distances, palmar_distances


def k_nearest_neighbours(dorsal_distances, palmar_distances, K):
    computed_labels = []
    i = 0
    j = 0

    for iter in range (0, len(dorsal_distances)):
        dorsal = 0
        palmar = 0
        while (i+j) < K:
            if dorsal_distances[iter][i] < palmar_distances[iter][j]:
                dorsal += 1
                i += 1
            else:
                palmar += 1
                j += 1
        if dorsal < palmar:
            computed_labels.append('dorsal')
        else:
            computed_labels.append('palmar')

    return computed_labels

def compute_accuracy(test_labels, computed_test_labels):
    positive_prediction = 0
    for i in range(len(test_labels)):
        if test_labels[i] == computed_test_labels[i]:
            positive_prediction += 1
    return positive_prediction


def startTask2():
    print('Starting Task 2')
    print("Enter the folder path containing the labeled images: ")  # Labelled images
    training_folder = input()  # /Users/ajinkyadande/Documents/ASU/CSE 515 - Multimedia and Web Databases/Project/Phase 3/phase3_sample_data/Labelled/Set1

    print("Enter the folder path containing the test images")  # Unlabelled images
    test_folder = input()  # /Users/ajinkyadande/Documents/ASU/CSE 515 - Multimedia and Web Databases/Project/Phase 3/phase3_sample_data/Unlabelled/Set 1

    # generating/loading the features and the file names
    train_features, training_file_names = load_features(training_folder)  # list of list
    test_features, testing_file_names = load_features(test_folder)

    # PCA
    pca = input('\nPCA? y/n ')
    if (pca == 'y'):
        train_features, test_features = pca_transform(train_features, test_features)

    # loading the metadata file
    metadata = pd.read_csv(config.METADATA_FOLDER)
    train_labels = get_labels(training_folder, metadata)
    test_labels = get_labels(test_folder, metadata)

    # Generating K clusters for Dorsal and Palmar Hand Images
    K = int(input('\nEnter the number of clusters '))
    dorsal_file_names, palmar_file_names, dorsal_features, palmar_features = seperate_labels(train_features,
                                                                                             training_file_names,
                                                                                             train_labels)
    dorsal_centers = train_k_means_clustering(dorsal_features, K, 100)
    palmar_centers = train_k_means_clustering(palmar_features, K, 100)

    # Visualize Dorsal and Palmar clusters
    dorsal_clusters = group_clusters(dorsal_features, dorsal_file_names, K, 'Dorsal')
    palmar_clusters = group_clusters(palmar_features, palmar_file_names, K, 'Palmar')
    plotInChromeForTask2(dorsal_clusters, palmar_clusters, 'Task2')

    # Finding accuracy for Unlabelled images
    dorsal_distances, palmar_distances = compute_distances_with_centers(dorsal_centers, palmar_centers, test_features)

    # knn
    k = 3
    computed_test_labels = k_nearest_neighbours(dorsal_distances, palmar_distances, k)
    accuracy = compute_accuracy(test_labels, computed_test_labels)
    print('\nAccuracy of the model is ', accuracy)

    test_images_dict = {}
    index = 0
    for imageID in testing_file_names:
        test_images_dict[imageID] = computed_test_labels[index]
        index += 1

    # plotting the unlabelled images
    plotInChromeForTask2_Unlabelled(test_images_dict, 'Task2_Unlabelled', accuracy)


if __name__ == '__main__':
    startTask2()
