import json

import pandas as pd
import numpy as np
import os
from os.path import join
import random

from Main import config
from Main.helper import progress


def startTask2():
    d = []
    p = []
    hogDescriptorsForDorsal = []
    hogDescriptorsForPalmar = []
    hogDescriptorForUnlabelled = []

    unlabeledImageMap = {}

    dorsal_image_index = {}
    palmar_image_index = {}

    metadata = pd.read_csv(config.METADATA_FOLDER)
    for index, row in metadata.iterrows():
        if row['aspectOfHand'] == 'dorsal right' or row['aspectOfHand'] == 'dorsal left':
            d.append(row['imageName'])
        elif row['aspectOfHand'] == 'palmar right' or row['aspectOfHand'] == 'palmar left':
            p.append(row['imageName'])

    dorsal = set(d)
    palmar = set(p)
    # K = int(input("Enter the number of clusters: "))
    # folderName = input("Enter the folder path containing the train images")
    # test_folder_name = input("Enter the folder path containing the test images")
    K = 10
    folderName = config.IMAGE_FOLDER_SET_1
    test_folder_name = config.IMAGE_FOLDER_SET_UNLABELLED_2
    i = 0
    j = 0
    counter = 1
    training_files = os.listdir(folderName)
    print("Calculating features for labelled images")
    for imageID in training_files:
        data = {}
        fileExists = os.path.exists(join(config.FEATURES_FOLDER, imageID + ".json"))
        if fileExists:
            with open(join(config.FEATURES_FOLDER, imageID + ".json"), "r") as f:
                data = json.load(f)
        progress(counter, len(training_files))
        counter = counter + 1
        if imageID in dorsal:
            dorsal_image_index[i] = imageID
            i += 1
        else:
            palmar_image_index[j] = imageID
            j += 1
        if imageID in dorsal:
            hogDescriptorsForDorsal.append(list(data.values()))
        elif imageID in palmar:
            hogDescriptorsForPalmar.append(list(data.values()))

    i = 0
    counter = 1
    print("Calculating features for unlabelled images")
    test_files = os.listdir(test_folder_name)
    for imageID in test_files:
        data = {}
        fileExists = os.path.exists(join(config.FEATURES_FOLDER, imageID + ".json"))
        if fileExists:
            with open(join(config.FEATURES_FOLDER, imageID + ".json"), "r") as f:
                data = json.load(f)
        hogDescriptorForUnlabelled.append(data.values())
        unlabeledImageMap[i] = [imageID]
        i = i + 1
        progress(counter, len(test_files))
        counter = counter + 1

    m = np.array(hogDescriptorsForDorsal)
    n = np.array(hogDescriptorsForPalmar)

    centers_Dorsal = train_k_means_clustering(hogDescriptorsForDorsal, K, 100)
    centers_Palmar = train_k_means_clustering(hogDescriptorsForPalmar, K, 100)

    print("----------------------------------------------------------------------------------------------------")
    print("Dorsal Centers - ")
    print(centers_Dorsal)
    print("----------------------------------------------------------------------------------------------------")
    print("Palmar Centers - ")
    print(centers_Palmar)

    ListClusters_Dorsal = []
    ListClusters_Palmar = []

    # Dorsal
    cluster_centers_Dorsal = []
    for cluster in centers_Dorsal:
        cluster_centers_Dorsal.append(cluster[:len(hogDescriptorForUnlabelled[0])])
    cluster_distance_Dorsal = {}
    for hog in hogDescriptorForUnlabelled:
        cluster_distance_Dorsal, nearest_Dorsal = predict_k_means_clustering(hog, cluster_centers_Dorsal)
        ListClusters_Dorsal.append(cluster_distance_Dorsal)

    # Palmar
    cluster_centers_Palmar = []
    for cluster in centers_Palmar:
        cluster_centers_Palmar.append(cluster[:len(hogDescriptorForUnlabelled[0])])
    cluster_distance_Palmar = {}
    for hog in hogDescriptorForUnlabelled:
        cluster_distance_Palmar, nearest_Palmar = predict_k_means_clustering(hog, cluster_centers_Palmar)
        ListClusters_Palmar.append(cluster_distance_Palmar)

    for i in range(0, len(hogDescriptorForUnlabelled)):
        print('***** Dorsal ', ListClusters_Dorsal[i])
        print('***** Palmar ', ListClusters_Palmar[i])
        print('============================================')

    # ## Find top K clusters in Dorsal and Palmer Centers ##

    # In[20]:

    topKPointsDorsal = []
    topKPointsPalmar = []

    for cluster in ListClusters_Dorsal:
        for key in list(cluster.keys()):
            topKPointsDorsal.append(['Dorsal', key, cluster[key]])

    for cluster in ListClusters_Palmar:
        for key in list(cluster.keys()):
            topKPointsPalmar.append(['Palmar', key, cluster[key]])

    ImageOrientation_List = {}  # this list will hold       Dorsal : [index of images which are dorsal]
    #  Palmar : [index of images which are palmar]

    ImageOrientation_List['Dorsal'] = []
    ImageOrientation_List['Palmar'] = []

    for i in range(0, len(hogDescriptorForUnlabelled)):
        L = []
        for key in list(ListClusters_Dorsal[i].keys()):
            L.append(['Dorsal', ListClusters_Dorsal[i][key]])
        for key in list(ListClusters_Palmar[i].keys()):
            L.append(['Palmar', ListClusters_Palmar[i][key]])

        L.sort(key=lambda x: x[1])
        print(L)
        print('===========================')

        # Extract 3 nearest neighbours
        k = 1
        dorsal = 0
        palmar = 0
        for j in range(0, k):
            if L[j][0] == 'Dorsal':
                dorsal += 1
            elif L[j][0] == 'Palmar':
                palmar += 1
        print(dorsal, palmar)
        if dorsal > palmar:
            ImageOrientation_List['Dorsal'].append(i)
        else:
            ImageOrientation_List['Palmar'].append(i)

    data = pd.read_csv(config.METADATA_FOLDER)
    test_data = data.iloc[:, [7, 8]].values

    test_data_hash = {}
    for i in test_data:
        test_data_hash[i[1]] = i[0]
    ImageOrientation_List

    correct = 0

    ImageOrientation_List['Dorsal']
    for l in ImageOrientation_List['Dorsal']:
        if test_data_hash[unlabeledImageMap[l][0]] == 'dorsal right' or test_data_hash[
            unlabeledImageMap[l][0]] == 'dorsal left':
            correct += 1

    for l in ImageOrientation_List['Palmar']:
        if test_data_hash[unlabeledImageMap[l][0]] == 'palmar right' or test_data_hash[
            unlabeledImageMap[l][0]] == 'palmar left':
            correct += 1

    print('Accuracy: ', correct / len(test_data_hash))
    visualizeMapDorsal = []
    visualizeMapPalmar = []

    for i in range(0, K):
        dorsalDict = {}
        palmarDict = {}
        for j in range(len(hogDescriptorsForDorsal)):
            if hogDescriptorsForDorsal[j][len(hogDescriptorsForDorsal[j]) - 1] == i:
                dorsalDict[dorsal_image_index[j]] = 'Dorsal'

        visualizeMapDorsal.append(dorsalDict)

        for j in range(len(hogDescriptorsForPalmar)):
            if hogDescriptorsForPalmar[j][len(hogDescriptorsForPalmar[j]) - 1] == i:
                palmarDict[palmar_image_index[j]] = 'Palmar'

        visualizeMapPalmar.append(palmarDict)

    plotInChromeForTask2(visualizeMapDorsal, visualizeMapPalmar, 'Task 2', 0)


def plotInChromeForTask2(dorsal_map, palmer_map, task, test_accuracy):
    s = "<style>"         ".images { width:160px;height:120px;float:left;margin:20px;}"         "img{width:160px;height:120px;}"        "h2,h5{text-align:center;margin-top:60px;}"        "</style>"
    s = s + "<h2 style='text-align:center;margin-top:60px;'> Classified Images using "+task+" </h2><div class='container'>"
    for row in range(len(dorsal_map)):
        s = s + "<h2 style='clear:both;'> Cluster :  "+str(row+1)+"</h2>"
        s = s + "<h5 style='clear:both;'> Dorsal </h5>"
        for key in dorsal_map[row]:
            news = "<div class='images'>"
            news = news + "<img src='"
            news = news + join("/Users/ajinkyadande/Documents/ASU/CSE 515 - Multimedia and Web Databases/Project/Phase 1/Hands", key)
            news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>"+dorsal_map[row][key]+"</span></div>"
            news = news + "</div>"
            s = s + news
        s = s + "<h5 style='clear:both;'> Palmer </h5>"
        for palkey in palmer_map[row]:
            news = "<div class='images'>"
            news = news + "<img src='"
            news = news + join(config.FULL_IMAGESET_FOLDER, palkey)
            news = news + "'><div style='text-align:center;'> Class: <span style='font-weight:bold;'>"+palmer_map[row][palkey]+"</span></div>"
            news = news + "</div>"
            s = s + news


    s = s + "</div>"
    f = open(task+".html", "w")
    f.write(s)
    f.close()

def random_centers(data, dim, k):
    centers = []
    for i in range(k):
        rand = random.randint(0, dim)
        centers.append(data[rand])
    return centers


def point_clustering(data, centers, dims, first_cluster=False):
    for point in data:
        # print(type(centers))
        nearest_center = 0
        nearest_center_dist = None
        for i in range(0, len(centers)):
            euclidean_dist = 0
            for d in range(0, dims):
                dist = abs(point[d] - centers[i][d])
                euclidean_dist += dist
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
    datapoints = len(data)
    features = len(data[0])

    centers = random_centers(data, datapoints, k)
    clustered_data = point_clustering(data, centers, features, first_cluster=True)

    for i in range(epochs):
        print("Iteration - ", i)
        centers = mean_center(clustered_data, centers, features)
        clustered_data = point_clustering(data, centers, features, first_cluster=False)

    return centers


def predict_k_means_clustering(point, centers):
    cluster_distance = {}
    dims = len(point)
    center_dims = len(centers[0])

    if dims != center_dims:
        raise ValueError('Point given for prediction have', dims, 'dimensions but centers have', center_dims,
                         'dimensions')

    nearest_center = None
    nearest_dist = None

    for i in range(len(centers)):
        euclidean_dist = 0
        for dim in range(1, dims):
            dist = point[dim] - centers[i][dim]
            euclidean_dist += dist ** 2
        euclidean_dist = np.sqrt(euclidean_dist)
        if nearest_dist == None:
            nearest_dist = euclidean_dist
            nearest_center = i
        elif nearest_dist > euclidean_dist:
            nearest_dist = euclidean_dist
            nearest_center = i
        cluster_distance[i] = euclidean_dist
    return cluster_distance, nearest_center


if __name__ == '__main__':
    startTask2()
