import json
import os
from os.path import join
import numpy as np
from Main.HOG_feature_descriptor import HOG
import pandas as pd
import pickle
import Main.config as config
from Main.PCA_Reducer import PCA_Reducer
from Main.helper import find_distance_2_vectors
import matplotlib.pyplot as plt
import networkx as nx


def startTask3():
    print("start task3")
    k = input("Please enter the k value for outgoing edges ")
    K = input("Please enter the K value for visualizing dominant images ")
    k = int(k)
    fileHOGFullExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG_FULL.json"))

    fileExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG.json"))
    if not fileExists:
        hog = HOG()
        featureVector = hog.HOGFeatureDescriptor()

        with open(join(config.DATABASE_FOLDER, "HOG.json"), 'w', encoding='utf-8') as f:
            json.dump(featureVector, f, ensure_ascii=True, indent=4)

    with open(join(config.DATABASE_FOLDER, "HOG.json"), "r") as f:
        data = json.load(f)

    reducerObject = list(data.values())

    pca = PCA_Reducer(reducerObject)
    latentFeatureDict = {}
    data = pca.reduceDimension(pca.featureDescriptor)
    i = 0
    imageNames = []
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        latent = data.iloc[i][:]
        imageNames.append(filename)
        latentFeatureDict[filename] = latent
        i = i + 1

    adjacency_matrix = [[0 for _ in range(len(latentFeatureDict))] for _ in range(len(latentFeatureDict))]
    for i in range(len(latentFeatureDict)):
        distances = []
        for j in range(len(latentFeatureDict)):
            # print(len(latentFeatureDict[imageNames[i]]), len(latentFeatureDict[imageNames[j]]))
            distances.append(find_distance_2_vectors(latentFeatureDict[imageNames[i]],
                                                     latentFeatureDict[imageNames[j]]))
        distances = np.asarray(distances)
        ind = np.argpartition(distances, k)[:k]
        total = sum(distances[ind])
        for distance_index in ind:
            # This is adding only k nearest neighbours into the matrix and doing ratio to get probablistic matrix
            adjacency_matrix[i][distance_index] = distances[distance_index] / total

    rowDict = {}
    i = 0
    for image in imageNames:
        rowDict[i] = image
        i = i + 1

    df = pd.DataFrame(adjacency_matrix, columns=imageNames)
    df.rename(index=rowDict, inplace=True)

    df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix.csv"))


if __name__ == '__main__':
    startTask3()
