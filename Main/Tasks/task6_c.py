import json
import os
from os.path import join
import numpy as np
from Main.HOG_feature_descriptor import HOG
import pandas as pd
import Main.config as config
from Main.PCA_Reducer import PCA_Reducer
import matplotlib.pyplot as plt
from Main.helper import find_distance_2_vectors



def startTask3():
    print("start task3")
    k = input("Please enter the k value for outgoing edges ")
    # K = input("Please enter the K value for visualizing dominant images ")
    k = int(k)
    classify_folder = input("Enter the folder to classify images ")
    fileHOGFullExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG_FULL.json"))

    fileExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG_classify.json"))
    if not fileExists:
        hog = HOG()
        featureVector = hog.HOGFeatureDescriptor()
        featureVector_classify = hog.HOGFeatureDescriptorForFolder(join(config.CLASSIFICATION_FOLDER, classify_folder))
        featureVector.update(featureVector_classify)
        with open(join(config.DATABASE_FOLDER, "HOG_classify.json"), 'w+', encoding='utf-8') as f:
            json.dump(featureVector, f, ensure_ascii=True, indent=4)

    with open(join(config.DATABASE_FOLDER, "HOG_classify.json"), "r") as f:
        data = json.load(f)

    reducerObject = list(data.values())

    pca = PCA_Reducer(reducerObject)
    latentFeatureDict = {}
    data = pca.reduceDimension(pca.featureDescriptor)
    print(data.shape)
    i = 0
    imageNames = []
    for file in os.listdir(str(config.IMAGE_FOLDER)):
        filename = os.fsdecode(file)
        latent = data.iloc[i][:]
        imageNames.append(filename)
        latentFeatureDict[filename] = latent
        i = i + 1

    for file in os.listdir(join(config.CLASSIFICATION_FOLDER, classify_folder)):
        filename = os.fsdecode(file)
        latent = data.iloc[i][:]
        imageNames.append(filename)
        latentFeatureDict[filename] = latent
        i = i + 1

    adjacency_matrix = [[0 for _ in range(len(latentFeatureDict))] for _ in range(len(latentFeatureDict))]
    for i in range(len(latentFeatureDict)):
        distances = []
        for j in range(len(latentFeatureDict)):
            distances.append(find_distance_2_vectors(latentFeatureDict[imageNames[i]], latentFeatureDict[imageNames[j]]))

        distances = np.asarray(distances)
        ind = np.argpartition(distances, k)[:k]
        total = 0
        for distance_index in ind:
            if distances[distance_index] != 0:
                total += 1/distances[distance_index]
        for distance_index in ind:
            # This is adding only k nearest neighbours into the matrix and doing ratio to get probablistic matrix
            if distances[distance_index] != 0:
                adjacency_matrix[distance_index][i] = 1/distances[distance_index] / total

    rowDict = {}
    i = 0
    for image in imageNames:
        rowDict[i] = image
        i = i + 1

    df = pd.DataFrame(adjacency_matrix, columns=imageNames)
    df.rename(index=rowDict, inplace=True)

    df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix.csv"))

    I = np.identity(df.shape[1])

    print("Enter the file where the meta-data of the images is present")
    fileName = input()
    metaData = pd.read_csv(join(config.METADATA_FOLDER, fileName))
    metaData.set_index('imageName')
    count = metaData.loc[metaData['aspectOfHand'].str.contains("dorsal")].shape[0]
    print(count)
    seed = pd.Series(0, index=df.index)
    seed[metaData.loc[metaData['aspectOfHand'].str.contains("dorsal")].imageName] = 1 / count

    page_rank = np.matmul(np.linalg.inv(I - .50*df), 0.50*seed)
    steady_state = pd.Series(page_rank, index=df.index)

    steady_state = steady_state.sort_values(ascending=True)
    steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix.csv"))
    steady_state.plot()
    plt.show()
if __name__ == '__main__':
    startTask3()
