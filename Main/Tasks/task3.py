from os.path import join
import os
from Main.HOG_feature_descriptor import HOG
import pandas as pd
import pickle
import Main.config as config
from Main.PCA_Reducer import PCA_Reducer
from Main.helper import find_distance_2_vectors


def startTask3():
    print("start task3")
    k = input("Please enter the k value for outgoing edges ")
    K = input("Please enter the K value for visualizing dominant images ")
    hog = HOG()
    featureVector = hog.HOGFeatureDescriptor()
    pca = PCA_Reducer(featureVector)
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

    adjacency_matrix = [[None for _ in range(len(latentFeatureDict))] for _ in range(len(latentFeatureDict))]
    for i in range(len(latentFeatureDict)):
        for j in range(len(latentFeatureDict)):
            # print(len(latentFeatureDict[imageNames[i]]), len(latentFeatureDict[imageNames[j]]))
            adjacency_matrix[i][j] = find_distance_2_vectors(latentFeatureDict[imageNames[i]], latentFeatureDict[imageNames[j]])

    df = pd.DataFrame(adjacency_matrix, columns = imageNames)

    print(df)
    # fileHandler = open(join(config.DATABASE_FOLDER, 'hog_features_task_3'), 'wb')
    # pickle.dump(pca, fileHandler)
    for i in range(df.shape[1]):
        tmp = df.nlargest(k, [imageNames[i]])
        df.loc[df[imageNames[i]] , 'First Season'] = 1



if __name__ == '__main__':
    startTask3()
