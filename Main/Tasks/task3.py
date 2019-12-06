import json
import os
from os.path import join

import numpy as np
import pandas as pd
import Main.config as config
from Main.HOG_feature_descriptor import HOG
from Main.PCA_Reducer import PCA_Reducer
from Main.helper import find_distance_2_vectors


def startTask3():
    print("start task3")
    k = input("Please enter the k value for outgoing edges ")
    K = input("Please enter the K value for visualizing dominant images ")
    k = int(k)
    K = int(K)
    folderPath = input("Please Select the folder to apply Page Rank  ")
    # if folder == "1":
    #     folderPath = config.IMAGE_FOLDER_SET_1
    # else:
    #     folderPath = config.IMAGE_FOLDER_SET_2

    data = {}

    for file in os.listdir(str(folderPath)):
        filename = os.fsdecode(file)
        fileExists = os.path.exists(join(config.FEATURES_FOLDER, file + ".json"))
        if fileExists:
            with open(join(config.FEATURES_FOLDER, filename+".json"), "r") as f:
                eachData = json.load(f)
                data.update(eachData)
        else:
            data = HOG().HOGForSingleImage(folderPath, file)
            data.update(eachData)
            # mergingFeatureJson.append(data)

    # print(mergingFeatureJson)

    # fileHOGFullExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG.json"))
    #
    # fileExists = os.path.exists(join(config.DATABASE_FOLDER, "HOG_set_2.json"))
    # if not fileExists:
    #     hog = HOG()
    #     featureVector = hog.HOGFeatureDescriptor()
    #
    #     with open(join(config.DATABASE_FOLDER, "HOG_set_2.json"), 'w', encoding='utf-8') as f:
    #         json.dump(featureVector, f, ensure_ascii=True, indent=4)
    #
    # with open(join(config.DATABASE_FOLDER, "HOG_set_2.json"), "r") as f:
    #     data = json.load(f)

    reducerObject = list(data.values())

    pca = PCA_Reducer(reducerObject)
    latentFeatureDict = {}
    data = pca.reduceDimension(pca.featureDescriptor)
    i = 0
    imageNames = []
    for file in os.listdir(str(folderPath)):
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
        total = 0
        for distance_index in ind:
            if distances[distance_index] != 0:
                total += 1 / distances[distance_index]
        for distance_index in ind:
            # This is adding only k nearest neighbours into the matrix and doing ratio to get probablistic matrix
            if distances[distance_index] != 0:
                adjacency_matrix[distance_index][i] = 1 / distances[distance_index] / total

    rowDict = {}
    i = 0
    for image in imageNames:
        rowDict[i] = image
        i = i + 1

    df = pd.DataFrame(adjacency_matrix, columns=imageNames)
    df.rename(index=rowDict, inplace=True)

    df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix.csv"))

    I = np.identity(df.shape[1])

    print("Enter the three imageIDs to be used as seed")
    imageID_1 = input()
    imageID_2 = input()
    imageID_3 = input()

    seed = pd.Series(0, index=df.index)
    seed.loc[imageID_1] = 0.33
    seed.loc[imageID_2] = 0.33
    seed.loc[imageID_3] = 0.34
    page_rank = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed)
    # ind = np.argpartition(page_rank, -K)[-K:]
    # print(page_rank[ind])
    steady_state = pd.Series(page_rank, index=df.index)
    # df.rename(columns={0:"imageName",1:"values"}, inplace=True)
    # steady_state.nlargest(K, ["values"],keep="all")
    steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix.csv"))

    col_Names = ["imageNames", "values"]
    my_CSV_File = pd.read_csv(join(config.DATABASE_FOLDER, "steady_state_matrix.csv"), names=col_Names)
    kDominant = my_CSV_File.nlargest(K, ["values"], keep="all")

    # print(my_CSV_File.nlargest(K, ["values"], keep="all"))
    s = "<style>" \
        "img { width:160px;height:120px" \
        "</style>"
    s = s + "<h2> 3 Seed Images</h2>"
    s = s + "<img src='"
    s = s + join(folderPath, imageID_1)
    s = s + "'>"
    s = s + "<img src='"
    s = s + join(folderPath, imageID_2)
    s = s + "'>"
    s = s + "<img src='"
    s = s + join(folderPath, imageID_3)
    s = s + "'>"
    s = s + "</br></br>"
    s = s + "<h2>" + str(K) + " Dominant Images</h2>"
    for index, row in kDominant.iterrows():
        news = ""
        news = news + "<img src='"
        news = news + join(folderPath, row["imageNames"])
        news = news + "'>"
        s = s + news

    f = open(join(config.DATABASE_FOLDER, "task3.html"), "w")
    f.write(s)
    f.close()

    import webbrowser

    url = join(config.DATABASE_FOLDER, "task3.html")
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open(url)


if __name__ == '__main__':
    startTask3()
