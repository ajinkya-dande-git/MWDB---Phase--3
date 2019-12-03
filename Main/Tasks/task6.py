import json
import os
from os.path import join

import pandas as pd
import numpy as np

from Main import config
from Main.PCA_Reducer import PCA_Reducer
from Main.helper import find_distance_2_vectors


def startTask6():
    print("Task 6")
    feedbackSystem = int(input(
        "Please select the relevance feedback system \n1.SVM Based \n2.Decision Tree Based \n3.PPR Based \n4.Probabilistic based\n"))

    filename = input("Please enter the name of the file (output of 5b)")
    with open(join(config.DATABASE_FOLDER, filename + ".json"), "r") as f:
        data = json.load(f)
    imagesNames = list(data.keys())
    reducerObject = list(data.values())

    pca = PCA_Reducer(reducerObject, k=len(imagesNames))
    latentFeatureDict = {}
    data = pca.reduceDimension(pca.featureDescriptor)

    relavantImages = []
    irrelavantImages = []

    # Below values are used for PPR, not to calculate everytime iteatively
    rowDict = {}
    calculated = False

    ch = "n"
    while ch == "n" or ch == "N":
        numberOfRelavant = int(input("Number of relevant images "))
        numberOfIrRelavant = int(input("Number of irrelevant images "))
        for i in range(numberOfRelavant):
            relavantImages.append(input("Please " + str(i + 1) + " relevant image "))
        for i in range(numberOfIrRelavant):
            irrelavantImages.append(input("Please " + str(i + 1) + " irrelevant image "))
        if feedbackSystem == 1:
            print("SVM Based Feedback system")
        elif feedbackSystem == 2:
            print("Decision Tree Based Feedback system")
        elif feedbackSystem == 3:
            print("PPR Based Feedback system")
            if not calculated:
                for i in range(len(imagesNames)):
                    latent = data.iloc[i][:]
                    latentFeatureDict[imagesNames[i]] = latent
                    rowDict[i] = imagesNames[i]

                adjacency_matrix = [[0 for _ in range(len(latentFeatureDict))] for _ in range(len(latentFeatureDict))]
                # print("")
                print("Generating Adjacency Matrix..")

                for i in range(len(latentFeatureDict)):
                    distances = []
                    for j in range(len(latentFeatureDict)):
                        # print(len(latentFeatureDict[imageNames[i]]), len(latentFeatureDict[imageNames[j]]))
                        distances.append(find_distance_2_vectors(latentFeatureDict[imagesNames[i]],
                                                                 latentFeatureDict[imagesNames[j]]))

                    distances = np.asarray(distances)
                    ind = np.argpartition(distances, 5)[:5]
                    total = 0
                    for distance_index in ind:
                        if distances[distance_index] != 0:
                            total += 1 / distances[distance_index]
                    for distance_index in ind:
                        # This is adding only k nearest neighbours into the matrix and doing ratio to get probablistic matrix
                        if distances[distance_index] != 0:
                            adjacency_matrix[distance_index][i] = 1 / distances[distance_index] / total
                    calculated = True
            
            seed = pd.Series(0, index=imagesNames)
            length = len(relavantImages)
            for img in relavantImages:
                seed.loc[img] = 1 / length

            df = pd.DataFrame(adjacency_matrix, columns=imagesNames)
            df.rename(index=rowDict, inplace=True)

            I = np.identity(df.shape[1])
            page_rank = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed)
            # ind = np.argpartition(page_rank, -K)[-K:]
            # print(page_rank[ind])
            steady_state = pd.Series(page_rank, index=df.index)
            # df.rename(columns={0:"imageName",1:"values"}, inplace=True)
            # steady_state.nlargest(K, ["values"],keep="all")
            steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix_6_c.csv"))
            steady_state = steady_state.sort_values(ascending=True)
            df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix_task6_c.csv"))

            # s = "<style>" \
            #     "img { width:160px;height:120px" \
            #     "</style>"
            # s = s + "<h2> 3 Relevant Images</h2>"
            # for i in relavantImages:
            #     s = s + "<img src='"
            #     s = s + join(config.FULL_IMAGESET_FOLDER, i)
            #     s = s + "'>"
            #     s = s + "</br></br>"
            # s = s + "<h2> 3 Irrelevant Images</h2>"
            # for i in irrelavantImages:
            #     s = s + "<img src='"
            #     s = s + join(config.FULL_IMAGESET_FOLDER, i)
            #     s = s + "'>"
            #     s = s + "</br></br>"
            #
            # for row in range(steady_state.shape[0]):
            #     news = ""
            #     news = news + "<img src='"
            #     news = news + join(config.FULL_IMAGESET_FOLDER, str(row))
            #     news = news + "'>"
            #     s = s + news
            #
            # f = open(join(config.DATABASE_FOLDER, "task3.html"), "w")
            # f.write(s)
            # f.close()
            #
            # import webbrowser
            #
            # url = join(config.DATABASE_FOLDER, "task3.html")
            # # MacOS
            # # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
            # # Windows
            # chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
            # # Linux
            # # chrome_path = '/usr/bin/google-chrome %s'
            # webbrowser.get(chrome_path).open(url)

        elif feedbackSystem == 4:
            pass
        else:
            print("Wrong input")
            exit()
        ch = input("Are you satisfied with the output? type Y for exit N for running again ")


if __name__ == '__main__':
    startTask6()
