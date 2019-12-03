import json
import math
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
    iteration = 0
    # Below values are used for PPR, not to calculate everytime iteatively
    rowDict = {}
    calculated = False

    ch = "n"
    while ch == "n" or ch == "N":
        iteration = iteration + 1
        numberOfRelavant = int(input("Number of relevant images "))
        numberOfIrRelavant = int(input("Number of irrelevant images "))
        for i in range(numberOfRelavant):
            relavantImages.append(input("Please " + str(i + 1) + " relevant image "))
        for i in range(numberOfIrRelavant):
            irrelavantImages.append(input("Please " + str(i + 1) + " irrelevant image "))

        relavantImages = set(relavantImages)
        irrelavantImages = set(irrelavantImages)

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

            df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix_task6_c.csv"))

            I = np.identity(df.shape[1])
            page_rank = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed)

            steady_state = pd.Series(page_rank, index=df.index)
            steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix_6_c_"+str(iteration)+".csv"))

            steady_state = steady_state.sort_values(ascending=False)
            finalResult = steady_state.to_dict()
            finalResult = list(finalResult.keys())
            plotTheResultInChrome(relavantImages,irrelavantImages,finalResult,iteration)

        elif feedbackSystem == 4:
            images_df = pd.read_json(join(config.DATABASE_FOLDER, filename + ".json"), "r")
            threshold = 0.02
            nQuery = []
            for q in range(images_df.shape[0]):
                nq = 0
                rq = 0
                irq = 0
                for column in images_df:
                    if images_df[column][q] >= threshold:
                        nq += 1
                        if column in relavantImages:
                            rq += 1
                        if column in irrelavantImages:
                            irq += 1
                pq = (rq + nq / images_df.shape[1]) / (len(relavantImages) + 1)
                uq = (irq + nq / images_df.shape[1]) / (len(irrelavantImages) + 1)
                if pq * (1 - uq) / (uq * (1 - pq) + 1) <= 0:
                    nQuery.append(0)
                else:
                    q = math.log((pq * (1 - uq)) / (uq * (1 - pq)), 10)
                    if q < 0:
                        nQuery.append(0)
                    elif q > 1:
                        nQuery.append(1)
                    else:
                        nQuery.append(q)
            finalResult = {}
            for i in range(len(imagesNames)):
                product = np.dot(nQuery,reducerObject[i])
                finalResult[imagesNames[i]] = product

            sortList = sorted(finalResult.items(), key=lambda x: x[1], reverse=True)
            finalResult = list(dict(sortList).keys())
            plotTheResultInChrome(relavantImages,irrelavantImages,finalResult,iteration)
        else:
            print("Wrong input")
            exit()

        ch = input("Are you satisfied with the output? type Y for exit N for running again ")


def plotTheResultInChrome(relavantImages,irrelavantImages,finalResult,iteration):
    s = "<style>" \
        ".images { width:160px;height:120px;float:left;margin:20px;}" \
        "img{width:160px;height:120px;}" \
        "h2{ text-align: center;margin-top: 45px;}" \
        "</style>"
    s = s + "<h2> " + str(len(relavantImages)) + " Relevant Images</h2>"
    for i in relavantImages:
        s = s + "<div class='images'>"
        s = s + "<img src='"
        s = s + join(config.FULL_IMAGESET_FOLDER, i)
        s = s + "'><div style='text-align:center;'><span style='font-weight:bold;'>" + i + "</span></div></div>"
    s = s + "</br></br><div style='clear:both;'></div>"
    s = s + "<h2> " + str(len(irrelavantImages)) + " Irrelevant Images</h2>"
    for i in irrelavantImages:
        s = s + "<div class='images'>"
        s = s + "<img src='"
        s = s + join(config.FULL_IMAGESET_FOLDER, i)
        s = s + "'><div style='text-align:center;'><span style='font-weight:bold;'>" + i + "</span></div></div>"
    s = s + "</br></br><div style='clear:both;'></div>"
    s = s + "<h2> Results :</h2>"
    for img in finalResult:
        news = "<div class='images'>"
        news = news + "<img src='"
        news = news + join(config.FULL_IMAGESET_FOLDER, img)
        news = news + "'><div style='text-align:center;'><span style='font-weight:bold;'>" + img + "</span></div></div>"
        s = s + news

    s = s + "</div>"

    f = open(join(config.DATABASE_FOLDER, "task_6_c_" + str(iteration) + ".html"), "w")
    f.write(s)
    f.close()

    import webbrowser

    url = join(config.DATABASE_FOLDER, "task_6_c_" + str(iteration) + ".html")
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open(url)

if __name__ == '__main__':
    startTask6()
