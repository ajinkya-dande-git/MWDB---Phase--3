import json
import math
import operator
from os.path import join

import pandas as pd
import numpy as np

from Main import config
from Main.PCA_Reducer import PCA_Reducer
from Main.Tasks.task4 import SVM, decisionTree
from Main.helper import find_distance_2_vectors


def startTask6():
    print("Task 6")
    feedbackSystem = int(input(
        "Please select the relevance feedback system \n1.SVM Based \n2.Decision Tree Based \n3.PPR Based \n4.Probabilistic based\n"))

    filename = input("Please enter the name of the file (output of 5b)")
    with open(join(config.DATABASE_FOLDER, filename), "r") as f:
        data = json.load(f)
    imagesNames = list(data.keys())
    reducerObjectpp = list(data.values())

    pca_pp = PCA_Reducer(reducerObjectpp, k=len(imagesNames))
    latentFeatureDict = {}
    data_pp = pca_pp.reduceDimension(pca_pp.featureDescriptor)

    relavantImages = set()
    irrelavantImages = set()
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
            relavantImages.add(input("Please " + str(i + 1) + " relevant image "))
        for i in range(numberOfIrRelavant):
            irrelavantImages.add(input("Please " + str(i + 1) + " irrelevant image "))

        if feedbackSystem == 1:
            print("SVM Based Feedback system")
            image_labels = []
            reducerObject = []
            for i in relavantImages:
                reducerObject.append(data.get(i))
                image_labels.append(-1)
            for i in irrelavantImages:
                reducerObject.append(data.get(i))
                image_labels.append(1)
            pca = PCA_Reducer(reducerObject, k=len(relavantImages) + len(irrelavantImages))
            pca_result = pca.reduceDimension(pca.featureDescriptor)
            svm_object = SVM()
            print("Training SVM")
            svm_object.svm_fit(pca_result, image_labels)
            print("Done Training SVM")
            tempList = list(set(imagesNames) - set(relavantImages))
            unlabelledImages = list(set(tempList) - set(irrelavantImages))
            predicted_values = []
            relevantDistances = {}
            irrelavantDistances = {}
            for i in unlabelledImages:
                pca_output = pca.reduceDimension([data.get(i)])
                # print("pca output: ", pca_output)
                output_label = np.asarray(svm_object.predict(pca_output))[0]
                # print(type(pca_output))
                pca_output = pca_output.values.tolist()
                distance = svm_object.distance(pca_output[0])
                # print(distance)
                # print(type(distance))
                predicted_values.append(output_label)
                if output_label == -1:
                    relevantDistances[distance] = i
                elif output_label == 1:
                    irrelavantDistances[distance] = i

            for i in relavantImages:
                pca_output = pca.reduceDimension([data.get(i)])
                pca_output = pca_output.values.tolist()
                distance = svm_object.distance(pca_output[0])
                # print(distance)
                relevantDistances[distance] = i

            for i in irrelavantImages:
                pca_output = pca.reduceDimension([data.get(i)])
                pca_output = pca_output.values.tolist()
                distance = svm_object.distance(pca_output[0])
                # print(distance)
                irrelavantDistances[distance] = i

            relevantDistancesList = sorted(relevantDistances, reverse=True)
            irrelavantDistancesList = sorted(irrelavantDistances)
            output_images_list = []
            for i in relevantDistancesList:
                output_images_list.append(relevantDistances.get(i))
            for i in irrelavantDistancesList:
                output_images_list.append(irrelavantDistances.get(i))
            # print(output_images_list)
            plotTheResultInChrome(relavantImages, irrelavantImages, output_images_list, iteration, "SVM")
        elif feedbackSystem == 2:
            print("Decision Tree Based Feedback system")
            reducerObject = []
            for i in relavantImages:
                reducerObject.append(data.get(i))
            for i in irrelavantImages:
                reducerObject.append(data.get(i))
            pca = PCA_Reducer(reducerObject, k=len(relavantImages) + len(irrelavantImages))
            pca_result = pca.reduceDimension(pca.featureDescriptor)
            pca_result = pca_result.values.tolist()
            class_labels = [-1, 1]
            for i in range(0, len(relavantImages)):
                pca_result[i].append(-1)
            count = len(relavantImages)
            for i in range(count, count + len(irrelavantImages)):
                pca_result[i].append(1)

            dtree_object = decisionTree()
            root = dtree_object.construct_dt(pca_result, class_labels, 2, 2)
            tempList = list(set(imagesNames) - set(relavantImages))
            unlabelledImages = list(set(tempList) - set(irrelavantImages))
            relevantConfidence = {}
            irrelevantConfidence = {}
            for i in unlabelledImages:
                pca_output = pca.reduceDimension([data.get(i)])
                pca_output = pca_output.values.tolist()[0]
                output_label = dtree_object.predict(root, pca_output)
                confidence = dtree_object.confidence(root, pca_output, output_label)
                if output_label == -1:
                    if relevantConfidence.get(confidence) is None:
                        relevantConfidence[i] = confidence
                else:
                    irrelevantConfidence[i] = confidence

            for i in relavantImages:
                pca_output = pca.reduceDimension([data.get(i)])
                pca_output = pca_output.values.tolist()[0]
                output_label = dtree_object.predict(root, pca_output)
                confidence = dtree_object.confidence(root, pca_output, output_label)
                relevantConfidence[i] = confidence

            for i in irrelavantImages:
                pca_output = pca.reduceDimension([data.get(i)])
                pca_output = pca_output.values.tolist()[0]
                output_label = dtree_object.predict(root, pca_output)
                confidence = dtree_object.confidence(root, pca_output, output_label)
                irrelevantConfidence[i] = confidence

            relevantConfidenceList = sorted(relevantConfidence.items(), key=operator.itemgetter(1), reverse=True)
            irrelavantConfidenceList = sorted(irrelevantConfidence.items(), key=operator.itemgetter(1))
            output_images_list = []
            # print(relevantConfidenceList)
            # print(irrelavantConfidenceList)
            for key, value in relevantConfidenceList:
                output_images_list.append(key)
            for key, value in irrelavantConfidenceList:
                output_images_list.append(key)
            # print(output_images_list)
            plotTheResultInChrome(relavantImages, irrelavantImages, output_images_list, iteration, "Decision Tree")
        elif feedbackSystem == 3:
            print("PPR Based Feedback system")
            if not calculated:
                for i in range(len(imagesNames)):
                    latent = data_pp.iloc[i][:]
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

            seed2 = pd.Series(0, index=imagesNames)
            length2 = len(irrelavantImages)
            for img in irrelavantImages:
                seed2.loc[img] = 1 / length2
            df = pd.DataFrame(adjacency_matrix, columns=imagesNames)
            df.rename(index=rowDict, inplace=True)

            df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix_task6_c.csv"))

            I = np.identity(df.shape[1])
            page_rank = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed)
            page_rank2 = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed2)

            steady_state = pd.Series(page_rank, index=df.index)
            steady_state2 = pd.Series(page_rank2, index=df.index)
            steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix_6_c_" + str(iteration) + ".csv"))
            finalResult = {}
            for i in range(len(imagesNames)):
                    finalResult[imagesNames[i]] = steady_state[imagesNames[i]] - steady_state2[imagesNames[i]]

            # finalResult = list(finalResult.keys())
            sortList = sorted(finalResult.items(), key=lambda x: x[1], reverse=True)
            finalResult = list(dict(sortList).keys())
            plotTheResultInChrome(relavantImages, irrelavantImages, finalResult, iteration, "PPR")

        elif feedbackSystem == 4:
            images_df = pd.read_json(join(config.DATABASE_FOLDER, filename), "r")
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
                product = np.dot(nQuery, reducerObjectpp[i])
                finalResult[imagesNames[i]] = product

            sortList = sorted(finalResult.items(), key=lambda x: x[1], reverse=True)
            finalResult = list(dict(sortList).keys())
            plotTheResultInChrome(relavantImages, irrelavantImages, finalResult, iteration, "Probabilistic")
        else:
            print("Wrong input")
            exit()

        ch = input("Are you satisfied with the output? type Y for exit N for running again ")


def plotTheResultInChrome(relavantImages, irrelavantImages, finalResult, iteration, typeOfAlgo):
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

    f = open(join(config.DATABASE_FOLDER, "task_6_c_" + typeOfAlgo + "_" + str(iteration) + ".html"), "w")
    f.write(s)
    f.close()

    import webbrowser

    url = join(config.DATABASE_FOLDER, "task_6_c_" + typeOfAlgo + "_" + str(iteration) + ".html")
    # MacOS
    # chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    # Windows
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    # Linux
    # chrome_path = '/usr/bin/google-chrome %s'
    webbrowser.get(chrome_path).open(url)


if __name__ == '__main__':
    startTask6()
