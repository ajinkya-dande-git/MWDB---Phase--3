import json
import os
from os.path import join
import numpy as np
import math
from Main.HOG_feature_descriptor import HOG
import pandas as pd
import Main.config as config
from Main.PCA_Reducer import PCA_Reducer
import matplotlib.pyplot as plt
from Main.helper import find_distance_2_vectors


def startTask6_d(imagePath):
    imagePath = "H:\Asu\mwdb\project-phase-2\MWDB---Phase--3\Metadata\handsHogInfo.json"
    images_df = pd.read_json(imagePath)
    # rImg = int(input("Enter the no of relevant images "))
    # rList = []
    # irList = []
    # for i in range(rImg):
    #     rList.append(input("Enter the "+str(i)+" relevant image id "))
    #
    # irImg = int(input("Enter the no of irrelevant images"))
    #
    # for i in range(irImg):
    #     irList.append(input("Enter the "+str(i)+" irrelevant image id "))

    rImg = 5
    rList = ["Hand_0008110.jpg", "Hand_0008111.jpg", "Hand_0008128.jpg", "Hand_0008129.jpg", "Hand_0008130.jpg"]
    irImg = 5
    irList = ["Hand_0008662.jpg", "Hand_0008663.jpg", "Hand_0008664.jpg", "Hand_0009001.jpg", "Hand_0009002.jpg"]

    threshold = 0.02
    nQuery = []
    print(images_df.shape)
    for q in range(images_df.shape[0]):
        nq = 0
        rq = 0
        irq = 0
        for column in images_df:
            if images_df[column][q] >= threshold:
                nq += 1
                if column in rList:
                    rq += 1
                if column in irList:
                    irq += 1
        print(str(rq) + " " + str(irq))
        pq = (rq + nq / images_df.shape[1]) / (rImg + 1)
        uq = (irq + nq / images_df.shape[1]) / (irImg + 1)
        if pq * (1 - uq) / (uq * (1 - pq) + 1) <= 0:
            nQuery.append(0)
        else:
            q = math.log((pq * (1 - uq)) / (uq * (1 - pq)), 10)
            if q < 0: nQuery.append(0)
            elif q > 1: nQuery.append(1)
            else: nQuery.append(q)

    print(nQuery)


if __name__ == '__main__':
    startTask6_d("")
