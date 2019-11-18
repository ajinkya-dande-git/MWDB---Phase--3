from os.path import join

from Main.HOG_feature_descriptor import HOG
import pickle
import Main.config as config

def startTask3():
    print("start task3")
    k = input("Please enter the k value for outgoing edges ")
    K = input("Please enter the K value for visualizing dominant images ")
    hog = HOG()
    featureVector = hog.HOGFeatureDescriptor()
    fileHandler = open(join(config.DATABASE_FOLDER, 'hog_features_task_3'), 'wb')
    pickle.dump(featureVector, fileHandler)


if __name__ == '__main__':
    startTask3()
