import json
import os
from os.path import join
import numpy as np
import pandas as pd
import Main.config as config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Main.HOG_feature_descriptor import HOG
from Main.helper import progress, plotInChromeForTask4
from sklearn.metrics import accuracy_score


def startTask1():
    print("starting task1")
    print("Enter the folder path containing the labeled images")
    training_folder = input()  # '/Users/vedavyas/Desktop/CSE515/phase3_data/Labelled/Set2'#input()

    print("Enter the K")
    K = int(input())

    print("Enter the folder path containing the test images")
    test_folder = input()  # '/Users/vedavyas/Desktop/CSE515/phase3_data/Unlabelled/Set2'#input()

    train_features, training_file_names = load_features(training_folder)
    test_features, testing_file_names = load_features(test_folder)

    metadata = pd.read_csv(config.METADATA_FOLDER)
    train_labels = get_labels(training_folder, metadata)
    test_labels = get_labels(test_folder, metadata)

    train_labels_df = pd.DataFrame(train_labels)
    train_labels_df.columns = ['label']
    train_df = pd.concat([train_labels_df, pd.DataFrame(train_features)], axis=1)

    dorsal_train_df = train_df[train_df['label'] == 'dorsal']
    palmar_train_df = train_df[train_df['label'] == 'palmar']

    dorsal_scaler = StandardScaler()
    dorsal_train_features_scaled = dorsal_scaler.fit_transform(dorsal_train_df.iloc[:, 1:])

    palmar_scaler = StandardScaler()
    palmar_train_features_scaled = palmar_scaler.fit_transform(palmar_train_df.iloc[:, 1:])

    pca_dorsal = PCA(K)
    pca_dorsal.fit(dorsal_train_features_scaled)

    pca_palmar = PCA(K)
    pca_palmar.fit(palmar_train_features_scaled)

    predicted_labels = []
    dorsal_score = []
    palmar_score = []
    for test_feature in test_features:

        dorsal_scaled_test_feature = dorsal_scaler.transform(np.array(test_feature).reshape(1, -1))
        palmar_scaled_test_feature = palmar_scaler.transform(np.array(test_feature).reshape(1, -1))

        dorsal_recreated = pca_dorsal.inverse_transform(pca_dorsal.transform(dorsal_scaled_test_feature))
        palmar_recreated = pca_palmar.inverse_transform(pca_palmar.transform(palmar_scaled_test_feature))

        dorsal_error = get_distance(dorsal_scaled_test_feature, dorsal_recreated)
        palmar_error = get_distance(palmar_scaled_test_feature, palmar_recreated)

        dorsal_score.append(dorsal_error)
        palmar_score.append(palmar_error)

        if palmar_error < dorsal_error:
            predicted_labels.append('palmar')
        else:
            predicted_labels.append('dorsal')

    print()
    accuracy = 0
    if test_labels is not None:
        accuracy = accuracy_score(test_labels, predicted_labels)
    else:
        print("Cannot find accuracy")
    print("Accuracy is: ", accuracy)
    test_labels_map = {}
    for i in range(len(testing_file_names)):
        test_labels_map[testing_file_names[i]] = predicted_labels[i]

    plotInChromeForTask4(test_labels_map, "Task_1", accuracy)
    out_df = pd.concat([pd.DataFrame(testing_file_names), pd.DataFrame(test_labels), pd.DataFrame(predicted_labels)],
                       axis=1)

    out_df.columns = ['Filename', 'Actual', 'Precicted']
    out_df.to_csv(join(config.DATABASE_FOLDER,'Task1_Result.csv'), index=False)


def load_features(folder_path):
    hog_feature_map = {}
    counter = 1
    training_files = os.listdir(folder_path)
    print("Extracting features for the training images!")
    for trainingFile in training_files:
        trainingFileJson = os.fsdecode(trainingFile).split('.')[0] + '.' + os.fsdecode(trainingFile).split('.')[
            1] + '.json'
        fileExists = os.path.exists(join(config.FEATURES_FOLDER, trainingFileJson))
        data = {}
        if fileExists:
            with open(join(config.FEATURES_FOLDER, trainingFileJson), "r") as f:
                data = json.load(f)
                hog_feature_map.update(data)
        else:
            data = HOG().HOGForSingleImage(folder_path,trainingFile)
            hog_feature_map.update(data)

        progress(counter, len(training_files))
        counter = counter + 1
    hog_values = list(hog_feature_map.values())
    return hog_values, training_files


def get_labels(image_folder, metadata):
    image_labels = []
    for file in os.listdir(image_folder):
        file_name = os.fsdecode(file)
        label = metadata.loc[metadata['imageName'] == file_name]['aspectOfHand'].iloc[0]
        if "dorsal" in label:
            image_labels.append("dorsal")
        elif "palmar" in label:
            image_labels.append("palmar")
    return image_labels


def get_distance(image1, image2):
    return np.linalg.norm(image1 - image2)


if __name__ == '__main__':
    startTask1()