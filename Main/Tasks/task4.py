# Took reference from https://towardsdatascience.com/support-vector-machine-python-example-d67d9b63f1c8
import json
import os
from os.path import join
import numpy as np
import pandas as pd
import Main.config as config
from Main.HOG_feature_descriptor import HOG
from Main.PCA_Reducer import PCA_Reducer
from Main.helper import progress, plotInChromeForTask4, find_distance_2_vectors
from sklearn.metrics import accuracy_score
import cvxopt as cvx
import matplotlib.pyplot as plt


def startTask4():
    print("starting task4")
    print("Enter the folder path containing the labeled images")
    training_folder = input()

    print("Choose one of the below classifier")
    print("1. SVM classifer\n2. Decision-Tree classifier\n3. PPR based classifier")
    classifier = int(input())

    print("Enter the folder path containing the test images")
    test_folder = input()

    hog_feature_map = {}
    counter = 1
    training_files = os.listdir(training_folder)
    print("Extracting features for the training images!")
    for trainingFile in training_files:
        trainingFileJson = os.fsdecode(trainingFile).split('.')[0] + '.' + os.fsdecode(trainingFile).split('.')[
            1] + '.json'
        fileExists = os.path.exists(join(config.FEATURES_FOLDER, trainingFileJson))
        if fileExists:
            with open(join(config.FEATURES_FOLDER, trainingFileJson), "r") as f:
                data = json.load(f)
                hog_feature_map.update(data)
        else:
            data = HOG().HOGForSingleImage(training_folder, trainingFile)
            hog_feature_map.update(data)
        progress(counter, len(training_files))
        counter = counter + 1
    reducer_object = list(hog_feature_map.values())
    print("Performing PCA!")
    pca = PCA_Reducer(reducer_object)
    data = pca.reduceDimension(pca.featureDescriptor)
    print("Done performing PCA!")
    if classifier == 1:
        # image labels are added to the imageLabels list. -1 for dorsal and 1 for palmar
        metadata = pd.read_csv(config.METADATA_FOLDER)
        image_lables = get_labels(training_folder, metadata)
        svm_object = SVM()
        print("Training SVM")
        svm_object.svm_fit(data, image_lables)
        print("Done Training SVM")
        test_labels_map = {}
        predicted_values = []
        actual_values = get_labels(test_folder, metadata)
        for file in os.listdir(test_folder):
            test_file = file
            test_file_json = file + '.json'
            file_exists = os.path.exists(join(config.FEATURES_FOLDER, test_file_json))
            if file_exists:
                with open(join(config.FEATURES_FOLDER, test_file_json), "r") as f:
                    data = json.load(f)
            else:
                data = HOG().HOGForSingleImage(test_folder, test_file)
            pca_output = pca.reduceDimension(list(data.values()))
            output_label = np.asarray(svm_object.predict(pca_output))[0]
            predicted_values.append(output_label)
            if output_label == -1:
                test_labels_map[test_file] = "dorsal"
            else:
                test_labels_map[test_file] = "palmar"
        print(test_labels_map)
        accuracy = 0
        if actual_values is not None:
            accuracy = accuracy_score(actual_values, predicted_values)
        else:
            print("Cannot find accuracy")
        plotInChromeForTask4(test_labels_map, "Task_4_SVM", accuracy)
        print("Test Accuracy: ", accuracy)

    if classifier == 2:
        data = data.values.tolist()  # decision tree takes data as 2d array
        class_labels = [-1, 1]
        i = 0
        metadata = pd.read_csv(config.METADATA_FOLDER)
        for file in os.listdir(training_folder):
            training_file = os.fsdecode(file)
            label = metadata.loc[metadata['imageName'] == training_file]['aspectOfHand'].iloc[0]
            if "dorsal" in label:
                data[i].append(-1)
            else:
                data[i].append(1)
            i = i + 1
        dtree_object = decisionTree()
        root = dtree_object.construct_dt(data, class_labels, 5, 2)
        test_labels_map = {}
        predicted_values = []
        actual_values = get_labels(test_folder, metadata)
        for file in os.listdir(test_folder):
            test_file = os.fsdecode(file).split('.')[0] + '.' + os.fsdecode(file).split('.')[1]
            test_file_json = os.fsdecode(file).split('.')[0] + '.' + os.fsdecode(file).split('.')[1] + '.json'
            file_exists = os.path.exists(join(config.FEATURES_FOLDER, test_file_json))
            if file_exists:
                with open(join(config.FEATURES_FOLDER, test_file_json), "r") as f:
                    data = json.load(f)
            else:
                data = HOG().HOGForSingleImage(test_folder, test_file)
            pca_output = pca.reduceDimension(list(data.values()))
            pca_output = pca_output.values.tolist()[0]
            output_label = dtree_object.predict(root, pca_output)
            predicted_values.append(output_label)
            if output_label == -1:
                test_labels_map[test_file] = "dorsal"
            else:
                test_labels_map[test_file] = "palmar"
        accuracy = 0
        if actual_values is not None:
            accuracy = accuracy_score(actual_values, predicted_values)
        else:
            print("Cannot find accuracy")
        plotInChromeForTask4(test_labels_map, "Task_4_DECISION", accuracy)
        print("Test Accuracy: ", accuracy)

    if classifier == 3:
        pca_for_all = data

        i = 0
        imageNames = []
        latentFeatureDict = {}

        # Preprocessing for UnLabelled set
        ppr_hog_map = {}
        for test_file in os.listdir(test_folder):
            trainingFileJson = str(test_file) + '.json'
            fileExists = os.path.exists(join(config.FEATURES_FOLDER, trainingFileJson))
            if fileExists:
                with open(join(config.FEATURES_FOLDER, trainingFileJson), "r") as f:
                    data = json.load(f)
                    ppr_hog_map.update(data)
            else:
                data = HOG().HOGForSingleImage(test_folder, test_file)
                ppr_hog_map.update(data)
        # Appending the labelled data values with unlabelled images data
        reducer_object = list(hog_feature_map.values())
        pp_reducer_object = list(ppr_hog_map.values())
        pp_reducer_object = reducer_object + pp_reducer_object
        pca = PCA_Reducer(pp_reducer_object)
        unlabelled_ppr_data = pca.reduceDimension(pca.featureDescriptor)
        pca_for_all = unlabelled_ppr_data

        for file in os.listdir(str(training_folder)):
            filename = os.fsdecode(file)
            latent = pca_for_all.iloc[i][:]
            imageNames.append(filename)
            latentFeatureDict[filename] = latent
            i = i + 1

        for file in os.listdir(join(test_folder)):
            filename = os.fsdecode(file)
            latent = pca_for_all.iloc[i][:]
            imageNames.append(filename)
            latentFeatureDict[filename] = latent
            i = i + 1

        # seed = pd.Series(0, index=imageNames)
        print("Generating Adjacency Matrix..")
        adjacency_matrix = [[0 for _ in range(len(latentFeatureDict))] for _ in range(len(latentFeatureDict))]
        for i in range(len(latentFeatureDict)):
            distances = []
            for j in range(len(latentFeatureDict)):
                # print(len(latentFeatureDict[imageNames[i]]), len(latentFeatureDict[imageNames[j]]))
                distances.append(find_distance_2_vectors(latentFeatureDict[imageNames[i]],
                                                         latentFeatureDict[imageNames[j]]))

            distances = np.asarray(distances)
            ind = np.argpartition(distances, 20)[:20]
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

        df.to_csv(join(config.DATABASE_FOLDER, "adjacency_matrix_for_task_4.csv"))

        I = np.identity(df.shape[1])
        seed = pd.Series(0, index=imageNames)
        metadata = pd.read_csv(config.METADATA_FOLDER)
        image_lables = get_labels(training_folder, metadata)
        count = image_lables.count(-1)
        val = 1 / count
        for i in range(len(os.listdir(training_folder))):
            if image_lables[i] == -1:
                seed.loc[imageNames[i]] = val
        # print(seed)
        seed2 = pd.Series(0, index=imageNames)
        count2 = image_lables.count(1)
        val2 = 1 / count2
        for i in range(len(os.listdir(training_folder))):
            if image_lables[i] == 1:
                seed2.loc[imageNames[i]] = val

        page_rank = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed)
        page_rank2 = np.matmul(np.linalg.inv(I - .75 * df), 0.25 * seed2)
        steady_state = pd.Series(page_rank, index=df.index)
        steady_state2 = pd.Series(page_rank2, index=df.index)
        test_labels_map = {}
        predicted_values = []
        for file in os.listdir(join(test_folder)):
            if steady_state[file] >= steady_state2[file]:
                test_labels_map[file] = "dorsal"
                predicted_values.append(-1)
            else:
                test_labels_map[file] = "palmar"
                predicted_values.append(1)

        actual_values = get_labels(test_folder, metadata)
        accuracy = 0
        if actual_values is not None:
            accuracy = accuracy_score(actual_values, predicted_values)
        else:
            print("Cannot find accuracy")
        plotInChromeForTask4(test_labels_map, "Task_4_PPR", accuracy)
        print("Test Accuracy: ", accuracy)
        steady_state = steady_state.sort_values(ascending=True)
        steady_state.to_csv(join(config.DATABASE_FOLDER, "steady_state_matrix_for_task_4.csv"))
        steady_state.plot()
        plt.show()


def get_labels(image_folder, metadata):
    image_lables = []
    for file in os.listdir(image_folder):
        file_name = os.fsdecode(file)
        try:
            label = metadata.loc[metadata['imageName'] == file_name]['aspectOfHand'].iloc[0]
            if "dorsal" in label:
                image_lables.append(-1)
            elif "palmar" in label:
                image_lables.append(1)
        except:
            return None
    return image_lables


class SVM:
    def __init__(self):
        self.lg_multipers = None
        self.sv = None
        self.sv_label = None
        self.intercept = None
        self.weights = None

    def svm_fit(self, x, y):
        n, k = x.shape
        kernel = np.zeros((n, n))
        self.weights = np.zeros(k)
        self.intercept = 0
        for i in range(n):
            for j in range(n):
                kernel[i, j] = np.dot(x.iloc[i], x.iloc[j])

        P = cvx.matrix(np.outer(y, y) * kernel, tc='d')
        q = cvx.matrix(np.ones(n) * -1)
        A = cvx.matrix(y, (1, n), tc='d')
        b = cvx.matrix(0, tc='d')
        G = cvx.matrix(np.diag(np.ones(n) * -1))
        h = cvx.matrix(np.zeros(n))

        lagrangian_solver = cvx.solvers.qp(P, q, G, h, A, b)

        lg_multipers = np.ravel(lagrangian_solver['x'])
        index = lg_multipers > 1e-5

        indexes = np.arange(len(lg_multipers))[index]
        self.lg_multipers = lg_multipers[index]
        self.sv = x[index]
        self.sv_label = np.asarray(y)[index]

        self.intercept = 0
        for i in range(len(self.lg_multipers)):
            self.intercept = self.intercept + self.sv_label[i]
            self.intercept = self.intercept - np.sum(self.lg_multipers * self.sv_label * kernel[indexes[i], index])
        self.intercept = self.intercept / len(self.lg_multipers)

        for i in range(len(self.lg_multipers)):
            self.weights = self.weights + np.sum(self.lg_multipers[i] * self.sv_label[i] * self.sv[i])

    def predict(self, x):
        equation = np.dot(x, self.weights) + self.intercept
        return np.sign(equation)

    def distance(self, x):
        equation = np.dot(x, self.weights) + self.intercept
        print("x", x)
        print("equation:", abs(equation))
        print(np.dot(x, x))
        return abs(equation) / np.dot(x, x)


class decisionTree():
    def temp_partition(self, idx, val, data):
        ltree = list()
        rtree = list()
        for image_vector in data:
            if image_vector[idx] < val:
                ltree.append(image_vector)
            else:
                rtree.append(image_vector)
        return ltree, rtree

    def compute_score(self, partitions, classLabels):
        samples = float(sum([len(partition) for partition in partitions]))
        partition_score = 0.0
        for partition in partitions:
            partition_size = float(len(partition))
            if partition_size == 0:
                continue
            score = 0.0
            for label in classLabels:
                val = [sample[-1] for sample in partition].count(label) / partition_size
                score = score + val * val
            partition_score = partition_score + (1.0 - score) * (partition_size / samples)
        return partition_score

    def find_best_partition(self, data, classLabels):
        partition_index = 1000
        partition_value = 1000
        partition_score = 1000
        final_partitions = None
        for idx in range(len(data[0]) - 1):
            for row in data:
                partitions = self.temp_partition(idx, row[idx], data)
                score = self.compute_score(partitions, classLabels)
                if score < partition_score:
                    partition_index = idx
                    partition_value = row[idx]
                    partition_score = score
                    final_partitions = partitions
        return {'index': partition_index, 'value': partition_value, 'partitions': final_partitions}

    def leaf_node(self, partition):
        output = [sample[-1] for sample in partition]
        return max(set(output), key=output.count)

    def label_count(self, partition):
        output = [sample[-1] for sample in partition]
        relevant = output.count(-1)
        irrelevant = output.count(1)
        return [relevant, irrelevant]

    def perform_partition(self, rootNode, classLabels, max_height, min_samples, height):
        ltree, rtree = rootNode['partitions']
        del (rootNode['partitions'])
        if not ltree or not rtree:
            val = self.leaf_node(ltree + rtree)
            rootNode['ltree'] = val
            rootNode['rtree'] = val
            rootNode['lcount'] = self.label_count(ltree + rtree)
            rootNode['rcount'] = self.label_count(ltree + rtree)
            return
        if height >= max_height:
            rootNode['ltree'] = self.leaf_node(ltree)
            rootNode['rtree'] = self.leaf_node(rtree)
            rootNode['lcount'] = self.label_count(ltree)
            rootNode['rcount'] = self.label_count(rtree)
            return
        if len(ltree) <= min_samples:
            rootNode['ltree'] = self.leaf_node(ltree)
            rootNode['lcount'] = self.label_count(ltree)
        else:
            rootNode['ltree'] = self.find_best_partition(ltree, classLabels)
            self.perform_partition(rootNode['ltree'], classLabels, max_height, min_samples, height + 1)
        if len(rtree) <= min_samples:
            rootNode['rtree'] = self.leaf_node(rtree)
            rootNode['rcount'] = self.label_count(rtree)
        else:
            rootNode['rtree'] = self.find_best_partition(rtree, classLabels)
            self.perform_partition(rootNode['rtree'], classLabels, max_height, min_samples, height + 1)

    def construct_dt(self, data, classLabels, max_height, min_samples):
        rootNode = self.find_best_partition(data, classLabels)
        self.perform_partition(rootNode, classLabels, max_height, min_samples, 1)
        return rootNode

    def predict(self, root, sample):
        if sample[root['index']] < root['value']:
            if isinstance(root['ltree'], dict):
                return self.predict(root['ltree'], sample)
            else:
                return root['ltree']
        else:
            if isinstance(root['rtree'], dict):
                return self.predict(root['rtree'], sample)
            else:
                return root['rtree']

    def confidence(self, root, sample, label):
        if sample[root['index']] < root['value']:
            if isinstance(root['ltree'], dict):
                return self.confidence(root['ltree'], sample, label)
            else:
                if label == -1:
                    return root['lcount'][0] / (root['lcount'][0] + root['lcount'][1])
                else:
                    return root['lcount'][1] / (root['lcount'][0] + root['lcount'][1])
        else:
            if isinstance(root['rtree'], dict):
                return self.confidence(root['rtree'], sample, label)
            else:
                if label == -1:
                    return root['rcount'][0] / (root['rcount'][0] + root['rcount'][1])
                else:
                    return root['rcount'][1] / (root['rcount'][0] + root['rcount'][1])


if __name__ == '__main__':
    startTask4()
