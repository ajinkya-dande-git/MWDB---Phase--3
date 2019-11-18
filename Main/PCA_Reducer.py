import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from Main.helper import find_distance_2_vectors


class PCA_Reducer:
    def __init__(self, featureDescriptor, k=80):
        self.featureDescriptor = featureDescriptor
        self.k = k
        self.imageID = None
        self.pca = PCA(n_components=self.k)
        self.scaler = StandardScaler()
        self.scaler.fit(self.featureDescriptor)
        self.normalizedFeatureDescriptor = self.scaler.transform(self.featureDescriptor)
        if min(self.normalizedFeatureDescriptor.shape) < k:
            print("Cannot compute on PCA on components higher than min of", self.normalizedFeatureDescriptor.shape)
            exit()
        self.pca.fit(self.normalizedFeatureDescriptor)
        self.featureLatentSemantics = self.pca.components_.T
        self.objectLatentSemantics = self.pca.transform(featureDescriptor)
        self.SIFT_info = None

    def set_SIFT_info(self, obj):
        self.SIFT_info = obj

    def reduceDimension(self, data):
        reducedDimesnions = self.pca.transform(self.scaler.transform(data))
        return pd.DataFrame(data=reducedDimesnions)

    def inv_transform(self, data):
        return self.scaler.inverse_transform(self.pca.inverse_transform(data))

    def saveImageID(self, imageID):
        self.imageID = imageID

    def compute_threshold(self):
        reconstructed_normalized_feat_desc = self.inv_transform(self.objectLatentSemantics)
        reconstructed_feat_desc = self.scaler.inverse_transform(reconstructed_normalized_feat_desc)
        reconstruction_err = find_distance_2_vectors(reconstructed_feat_desc, self.featureDescriptor)
        # print('shape: ', np.shape(reconstruction_err), np.average(reconstruction_err))
        self.threshold = np.percentile(reconstruction_err, 85)
