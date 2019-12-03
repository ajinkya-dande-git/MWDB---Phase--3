import json
import os
from os.path import join

from skimage import io
from skimage.feature import hog
from skimage.transform import rescale

import Main.config as config
from Main.helper import progress


class HOG:
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    bins = 9  # number of orientation bins

    @classmethod
    def HOGForSingleImage(self, filename):
        image = io.imread(filename)
        image_rescaled = rescale(image, 0.1, anti_aliasing=False)
        # image = rescale(file, scale=0.1, anti_aliasing=True)
        fd, hog_image = hog(image_rescaled, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True, transform_sqrt=True)
        return fd

    @classmethod
    def HOGFeatureDescriptor(self):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images
        storeHogFD = {}
        files = os.listdir(str(config.FULL_IMAGESET_FOLDER))  # dir is your directory path
        number_files = len(files)
        i = 0
        for file in os.listdir(str(config.FULL_IMAGESET_FOLDER)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                storeHogFD = {}
                i = i + 1
                hognp = self.HOGForSingleImage(join(str(config.FULL_IMAGESET_FOLDER), filename))
                progress(i, number_files)
                storeHogFD[filename] = (hognp.tolist())
                with open(join(config.FEATURES_FOLDER, filename + ".json"), 'w', encoding='utf-8') as f:
                    json.dump(storeHogFD, f, ensure_ascii=True)
        print()

    @classmethod
    def HOGFeatureDescriptorForImageSubset(self, imageSet):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images in the subset
        storeHogFD = []
        hog = HOG()
        number_files = len(imageSet)
        i = 0
        for filename in imageSet:
            i = i + 1
            hognp = hog.HOGForSingleImage(join(str(config.IMAGE_FOLDER), filename))
            progress(i, number_files)
            storeHogFD.append(hognp.tolist())
        print(len(storeHogFD))
        return storeHogFD

    def HOGFeatureDescriptorForFolder(self, folderName):
        # Iterating on all the images in the selected folder to calculate HOG FD for each of the images
        storeHogFD = {}
        hog = HOG()
        files = os.listdir(str(folderName))  # dir is your directory path
        number_files = len(files)
        i = 0
        for file in os.listdir(str(folderName)):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                i = i + 1
                hognp = hog.HOGForSingleImage(join(str(folderName), filename))
                progress(i, number_files)
                storeHogFD[filename] = (hognp.tolist())
        print()
        return storeHogFD
