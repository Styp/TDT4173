import os, os.path
import string
import scipy

import numpy as np

import PIL

from skimage import filters
from skimage import io, color, exposure, morphology


class DataCarrierObject(object):
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test


class DataLoader(object):

    def loadData(path) -> DataCarrierObject:
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        img_letter = string.ascii_lowercase[:26]

        for letter in img_letter:
            counter = 0;
            print("Loading letter: " + letter)
            absPath = path + "/" + letter

            trainPath = absPath + "/TrainingSet"
            testPath = absPath + "/TestingSet"

            for f in os.listdir(trainPath):
                if(counter < 2):
                    #counter = counter + 1;
                    fullPath = trainPath + "/" + f

                    rawImg = io.imread(fullPath, as_grey=True)

                    otsuThreshold = filters.threshold_otsu(rawImg)
                    img_bw = rawImg > otsuThreshold
                    intArr = np.array(img_bw).astype(int)
                    sciImg = np.multiply(intArr, 255)

                    lineImgArray = sciImg.reshape((1,400))

                    if(X_train == []):
                        X_train = lineImgArray
                    else:
                        X_train = scipy.vstack((X_train, lineImgArray))
                    Y_train.append(letter)

            for f in os.listdir(testPath):
                fullPath = testPath + "/" + f
                rawImg = scipy.array(PIL.Image.open(fullPath).convert("L"))
                lineImgArray = rawImg.reshape((1,400))
                #            print(rawImg)
                if(X_test == []):
                    X_test = lineImgArray
                else:
                    X_test = scipy.vstack((X_test, lineImgArray))
                Y_test.append(letter)

        return DataCarrierObject(X_train, Y_train, X_test, Y_train)