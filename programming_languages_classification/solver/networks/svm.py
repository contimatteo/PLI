# /usr/bin/env python3

import os
import json
from .base import _Network
from sklearn import svm
from sklearn import preprocessing
from configurations import ConfigurationManager

FILE_NAMES: dict = ConfigurationManager.getFileNames()

class SvmNetwork(_Network):

    def __init__(self):
        super().__init__()
        self.type = 'SVM'


    def train(self):
        if os.path.exists(self.getTrainedModelFileUri()):
            return self;

        # preparing training data
        self.prepareTraining()

        # get training data
        X = self.training['X'].tolist() # numpy array
        y = self.training['y'].tolist() # numpy array

        # train
        model = svm.SVC()
        model.fit(X, y)

        # export the model
        self.exportTrainedModel(model)


    def test(self):
        if not os.path.exists(self.getTrainedModelFileUri()):
            raise Exception('You can\'t test a model without training it')

        # preparing testing data
        
        self.prepareTesting()

        # get testing data
        X = self.testing['X'].tolist() # numpy array
        y = self.testing['y'].tolist() # numpy array

        # model
        model = self.importTrainedModel()

        # make predictions
        prediction = model.predict(X)
        print('')
        # print('[prediction] ==> ' + str(Y_Encoder.inverse_transform(prediction)))
        print('[prediction] ==> ' + str(prediction))
        print('')