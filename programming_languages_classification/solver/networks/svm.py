# /usr/bin/env python3

import os
from .base import _Network
from sklearn import svm
from sklearn import preprocessing
from configurations import ConfigurationManager


class SvmNetwork(_Network):

    def __init__(self):
        super().__init__()
        self.type = 'SVM'

    def train(self):
        if os.path.exists(self.getTrainedModelFileUri()):
            return self

        # preparing training data
        self.prepareTraining()

        # get training data
        X = self.training['X'].tolist()
        y = self.training['y'].tolist()

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

        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # get testing data
        X = self.testing['X'].tolist() # numpy array
        y = self.testing['y'].tolist() # numpy array

        # model
        model = self.importTrainedModel()

        # make predictions
        predictions = model.predict(X)

        matched = 0
        for index, prediction in enumerate(predictions):
            predictedLanguage = Y_Encoder.inverse_transform([prediction])[0]
            if predictedLanguage == y[index]:
                matched += 1

        print('')
        print(' > [results] % matched ==> ' + str(matched / len(predictions)))
        print('')



