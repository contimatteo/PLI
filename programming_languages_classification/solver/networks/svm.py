# /usr/bin/env python3

import os
import json
from sklearn import svm, datasets
import json
from .base import _Network
from sklearn import svm
import numpy
from sklearn import preprocessing
from configurations import ConfigurationManager


FILE_NAMES: dict = ConfigurationManager.getFileNames()


class SvmNetwork(_Network):

    def __init__(self):
        super().__init__()
        self.type = 'SVM'


    def prepare(self):        
        X = []
        y = []
        tokens = []
        languagesFeaturesFileContents: dict = {};
        X_Encoder = preprocessing.LabelEncoder()
        
        # languages
        for languageFolder in [f for f in os.scandir(self.datasets['training']['uri']) if f.is_dir()]:
            language = str(languageFolder.name).lower()
            # read features file
            featuresFileUri = os.path.join(languageFolder.path, FILE_NAMES['FEATURES'])
            featuresFile = open(featuresFileUri, 'r')
            featureFileContent = json.loads(featuresFile.read())
            featuresFile.close()
            for word, f in featureFileContent['words_frequencies'].items(): 
                tokens.append(word)
            languagesFeaturesFileContents[language] = featureFileContent['words_frequencies']
            
        # import or create labels encoder
        if not os.path.exists(self.getEncoderLabelsFileUri()):
            # labels encoder fitting
            X_Encoder.fit(tokens)
            # export encoder labels file
            self.exportEncoderLabels(X_Encoder.classes_)
        else:
            # import encoder labels file
            X_Encoder.classes_ = self.importEncoderLabels()

        # add features to training data
        for language in languagesFeaturesFileContents: 
            for word, frequency in languagesFeaturesFileContents[language].items(): 
                token = X_Encoder.transform([word])
                X.append([token[0], frequency])
                y.append(language)

        # save data
        self.training['X'] = numpy.array(X)
        self.training['y'] = numpy.array(y)

        return self


    def train(self):
        X = self.training['X'].tolist()
        
        # prepare y training data
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(self.training['y'].tolist())
        y = Y_Encoder.transform(self.training['y'].tolist())

        # train
        model = svm.SVC()
        model.fit(X, y)

        # export the model
        self.exportTrainedModel(model)


    # def test(self):
    #     # prepare X testing data
    #     X_Encoder = preprocessing.LabelEncoder()
    #     X_encoder.classes_ = self.importEncoderLabels()
    #     X = X_Encoder.transform(self.testing['X'])

    #     # prepare y testing data
    #     Y_Encoder = preprocessing.LabelEncoder()
    #     Y_Encoder.fit(self.testing['y'])
    #     y = Y_Encoder.transform(self.testing['y'])

    #     # model
    #     model = self.importTrainedModel()