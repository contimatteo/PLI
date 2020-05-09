# /usr/bin/env python3

import os
import joblib
import numpy
import json
from sklearn import preprocessing
from utils import FileManager, ConfigurationManager
from dataset import DatasetInstance


class _ScikitLearnAlgorithm:
    type: str = 'MISSING'
    dataset: DatasetInstance = None

    def initialize(self, datasetInstance: DatasetInstance):
        self.dataset = datasetInstance
        return self

    def importEncoderLabels(self):
        return json.loads(FileManager.readFile(FileManager.getEncoderLabelsFileUrl(self.type)))

    def exportEncoderLabels(self, labels):
        path = FileManager.getEncoderLabelsFileUrl(self.type)
        if not os.path.exists(path):
            FileManager.writeFile(path, json.dumps(labels))

    def importTrainedModel(self):
        return joblib.load(FileManager.getTrainedModelFileUrl(self.type))

    def exportTrainedModel(self, model):
        joblib.dump(model, FileManager.getTrainedModelFileUrl(self.type))

    # def prepareTraining(self):
    #     languagesFeaturesFileContents: dict = {}
    #
    #     # creating encoders
    #     X_Encoder: dict = {}
    #     Y_Encoder = preprocessing.LabelEncoder()
    #
    #     # languages
    #     counter = 0
    #     for languageFolder in FileManager.getLanguagesFolders(self.datasets['training']['url']):
    #         # read features file
    #         featureFileContent = FileManager.readFile(FileManager.getFeaturesMapFileUrl(languageFolder.path))
    #         featureFileContent = json.loads(featureFileContent)
    #         for word, f in featureFileContent['words_frequencies'].items():
    #             if word not in X_Encoder:
    #                 counter += 1
    #                 X_Encoder[word] = counter
    #         # re-use file content after this function ...
    #         languagesFeaturesFileContents[languageFolder.name] = featureFileContent['words_frequencies']
    #
    #     # Y labels encoder
    #     Y_Encoder.fit(ConfigurationManager.getLanguages())
    #     # X labels encoder
    #     if not os.path.exists(FileManager.getEncoderLabelsFileUrl(self.type)):
    #         self.exportEncoderLabels(X_Encoder) # export
    #     else:
    #         X_Encoder = self.importEncoderLabels() # import
    #
    #     # prepare training data
    #     X = []
    #     y = []
    #     for language in languagesFeaturesFileContents:
    #         for word, frequency in languagesFeaturesFileContents[language].items():
    #             # x
    #             X.append([X_Encoder[word], frequency])
    #             # y
    #             languageEncoded = Y_Encoder.transform([language]).tolist()
    #             y.append(languageEncoded[0])
    #
    #     # save data
    #     # self.training['X'] = numpy.array(X)
    #     # self.training['y'] = numpy.array(y)
    #
    #     return numpy.array(X), numpy.array(y)
    #
    # def prepareTesting(self):
    #     # import/create labels encoder
    #     X_Encoder: dict = self.importEncoderLabels()
    #
    #     # prepare testing data
    #     X = []
    #     y = []
    #     counter = max(X_Encoder.values())
    #     for languageFolder in FileManager.getLanguagesFolders(self.datasets['testing']['url']):
    #         language = languageFolder.name
    #         # list all examples in {languageFolder.name} folder
    #         for exampleFolder in FileManager.getExamplesFolders(languageFolder.path):
    #             dictionaryContent = FileManager.readFile(FileManager.getDictionaryFileUrl(exampleFolder.path))
    #             dictionaryContent = json.loads(dictionaryContent)
    #             # get tokens
    #             for word in dictionaryContent['words']:
    #                 if word not in X_Encoder:
    #                     counter += 1
    #                     X_Encoder[word] = counter
    #                 # x
    #                 X.append([X_Encoder[word], 1])
    #                 # y
    #                 y.append(language)
    #
    #     # save data
    #     self.testing['X'] = numpy.array(X)
    #     self.testing['y'] = numpy.array(y)
    #
    #     return self
