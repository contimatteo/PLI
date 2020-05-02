# /usr/bin/env python3

import os
import joblib
import numpy
import json
from sklearn import preprocessing
from configurations import ConfigurationManager

FILE_NAMES: dict = ConfigurationManager.getFileNames()


class _Network:

    def __init__(self):
        self.type = 'MISSING'
        self.training = {}
        self.testing = {}
        self.datasets = {}

    def initialize(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def getEncoderLabelsFileUri(self):
        labelEncoderFileName: str = self.type + '-encoded-labels.json'
        return os.path.join(self.datasets['training']['uri'], *['../', labelEncoderFileName])

    def getTrainedModelFileUri(self):
        modelExportFileName: str = self.type.lower() + '.joblib'
        return os.path.join(self.datasets['training']['uri'], *['../', modelExportFileName])

    def importEncoderLabels(self):
        file = open(self.getEncoderLabelsFileUri(), 'r')
        fileContent = file.read()
        file.close()
        return json.loads(fileContent)

    def exportEncoderLabels(self, labels):
        if not os.path.exists(self.getEncoderLabelsFileUri()):
            file = open(self.getEncoderLabelsFileUri(), 'w+')
            file.write(json.dumps(labels))
            file.close()

    def importTrainedModel(self):
        return joblib.load(self.getTrainedModelFileUri())

    def exportTrainedModel(self, model):
        joblib.dump(model, self.getTrainedModelFileUri())

    def prepareTraining(self):
        languagesFeaturesFileContents: dict = {}

        # creating encoders
        X_Encoder: dict = {}
        Y_Encoder = preprocessing.LabelEncoder()

        # languages
        counter = 0
        for languageFolder in [f for f in os.scandir(self.datasets['training']['uri']) if f.is_dir()]:
            language = languageFolder.name
            # read features file
            featuresFileUri = os.path.join(languageFolder.path, FILE_NAMES['FEATURES'])
            featuresFile = open(featuresFileUri, 'r')
            featureFileContent = json.loads(featuresFile.read())
            featuresFile.close()
            for word, f in featureFileContent['words_frequencies'].items():
                if not word in X_Encoder:
                    counter += 1
                    X_Encoder[word] = counter
            # re-use file content after this function ...
            languagesFeaturesFileContents[language] = featureFileContent['words_frequencies']

        # import/create labels encoder
        Y_Encoder.fit(ConfigurationManager.getLanguages())
        if not os.path.exists(self.getEncoderLabelsFileUri()):
            # export encoder labels file
            self.exportEncoderLabels(X_Encoder)
        else:
            # import encoder labels file
            X_Encoder = self.importEncoderLabels()

        # prepare training data
        X = []
        y = []
        for language in languagesFeaturesFileContents:
            for word, frequency in languagesFeaturesFileContents[language].items():
                # x
                X.append([X_Encoder[word], frequency])
                # y
                languageEncoded = Y_Encoder.transform([language]).tolist()
                y.append(languageEncoded[0])

        # save data
        self.training['X'] = numpy.array(X)
        self.training['y'] = numpy.array(y)

        return self

    def prepareTesting(self):
        # import/create labels encoder
        X_Encoder: dict = self.importEncoderLabels()

        # prepare testing data
        X = []
        y = []
        counter = max(X_Encoder.values())
        for languageFolder in [f for f in os.scandir(self.datasets['training']['uri']) if f.is_dir()]:
            language = languageFolder.name
            # list all examples in {languageFolder.name} folder
            for exampleFolder in [f for f in os.scandir(languageFolder.path) if f.is_dir()]:
                # list all examples versions in {exampleFolder.name} folder
                for exampleVersionFile in [f for f in os.scandir(exampleFolder.path) if f.is_file()]:
                    dictionaryFileName = FILE_NAMES['3N_DICTIONARY']
                    dictionaryFileUri = os.path.join(exampleFolder.path, dictionaryFileName)
                    # read file
                    dictionaryFile = open(dictionaryFileUri, 'r')
                    dictionaryContent: dict = json.loads(str(dictionaryFile.read()))
                    dictionaryFile.close()
                    # get tokens
                    for word in dictionaryContent['words']:
                        if word not in X_Encoder:
                            counter += 1
                            X_Encoder[word] = counter
                        # x
                        X.append([X_Encoder[word], 1])
                        # y
                        y.append(language)

        # save data
        self.testing['X'] = numpy.array(X)
        self.testing['y'] = numpy.array(y)

        return self
