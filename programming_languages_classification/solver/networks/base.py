# /usr/bin/env python3

import os
import joblib
import numpy


class _Network:

    def __init__(self):
        self.type = 'MISSING'
        self.training = {}
        self.testing = {}

    def initialize(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets = {}
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def getEncoderLabelsFileUri(self):
        labelEncoderFileName: str = self.type + '-encoder-labels-exported.npy'
        return os.path.join(self.datasets['training']['uri'], labelEncoderFileName)

    def getTrainedModelFileUri(self):
        modelExportFileName: str = self.type.lower() + '.joblib'
        return os.path.join(self.datasets['training']['uri'], modelExportFileName)

    def exportEncoderLabels(self, labels):
        labelEncoderFileUri = self.getEncoderLabelsFileUri()
        # export
        if not os.path.exists(labelEncoderFileUri):
            numpy.save(labelEncoderFileUri, labels)

    def exportTrainedModel(self, model):
        joblib.dump(model, self.getTrainedModelFileUri()) 
    
    def importEncoderLabels(self):
        return numpy.load(self.getEncoderLabelsFileUri())

    def importTrainedModel(self):
        return joblib.load(self.getTrainedModelFileUri()) 
