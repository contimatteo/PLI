# /usr/bin/env python3

import json

class BayesNetwork:

    def __init__(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets = {}
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def train(self):
        print('Bayes training' + json.dumps(self.datasets['training']))
