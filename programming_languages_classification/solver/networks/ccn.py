# /usr/bin/env python3

import json

class CcnNetwork:

    def __init__(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets = {}
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def train(self):
        print('CCN training' + json.dumps(self.datasets['training']))
