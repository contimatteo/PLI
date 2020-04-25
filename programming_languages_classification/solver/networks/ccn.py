# /usr/bin/env python3

import json

class CcnNetwork:

    def __init__(self):
        self.datasets = {}

    def initialize(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets = {}
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def train(self):
        # TODO: missing training logic ...
        return
