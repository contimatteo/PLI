# /usr/bin/env python3

class _Network:

    def __init__(self):
        self.datasets = {}

    def initialize(self, trainingDatasetConfig, testingDatasetConfig):
        self.datasets = {}
        self.datasets['training'] = trainingDatasetConfig
        self.datasets['testing'] = testingDatasetConfig

    def train(self):
        return
