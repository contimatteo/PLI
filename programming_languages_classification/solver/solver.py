# /usr/bin/env python3

from dataset_loader import DatasetLoader
from .networks.svn import SvnNetwork
from .networks.ccn import CcnNetwork
from .networks.bayes import BayesNetwork


class ProblemSolver:

    def __init__(self):
        self.DatasetInstance = DatasetLoader()
        self.Networks = {}

        self.Networks['SVN'] = SvnNetwork()
        self.Networks['CCN'] = CcnNetwork()
        self.Networks['BAYES'] = BayesNetwork()

    def loadDataset(self):
        config = {}
        self.DatasetInstance.load()

        config['training'] = {}
        config['training']['uri'] = self.DatasetInstance.TRAINING_ABS_URI
        config['testing'] = {}
        config['testing']['uri'] = self.DatasetInstance.TESTING_ABS_URI

        self.datasetConfig = config

    def train(self, networkType):
        dataset = self.datasetConfig['training']

        if networkType == 'SVN':
            return self.Networks['SVN'].train(dataset)
        elif networkType == 'CCN':
            return self.Networks['CCN'].train(dataset)
        else:
            return self.Networks['BAYES'].train(dataset)
