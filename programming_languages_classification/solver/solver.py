# /usr/bin/env python3

from dataset_loader import DatasetLoader
from .networks.svn import SvnNetwork
from .networks.ccn import CcnNetwork
from .networks.bayes import BayesNetwork


class ProblemSolver:

    def __init__(self):
        self.Networks = {}
        self.DatasetInstance = DatasetLoader()

        config = {}
        config['training'] = {}
        config['training']['uri'] = self.DatasetInstance.TRAINING_ABS_URI
        config['testing'] = {}
        config['testing']['uri'] = self.DatasetInstance.TESTING_ABS_URI
        self.datasetConfig = config

        self._initialize()


    def _initialize(self):
        config = self.datasetConfig
        self.Networks = {}
        self.Networks['SVN'] = SvnNetwork(config['training'], config['testing'])
        self.Networks['CCN'] = CcnNetwork(config['training'], config['testing'])
        self.Networks['BAYES'] = BayesNetwork(config['training'], config['testing'])


    def loadDataset(self):
        self.DatasetInstance.load()        


    def train(self, networkType):
        if networkType == 'SVN':
            return self.Networks['SVN'].train()
        elif networkType == 'CCN':
            return self.Networks['CCN'].train()
        else:
            return self.Networks['BAYES'].train()
