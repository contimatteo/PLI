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
        self.DatasetInstance.load()

    def train(self, networkType):
        if networkType == 'SVN':
            return self.Networks['SVN'].train()
        elif networkType == 'CCN':
            return self.Networks['CCN'].train()
        else:
            return self.Networks['BAYES'].train()
