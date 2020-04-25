# /usr/bin/env python3

from dataset_loader import DatasetLoader
from tokenizer import Tokenizer
from .networks.ccn import CcnNetwork


class ProblemSolver:

    def __init__(self):
        # tokenizer
        self.Tokenizer = Tokenizer()

        # dataset
        self.datasetConfig = {}
        self.datasetConfig['training'] = {}
        self.datasetConfig['testing'] = {}
        self.DatasetInstance = DatasetLoader()
        
        # networks
        self.Networks = {}
        self.Networks['CCN'] = CcnNetwork()


    def initialize(self):
        self.datasetConfig['training']['uri'] = self.DatasetInstance.TRAINING_ABS_URI
        self.datasetConfig['testing']['uri'] = self.DatasetInstance.TESTING_ABS_URI
        self.datasetConfig['traintestingitestingngUri'] = self.DatasetInstance.TRAINING_ABS_URI

        trainingConfig = self.datasetConfig['training']
        testingConfig = self.datasetConfig['testing']

        self.Networks['CCN'].initialize(trainingConfig, testingConfig)


    def loadDataset(self):
        self.DatasetInstance.load()    


    def train(self, networkType):
        self.Tokenizer.initialize('CCN', self.datasetConfig['training']['uri'])
        print('\n > [training] ==> Creating parsed files ...')
        self.Tokenizer.parse()
        print(' > [training] ==> Creating dictionaries ...')
        self.Tokenizer.tokenize()

        # if networkType == 'SVN':
        #     return self.Networks['SVN'].train()
        # elif networkType == 'CCN':
        #     return self.Networks['CCN'].train()
        # else:
        #     return self.Networks['BAYES'].train()
        print(' > [training] ==> CCN Training execution ...')
        self.Networks['CCN'].train()
