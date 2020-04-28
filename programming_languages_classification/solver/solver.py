# /usr/bin/env python3

from dataset_loader import DatasetLoader
from features_extractor import FeaturesExtractor
from .networks.cnn import CnnNetwork
from .networks.svm import SvmNetwork


class ProblemSolver:

    def __init__(self):
        # tokenizer
        self.FeaturesExtractor = FeaturesExtractor()

        # dataset
        self.datasetConfig = {}
        self.datasetConfig['training'] = {}
        self.datasetConfig['testing'] = {}
        self.DatasetInstance = DatasetLoader()
        
        # networks
        self.Networks = {}
        self.Networks['CNN'] = CnnNetwork()
        self.Networks['SVM'] = SvmNetwork()


    def initialize(self):
        self.datasetConfig['training']['uri'] = self.DatasetInstance.TRAINING_ABS_URI
        self.datasetConfig['testing']['uri'] = self.DatasetInstance.TESTING_ABS_URI
        self.datasetConfig['traintestingitestingngUri'] = self.DatasetInstance.TRAINING_ABS_URI

        trainingConfig = self.datasetConfig['training']
        testingConfig = self.datasetConfig['testing']

        self.Networks['CNN'].initialize(trainingConfig, testingConfig)
        self.Networks['SVM'].initialize(trainingConfig, testingConfig)


    def loadDataset(self):
        self.DatasetInstance.load()    


    def train(self, networkType):
        # feature extractor initialization
        self.FeaturesExtractor.initialize('CNN', self.datasetConfig['training']['uri'])
        # features pre-processing
        print('\n > [training] ==> start pre-processing ...')
        self.FeaturesExtractor.process()
        # features extraction
        print(' > [training] ==> start features extraction ...')
        self.FeaturesExtractor.extract()

        if networkType == 'SVM':
            print(' > [training] ==> SVM training execution ...')
            return self.Networks['SVM'].train()
        else:
            print(' > [training] ==> CNN training execution ...')
            return self.Networks['CNN'].train()
