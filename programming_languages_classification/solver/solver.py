# /usr/bin/env python3

from dataset_loader import DatasetLoader
from features_extractor import FeaturesExtractor
from .networks.bayes import BayesNetwork
from .networks.svm import SvmNetwork


class ProblemSolver:

    def __init__(self):
        # tokenizer
        self.FeaturesExtractor = FeaturesExtractor()
        # dataset
        self.datasetConfig = {'training': {}, 'testing': {}}
        self.DatasetInstance = DatasetLoader()
        # networks
        self.Networks = {'BAYES': BayesNetwork(), 'SVM': SvmNetwork()}

    def initialize(self):
        # daraset
        self.datasetConfig['training']['uri'] = self.DatasetInstance.TRAINING_ABS_URI
        self.datasetConfig['testing']['uri'] = self.DatasetInstance.TESTING_ABS_URI
        trainingConfig = self.datasetConfig['training']
        testingConfig = self.datasetConfig['testing']
        # networks
        self.Networks['BAYES'].initialize(trainingConfig, testingConfig)
        self.Networks['SVM'].initialize(trainingConfig, testingConfig)

    def loadDataset(self):
        self.DatasetInstance.load()

    def prepare(self, networkType, datasetUri):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, datasetUri)
        # features pre-processing
        print(' > [training] ==> start pre-processing ...')
        self.FeaturesExtractor.process()
        # features extraction
        print(' > [training] ==> start features extraction ...')
        self.FeaturesExtractor.extract()

    def train(self, networkType):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, self.datasetConfig['training']['uri'])
        # features pre-processing
        print(' > [training] ==> start pre-processing ...')
        self.FeaturesExtractor.process()
        # features extraction
        print(' > [training] ==> start features extraction ...')
        self.FeaturesExtractor.extract(calculateWordsFrequency=True)
        # train
        print(' > [training] ==> ' + networkType + ' training ...')
        return self.Networks[networkType].train()

    def test(self, networkType):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, self.datasetConfig['testing']['uri'])
        # features pre-processing
        print(' > [testing] ==> start pre-processing ...')
        self.FeaturesExtractor.process()
        # features extraction
        print(' > [testing] ==> start features extraction ...')
        self.FeaturesExtractor.extract(calculateWordsFrequency=False)
        # test
        print(' > [testing] ==> ' + networkType + ' testing ...')
        return self.Networks[networkType].test()
