# /usr/bin/env python3

from dataset import DatasetManager
from features import FeaturesManager
from .networks.bayes import BayesNetwork
from .networks.svm import SvmNetwork
from utils import FileManager


class ProblemSolver:

    def __init__(self):
        # tokenizer
        self.FeaturesExtractor = FeaturesManager()
        # dataset
        self.DatasetManger = DatasetManager()
        # networks
        self.Networks = {'BAYES': BayesNetwork(), 'SVM': SvmNetwork()}

    def initialize(self):
        trainingConfig = FileManager.datasets['training']
        testingConfig = FileManager.datasets['testing']
        # networks
        self.Networks['BAYES'].initialize(trainingConfig, testingConfig)
        self.Networks['SVM'].initialize(trainingConfig, testingConfig)

    def loadDataset(self):
        self.DatasetManger.load()

    def prepare(self, networkType, datasetUri):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, datasetUri)
        # features pre-processing
        print(' > [training] ==> start pre-processing ...')
        self.FeaturesExtractor.parse()
        # features extraction
        print(' > [training] ==> start features extraction ...')
        self.FeaturesExtractor.generateFeatures()

    def train(self, networkType):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, FileManager.datasets['training']['url'])
        # features pre-processing
        print(' > [training] ==> start pre-processing ...')
        self.FeaturesExtractor.parse()
        # features extraction
        print(' > [training] ==> start features extraction ...')
        self.FeaturesExtractor.generateFeatures(calculateWordsFrequency=True)
        # train
        print(' > [training] ==> ' + networkType + ' training ...')
        return self.Networks[networkType].train()

    def test(self, networkType):
        # feature extractor initialization
        self.FeaturesExtractor.initialize(networkType, FileManager.datasets['testing']['url'])
        # features pre-processing
        print(' > [testing] ==> start pre-processing ...')
        self.FeaturesExtractor.parse()
        # features extraction
        print(' > [testing] ==> start features extraction ...')
        self.FeaturesExtractor.generateFeatures(calculateWordsFrequency=False)
        # test
        print(' > [testing] ==> ' + networkType + ' testing ...')
        return self.Networks[networkType].test()
