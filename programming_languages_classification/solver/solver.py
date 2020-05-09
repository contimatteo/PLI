# /usr/bin/env python3

from dataset import DatasetManager
from features import FeaturesManager
from .algorithms.bayes import NaiveBayes
from .algorithms.cnn import CNN
from utils import FileManager


class ProblemSolver:

    DatasetManager: DatasetManager = None
    Algorithms: dict = {'CNN': None}

    def initialize(self):
        # dataset
        self.DatasetManger = DatasetManager()

        # CNN Algorithm
        self.Algorithms['CNN'] = CNN()
        self.Algorithms['CNN'].initialize(self.DatasetManger.Dataset)

        return self

    def loadDataset(self):
        self.DatasetManger.initialize().load()
        return self

    def train(self, algorithm):
        self.Algorithms[algorithm].train()
        return self

    def test(self, algorithm):
        print(' > [testing] ==> ' + algorithm + ' testing ...')
        self.Algorithms[algorithm].test()
        return self
