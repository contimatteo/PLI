# /usr/bin/env python3

from dataset import DatasetManager
from solver.svm import SVM
from solver.cnn import CNN
from solver.bayes import NaiveBayes


class ProblemSolver:

    DatasetManager: DatasetManager = None
    Algorithms: dict = {'CNN': None, 'SVM': None, 'BAYES': None}

    def initialize(self):
        # dataset
        self.DatasetManger = DatasetManager()

        # CNN Algorithm
        self.Algorithms['CNN'] = CNN()
        self.Algorithms['CNN'].initialize(self.DatasetManger.Dataset)

        # SVM Algorithm
        self.Algorithms['SVM'] = SVM()
        self.Algorithms['SVM'].initialize(self.DatasetManger.Dataset)

        # BAYES Algorithm
        self.Algorithms['BAYES'] = NaiveBayes()
        self.Algorithms['BAYES'].initialize(self.DatasetManger.Dataset)

        return self

    def loadDataset(self):
        self.DatasetManger.initialize().load()
        return self

    def train(self, algorithm):
        self.Algorithms[algorithm].train()
        return self

    def test(self, algorithm):
        self.Algorithms[algorithm].test()
        return self
