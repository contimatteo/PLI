# /usr/bin/env python3


class DatasetInstance:

    training: dict = None
    testing: dict = None

    def initialize(self):
        self.training = {'sources': {}, 'counters': {}, 'size': 0}
        self.testing = {'sources': {}, 'counters': {}, 'size': 0}

        return self

    def __getByType(self, dataset: str):
        if dataset == 'testing':
            return self.testing

        return self.training

    ##

    def getSources(self, dataset: str = 'training'):
        return self.__getByType(dataset)['sources']

    def getCounters(self, dataset: str = 'training'):
        return self.__getByType(dataset)['counters']

    def countExamples(self, dataset: str = 'training'):
        return self.__getByType(dataset)['size']

    ##

    def addLanguage(self, dataset: str, language: str):
        if language not in self.__getByType(dataset)['sources']:
            self.__getByType(dataset)['sources'][language]: list = []
            self.__getByType(dataset)['counters'][language]: int = 0

        return self

    def addExample(self, dataset: str, language: str, example: dict):
        self.__getByType(dataset)['sources'][language].append(example)
        self.__getByType(dataset)['counters'][language] += 1
        self.__getByType(dataset)['size'] += 1

        return self
