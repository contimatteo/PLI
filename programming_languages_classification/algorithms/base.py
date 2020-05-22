# /usr/bin/env python3

import json
from utils import FileManager
from dataset import DatasetInstance
import keras.preprocessing.text as kpt
from utils import ConfigurationManager
from keras.models import load_model
import joblib
from dataset import DatasetManager


TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration


class _BaseAlgorithm:

    def __init__(self):
        self.type: str = 'MISSING'
        self.Dataset: DatasetInstance = None
        self.model: None
        self.config: dict = {}
        self.DatasetManger = DatasetManager()

    def initialize(self):
        # load the dataset
        self.DatasetManger.initialize(self.type).load()
        # save the dataset instance
        self.Dataset = self.DatasetManger.Dataset

        return self

    #

    def importWordsIndexes(self):
        return json.loads(FileManager.readFile(FileManager.getWordsIndexesFileUrl(self.type)))

    def exportWordsIndexes(self, indexes):
        FileManager.writeFile(FileManager.getWordsIndexesFileUrl(self.type), json.dumps(indexes))
        return self

    def importKerasTrainedModel(self):
        self.model = load_model(FileManager.getTrainedModelFileUrl(self.type))
        return self

    def importScikitTrainedModel(self):
        self.model = joblib.load(FileManager.getTrainedModelFileUrl(self.type))
        return self

    def exportKerasTrainedModel(self):
        self.model.save(FileManager.getTrainedModelFileUrl(self.type))
        return self

    def exportScikitTrainedModel(self):
        joblib.dump(self.model, FileManager.getTrainedModelFileUrl(self.type))
        return self

    #

    def generateWordsIndexesForUnknownExample(self, wordsIndexes, source: str):
        wordvec = []
        max_features: int = self.config['max_features']

        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        for word in kpt.text_to_word_sequence(source, filters=TOKENIZER_CONFIG['filter']):
            if word in wordsIndexes:
                if wordsIndexes[word] <= max_features:
                    wordvec.append([wordsIndexes[word]])
                else:
                    wordvec.append([0])
            else:
                wordvec.append([0])

        return wordvec