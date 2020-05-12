# /usr/bin/env python3

import json
from utils import FileManager
from dataset import DatasetInstance
import keras.preprocessing.text as kpt


class _BaseAlgorithm:

    type: str = 'MISSING'
    dataset: DatasetInstance = None
    model: None
    config: dict = {}

    def initialize(self, datasetInstance: DatasetInstance):
        self.dataset = datasetInstance
        return self

    #

    def importWordsIndexes(self):
        return json.loads(FileManager.readFile(FileManager.getWordsIndexesFileUrl(self.type)))

    def exportWordsIndexes(self, indexes):
        FileManager.writeFile(FileManager.getWordsIndexesFileUrl(self.type), json.dumps(indexes))
        return self

    #

    def generateWordsIndexesForUnknownExample(self, wordsIndexes, source: str):
        wordvec = []
        max_features: int = self.config['max_features']

        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        for word in kpt.text_to_word_sequence(source):
            if word in wordsIndexes:
                if wordsIndexes[word] <= max_features:
                    wordvec.append([wordsIndexes[word]])
                else:
                    wordvec.append([0])
            else:
                wordvec.append([0])

        return wordvec
