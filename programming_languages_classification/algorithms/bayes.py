# /usr/bin/env python3

import os
from sklearn import naive_bayes
from .base import _BaseAlgorithm
from sklearn import preprocessing
from utils import ConfigurationManager, FileManager
from keras.preprocessing.text import Tokenizer
import keras.preprocessing.text as kpt
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report


ESCAPED_TOKENS = ConfigurationManager.escaped_tokens
TOKENIZER_CONFIG: dict = ConfigurationManager.tokenizerConfiguration
MODEL_CONFIG: dict = {
    'max_features': 10000,
    'max_len_sequences': 500
}


class NaiveBayes(_BaseAlgorithm):

    def __init__(self):
        super().__init__()
        self.type = 'BAYES'
        self.config = MODEL_CONFIG.copy()

        self.initialize()

    def train(self):
        if os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            return self

        # preparing features
        X, languages = self.__prepareFeatures('training', False)

        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # (X, Y) creation
        Y = Y_Encoder.transform(languages)

        # prepare model
        self.__prepareModel()
        # training
        self.model.fit(X, Y)
        # export the trained model
        self.exportScikitTrainedModel()

        return self

    def test(self):
        if not os.path.exists(FileManager.getTrainedModelFileUrl(self.type)):
            raise Exception('You can\'t test a model without training it')

        # label encoder
        Y_Encoder = preprocessing.LabelEncoder()
        Y_Encoder.fit(ConfigurationManager.getLanguages())

        # preparing features
        X, languages = self.__prepareFeatures('testing', True)

        # import trained model
        self.importScikitTrainedModel()

        # make predictions
        Y_real = Y_Encoder.transform(languages)
        Y_predicted = self.model.predict(X)

        # metrics
        accuracy = accuracy_score(Y_real, Y_predicted)
        report = classification_report(Y_real, Y_predicted, target_names=Y_Encoder.classes_)
        print(' >  [BAYES]  classification report exported!')
        print(' >  [BAYES]  total accuracy = ' + str(float("{:.2f}".format(accuracy)) * 100) + '%')

        # export the classification report
        self.exportClassificationReport(str(report))

        return self

    def __prepareModel(self):
        self.model = naive_bayes.MultinomialNB()
        return self

    def __prepareFeatures(self, dataset: str, importWordsIndexes=False):
        max_features: int = self.config['max_features']
        input_length: int = self.config['max_len_sequences']

        X = []
        Y = []
        sources, languages = self.extractSources(dataset)

        wordsIndexes = {}
        if not importWordsIndexes:
            # tokenization
            tokenizer = Tokenizer(num_words=max_features, filters=TOKENIZER_CONFIG['filter'])
            tokenizer.fit_on_texts(sources)
            # export words indexes
            self.exportWordsIndexes(tokenizer.word_index)
            # set the indexes
            wordsIndexes = tokenizer.word_index
        else:
            # import and set the words indexes
            wordsIndexes = self.importWordsIndexes()

        # sort the dict by the words indexes
        wordsIndexesSortedByIndex = {k: v for k, v in sorted(wordsIndexes.items(), key=lambda item: item[1])}

        # count the occurencies of each token in the source code.
        for i, source in enumerate(sources):
            language = languages[i]
            features: list = []

            sourceTokens = set(source.split(' '))
            sourceTokensOccurencies = Counter(list(sourceTokens))
            for token, indexValue in wordsIndexesSortedByIndex.items():
                if token not in sourceTokensOccurencies:
                    features.append(0)
                else:
                    features.append(sourceTokensOccurencies[token])
            # X + Y
            X.append(features)
            Y.append(language)

        return X, Y

    #

    # @override
    def extractSources(self, dataset: str):
        return super().extractSources(dataset, 'filtered')
